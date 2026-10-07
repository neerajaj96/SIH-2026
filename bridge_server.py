"""
bridge_server.py — REST bridge between a web/mobile frontend and the
real MATLAB pipeline, via MATLAB Engine for Python.

STATUS: written against the documented MATLAB Engine for Python API,
but the LIVE engine path is MATLAB-GATED (no MATLAB here): Python
syntax compiles, contract tests pass on stubs, but no real engine call
has executed in this workspace. See STAGE7_NETRASETU_HANDOFF.md.

CONCURRENCY POLICY (explicit): the MATLAB Engine call runs on ONE
dedicated worker thread behind a threading lock - screenings are
strictly serialized. Concurrent MATLAB Engine screening is NOT assumed
safe (unverified). A second concurrent request gets HTTP 429
{"status":"busy"} immediately; the event loop is never blocked because
the blocking engine call runs via asyncio.to_thread, never inline in
an async route.

WHAT THIS DOES NOT DO: database/frontend hosting (static file served
separately). Auth (fail-closed shared key), allowlist CORS, sliding-window
rate limiting, and upload validation ARE implemented here (Stage 9);
see STAGE9_SECURITY_HANDOFF.md.
"""

import asyncio
import atexit
import hashlib
import hmac
import tempfile
import threading
import time
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bridge_schema import (
    API_VERSION, MAX_UPLOAD_BYTES, ALLOWED_CONTENT_PREFIX,
    API_KEY_ENV, CORS_ORIGINS_ENV, RATE_LIMIT_ENV, TRUSTED_PROXY_ENV,
    DEFAULT_RATE_LIMIT_PER_MIN, MAX_IMAGE_DIMENSION,
    SAFE_ERROR_DETAILS,
    to_null_number as safe_num, to_null_str as safe_str,
    to_str_list as safe_str_list, to_null_bool as safe_bool,
    check_size_ok, detect_image_kind, probe_jpeg_dimensions,
    probe_png_dimensions, valid_request_id, new_request_id,
)

try:
    import matlab.engine
except ImportError as e:
    raise SystemExit(
        "matlab.engine not found. Install it from your MATLAB installation: "
        "cd \"<matlabroot>/extern/engines/python\" && python setup.py install "
        "(or `pip install matlabengine` if a matching version is on PyPI for "
        "your MATLAB release)."
    ) from e

PROJECT_DIR = os.environ.get("SIH_PROJECT_DIR", str(Path(__file__).resolve().parent))

# Isolated temp directory for uploads (created 0700 best-effort; falls
# back to the platform temp dir when creation fails, never crashes boot).
_BRIDGE_TMPDIR = os.environ.get("SIH_TMPDIR", os.path.join(tempfile.gettempdir(), "netrasetu-bridge"))
try:
    os.makedirs(_BRIDGE_TMPDIR, mode=0o700, exist_ok=True)
except Exception:
    _BRIDGE_TMPDIR = tempfile.gettempdir()


def _secure_unlink(path: str):
    """Idempotent temp cleanup (missing/already-removed is not an error)."""
    try:
        os.unlink(path)
    except (OSError, TypeError):
        pass


def _secure_chmod(path: str):
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass

# ---- Production configuration (fail-closed, validated at startup) ----
API_KEY = os.environ.get(API_KEY_ENV, "")
CORS_ORIGINS = [o.strip() for o in os.environ.get(CORS_ORIGINS_ENV, "http://localhost:8000").split(",") if o.strip()]
try:
    RATE_LIMIT_PER_MIN = max(1, int(os.environ.get(RATE_LIMIT_ENV, str(DEFAULT_RATE_LIMIT_PER_MIN))))
except (TypeError, ValueError):
    RATE_LIMIT_PER_MIN = DEFAULT_RATE_LIMIT_PER_MIN
TRUSTED_PROXY = os.environ.get(TRUSTED_PROXY_ENV, "").strip()

if not API_KEY:
    # Fail-closed contract: /screen refuses everything until an operator
    # configures SIH_API_KEY (development uses an explicit dev key, never
    # an anonymous bypass). Startup continues so /health explains state.
    print("SECURITY: SIH_API_KEY unset - /screen will fail closed (503) until configured.", flush=True)
if not CORS_ORIGINS:
    raise SystemExit("SECURITY: SIH_CORS_ORIGINS resolved empty - refusing to start with no allowed origins.")
for _o in CORS_ORIGINS:
    if _o != "*" and not (_o.startswith("http://") or _o.startswith("https://")):
        raise SystemExit(f"SECURITY: SIH_CORS_ORIGINS entry invalid (must be http(s):// or unset): {_o!r}")
if "*" in CORS_ORIGINS:
    raise SystemExit("SECURITY: SIH_CORS_ORIGINS must not contain '*'.")
if not os.path.isdir(PROJECT_DIR):
    print(f"CONFIG: SIH_PROJECT_DIR does not exist ({PROJECT_DIR}) - engine start will fail; "
          f"/health/deep will report degraded.", flush=True)
if TRUSTED_PROXY and len(TRUSTED_PROXY) > 256:
    raise SystemExit("SECURITY: SIH_TRUSTED_PROXY value too long.")
if not (1 <= RATE_LIMIT_PER_MIN <= 1000):
    raise SystemExit("CONFIG: SIH_RATE_LIMIT_PER_MIN must be within 1..1000.")

# Per-IP sliding-window rate limiter (in-memory; single process).
# Trust model: direct peer address ONLY, unless SIH_TRUSTED_PROXY names an
# explicitly configured proxy whose X-Forwarded-For we then honor (first
# entry). Blindly trusting X-Forwarded-For lets clients spoof identity.
# Buckets are bounded: idle entries expire after the window and the table
# is capped so an IP scan cannot grow memory without bound.
_rate_buckets: dict = {}
_rate_lock = threading.Lock()
_RATE_BUCKET_CAP = 5000


def _client_ip(request: Request) -> str:
    if TRUSTED_PROXY:
        fwd = request.headers.get("x-forwarded-for", "")
        first = (fwd.split(",")[0] if fwd else "").strip()
        if first:
            return "proxy:" + first[:64]
    try:
        return str(request.client.host) if request.client else "unknown"
    except Exception:
        return "unknown"


def _rate_allowed(ip: str) -> bool:
    now = time.monotonic()
    window = 60.0
    with _rate_lock:
        # Opportunistic reap: drop expired buckets when the table grows.
        if len(_rate_buckets) > _RATE_BUCKET_CAP:
            for k in [k for k, v in _rate_buckets.items()
                      if not v or now - v[-1] >= window]:
                _rate_buckets.pop(k, None)
            # Hard cap: evict oldest entries if still overfull (scan defense).
            while len(_rate_buckets) > _RATE_BUCKET_CAP:
                _rate_buckets.pop(next(iter(_rate_buckets)), None)
        hits = _rate_buckets.get(ip, [])
        hits = [t for t in hits if now - t < window]
        if len(hits) >= RATE_LIMIT_PER_MIN:
            _rate_buckets[ip] = hits
            return False
        _rate_buckets[ip] = hits + [now]
        return True


def _request_id(request: Request) -> str:
    """Accepts a client X-Request-ID only if allowlisted; else mints one.
    Prevents control-character / log-injection via crafted IDs."""
    v = request.headers.get("x-request-id", "")
    if valid_request_id(v):
        return v
    return new_request_id()


def _log(event: str, **fields):
    """Structured operational log: event + request id + route + status +
    durations + categories. NEVER image bytes, filenames, patient data,
    evidence text, or exception internals."""
    safe = {k: v for k, v in fields.items()
            if k in ("rid", "route", "status", "ms", "category", "size", "detail")}
    print(f"evt={event} " + " ".join(f"{k}={v}" for k, v in safe.items()), flush=True)


def _require_api_key(request: Request):
    """Fail-closed gate: no key configured -> 503; wrong/missing key -> 401.
    Constant-time comparison against the configured key."""
    if not API_KEY:
        raise HTTPException(status_code=503, detail=SAFE_ERROR_DETAILS["unconfigured"])
    got = request.headers.get("x-api-key", "")
    if not got or not hmac.compare_digest(got, API_KEY):
        raise HTTPException(status_code=401, detail=SAFE_ERROR_DETAILS["unauth"])

# FastAPI 413 for oversized bodies is handled manually below (streaming
# read with early abort) so an oversized payload is never fully retained.
MAX_UPLOAD_READ = MAX_UPLOAD_BYTES + 1

app = FastAPI(title="NetraSetu screening bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # explicit allowlist (default localhost dev); "*" forbidden here
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "X-Request-ID", "Content-Type"],
    expose_headers=["X-Request-ID"],  # browser JS must read echoed ID under CORS
    allow_credentials=False,
    max_age=600,
)

_engine = None
_engine_lock = threading.Lock()  # serializes ALL engine screening calls


class BridgeSerializationError(Exception):
    """matlab_result_to_dict could not interpret the engine result.

    The engine itself may be healthy; callers MUST NOT invalidate/restart
    it for this class. Mapped to HTTP 500 (failed), never 503 (engine).
    """


def _engine_invalidate():
    """Clears a dead engine reference (lock must be held by caller)."""
    global _engine
    old = _engine
    _engine = None
    if old is not None:
        try:
            old.quit()
        except Exception:
            pass


def _engine_start_locked():
    """Starts the engine; caller must hold _engine_lock."""
    global _engine
    print("Starting MATLAB engine (takes a few seconds)...", flush=True)
    _engine = matlab.engine.start_matlab()
    _engine.cd(PROJECT_DIR)
    _engine.addpath(PROJECT_DIR)
    print(f"MATLAB engine ready, cwd = {PROJECT_DIR}", flush=True)
    return _engine


def get_engine():
    """Returns the shared engine, starting it once. NOT thread-safe on
    its own - screening callers must go through screening_lock_held()."""
    global _engine
    if _engine is None:
        _engine = _engine_start_locked()
    return _engine


from contextlib import contextmanager


@contextmanager
def _locked_engine():
    """Yields the live engine once. Lock MUST be held by the caller.
    Raises HTTPException(503) if the engine cannot be started. Never
    yields twice and never returns a dead object."""
    if not _engine_lock.locked():
        raise BridgeSerializationError(
            "internal lock contract violated: _locked_engine without _engine_lock")
    try:
        yield get_engine()
    except HTTPException:
        raise
    except BridgeSerializationError:
        raise
    except Exception as e:
        _engine_invalidate()
        raise HTTPException(status_code=503, detail=f"MATLAB engine failed: {e}")


def _screen_once(tmp_path: str) -> dict:
    with _locked_engine() as eng:
        try:
            m_result = eng.screenOneImage(tmp_path, nargout=1)
        except HTTPException:
            raise
        except Exception as e:
            # Engine call itself failed -> engine suspect (invalidate+503).
            _engine_invalidate()
            raise HTTPException(status_code=503, detail=f"MATLAB engine failed: {e}")
        try:
            return matlab_result_to_dict(m_result)
        except BridgeSerializationError:
            raise
        except HTTPException:
            raise
        except Exception as e:
            # Conversion failed on a live engine -> healthy engine, bad
            # payload shape. Never invalidate; surface as 500 downstream.
            raise BridgeSerializationError(f"result conversion failed: {e}")


def _do_screen(tmp_path: str) -> dict:
    """Runs one screening on the worker thread with crash recovery: on
    ENGINE failure the stale engine is invalidated and the call retried
    exactly once on a fresh engine. Serialization failures are NOT
    retried (retrying a healthy engine cannot fix a bad payload shape).
    Lock MUST be held by the caller (released in finally at call site)."""
    try:
        return _screen_once(tmp_path)
    except BridgeSerializationError:
        raise
    except HTTPException:
        pass  # state already invalidated; fall through to the single retry
    _engine_invalidate()
    return _screen_once(tmp_path)  # raises 503 itself if still broken


@atexit.register
def _shutdown_engine():
    global _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        _engine = None


def _nested(struct_like, key, default=None):
    try:
        return struct_like[key]
    except (KeyError, TypeError, IndexError):
        return default


def _struct_to_dict(s, keys):
    out = {}
    for k in keys:
        try:
            out[k] = s[k]
        except (KeyError, TypeError, IndexError):
            out[k] = None
    return out


def matlab_result_to_dict(m_result) -> dict:
    """Converts screenOneImage.m's returned struct into JSON-safe dict.

    Schema version: bridge_schema.API_VERSION. Legacy flat keys are
    frozen compatibility aliases; nested Stage-7 groups are the
    canonical semantic source (both computed once in MATLAB - this
    function only converts, never recomputes).

    MATLAB-GATED: struct/cell conversion assumptions below are
    documented, not yet executed against a live engine here.
    """
    def safe_get(key, default=None):
        try:
            return m_result[key]
        except (KeyError, TypeError, IndexError):
            return default

    def need(key):
        # Legacy keys are frozen-required: absence is a conversion defect
        # (HTTP 500), never engine failure (HTTP 503).
        try:
            return m_result[key]
        except (KeyError, TypeError, IndexError) as e:
            raise BridgeSerializationError(f"missing legacy key: {key}") from e

    try:
        evidence_raw = m_result["evidence"]
    except (KeyError, TypeError, IndexError) as e:
        raise BridgeSerializationError("missing legacy key: evidence") from e
    if isinstance(evidence_raw, str):
        evidence_list = [evidence_raw]
    else:
        try:
            evidence_list = [str(x) for x in list(evidence_raw)]
        except TypeError:
            evidence_list = []

    dis_raw = safe_get("disagreement")
    if dis_raw is None:
        disagreement = None
    else:
        d = _struct_to_dict(dis_raw, ("clinicalEvidenceStatus", "gradeRelationship",
                                      "confidenceStatus", "explanationStatus",
                                      "escalate", "reasons", "configVersion"))
        d["reasons"] = safe_str_list(d.get("reasons"))
        d["escalate"] = safe_bool(d.get("escalate"))
        disagreement = d

    return {
        "apiVersion": API_VERSION,
        # Legacy keys (frozen compatibility aliases).
        "status": str(need("status")),
        "errorMessage": str(need("errorMessage")),
        "focus": safe_num(need("focus")),
        "entropy": safe_num(need("entropy")),
        "roiPassed": safe_bool(need("roiPassed")),
        "dl": safe_num(need("dl")),
        "conf": safe_num(need("conf")),
        "rule": safe_num(need("rule")),
        "evidence": evidence_list,
        "nv": safe_bool(need("nv")),
        "nvStatus": safe_str(safe_get("nvStatus")),
        "gradCamOnDisc": safe_bool(need("gradCamOnDisc")),
        # Canonical nested groups (nullable; null when MATLAB predates them).
        "quality": {
            "decision": safe_str(safe_get("qualityDecision")),
            "reasons": safe_str_list(safe_get("qualityReasons", [])),
            "guidance": safe_str_list(safe_get("qualityGuidance", [])),
            "calibrated": safe_bool(safe_get("qualityCalibrated")),
        },
        "clinical": {
            "ruleStatus": safe_str(safe_get("ruleStatus")),
        },
        "explanation": {
            "status": safe_str(safe_get("explainStatus")),
        },
        "disagreement": disagreement,
        "temperature": {
            "value": safe_num(safe_get("temperatureT")),
            "state": safe_str(safe_get("temperatureState")),
        },
        "landmarkStatus": {
            "od": safe_str(safe_get("odValidity")),
            "fovea": safe_str(safe_get("foveaValidity")),
            "quadrant": safe_str(safe_get("quadrantValid")),
        },
        "modelPresent": safe_bool(safe_get("modelPresent")),
    }


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    """Assigns/validates the request ID before routing so EVERY response
    (including auth/validation rejections raised before the route body
    runs) can echo it for operator correlation."""
    rid = _request_id(request)
    request.state.rid = rid
    try:
        response = await call_next(request)
    except HTTPException:
        raise
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(HTTPException)
async def _http_error_shape(request: Request, exc: HTTPException):
    """Safe error envelope: numeric code + allowlisted message + request
    ID. Never traceback text, paths, or engine internals."""
    rid = getattr(getattr(request, "state", None), "rid", None) or new_request_id()
    headers = {"X-Request-ID": rid}
    if exc.status_code == 429:
        headers["Retry-After"] = "5"
    return JSONResponse(status_code=exc.status_code,
                        content={"detail": str(exc.detail)[:300], "requestId": rid},
                        headers=headers)


@app.exception_handler(Exception)
async def _unhandled_error_shape(request: Request, exc: Exception):
    """Preserves request-ID correlation for non-HTTP failures."""
    rid = getattr(getattr(request, "state", None), "rid", None) or new_request_id()
    _log("unhandled", rid=rid, route=str(getattr(request, "url", ""))[:120],
         status="error", category="error")
    return JSONResponse(status_code=500,
                        content={"detail": SAFE_ERROR_DETAILS["failed"], "requestId": rid},
                        headers={"X-Request-ID": rid})


@app.get("/health")
def health(request: Request):
    """Bridge process alive + contract version. Does NOT imply MATLAB or
    model readiness - use /health/deep for that. Unauthenticated (load
    balancers); exposes no model/filesystem state."""
    return {"status": "ok", "apiVersion": API_VERSION,
            "engine_started": _engine is not None,
            "requestId": _request_id(request)}


@app.get("/health/deep")
def health_deep(request: Request):
    """Readiness: engine startable, project files on path, model assets
    present by NAME only (no filesystem details leak). Unauthenticated
    (readiness only, no screening). MATLAB-GATED."""
    rid = _request_id(request)
    acquired = _engine_lock.acquire(blocking=False)
    if not acquired:
        return JSONResponse(status_code=429, content={"status": "busy", "requestId": rid},
                            headers={"X-Request-ID": rid, "Retry-After": "5"})
    try:
        try:
            eng = get_engine()
            eng.eval("1+1;", nargout=0)
            engine_ok = True
        except Exception:
            _engine_invalidate()
            _log("health.deep", rid=rid, route="/health/deep", status="engine-unavailable")
            return JSONResponse(status_code=503,
                                content={"status": "engine-unavailable", "requestId": rid})
        try:
            on_path = bool(eng.eval("which('screenOneImage')", nargout=1))
        except Exception:
            on_path = False
        assets = {}
        for name in ("unet_Vessels.mat", "unet_MicroaneurysmsHemorrhages.mat",
                     "unet_Exudates.mat", "trained_dr_grader.mat",
                     "calibrated_temperature.mat"):
            assets[name] = os.path.isfile(os.path.join(PROJECT_DIR, name))
        ready = engine_ok and on_path and all(assets[k] for k in assets if k != "calibrated_temperature.mat")
        config_valid = bool(os.path.isdir(PROJECT_DIR)) and bool(CORS_ORIGINS)
        return {"status": "ready" if ready else "degraded", "requestId": rid,
                "apiVersion": API_VERSION, "engine": engine_ok,
                "projectFiles": on_path, "models": assets,
                "configValid": config_valid}
    finally:
        _engine_lock.release()


@app.post("/screen")
async def screen(request: Request, image: UploadFile = File(...)):
    """Screens one uploaded fundus photo. Fail-closed auth, rate limits,
    serialized engine use (HTTP 429 when busy - the engine call runs in a
    worker thread, never blocking the event loop)."""
    t0 = time.monotonic()
    rid = _request_id(request)
    ip = _client_ip(request)

    def _elapsed_ms() -> str:
        return f"{(time.monotonic() - t0) * 1000:.0f}"

    if not _rate_allowed(ip):
        _log("screen", rid=rid, route="/screen", status="rate-limited", ms=_elapsed_ms())
        raise HTTPException(status_code=429, detail=SAFE_ERROR_DETAILS["ratelimit"])
    _require_api_key(request)  # 401 wrong/missing key, 503 unconfigured - fail-closed, no bypass
    if not image.content_type or not image.content_type.startswith(ALLOWED_CONTENT_PREFIX):
        _log("screen", rid=rid, route="/screen", status="rejected", category="badtype", ms=_elapsed_ms())
        raise HTTPException(status_code=400, detail=SAFE_ERROR_DETAILS["badtype"])
    # Streaming size enforcement (never rely on Content-Length alone;
    # works with missing/chunked lengths; oversized payload not retained).
    data = bytearray()
    try:
        while True:
            chunk = await image.read(1024 * 1024)
            if not chunk:
                break
            data += chunk
            if len(data) > MAX_UPLOAD_BYTES:
                _log("screen", rid=rid, route="/screen", status="rejected", category="oversize",
                     ms=_elapsed_ms())
                raise HTTPException(status_code=413, detail=SAFE_ERROR_DETAILS["oversize"])
    finally:
        try:
            await image.close()
        except Exception:
            pass
    if not check_size_ok(len(data)) or len(data) == 0:
        _log("screen", rid=rid, route="/screen", status="rejected", category="empty", ms=_elapsed_ms())
        raise HTTPException(status_code=400, detail=SAFE_ERROR_DETAILS["badtype"])
    # Signature + structural validation (provable invalidity only: wrong
    # magic or impossible dimensions reject here; well-formed-but-
    # undecodable files remain decoder-gated downstream, reported as
    # input failures - never silently graded).
    head = bytes(data[:64])
    kind = detect_image_kind(head)
    if kind is None:
        _log("screen", rid=rid, route="/screen", status="rejected", category="badmagic", ms=_elapsed_ms())
        raise HTTPException(status_code=400, detail=SAFE_ERROR_DETAILS["badmagic"])
    dims = probe_jpeg_dimensions(head) if kind == "jpeg" else (
        probe_png_dimensions(head) if kind == "png" else None)
    if dims is not None and (dims[0] <= 0 or dims[1] <= 0
                             or dims[0] > MAX_IMAGE_DIMENSION or dims[1] > MAX_IMAGE_DIMENSION):
        _log("screen", rid=rid, route="/screen", status="rejected", category="baddims", ms=_elapsed_ms())
        raise HTTPException(status_code=400, detail=SAFE_ERROR_DETAILS["baddims"])
    # Capability-gated deep decode check: when Pillow is installed (not a
    # mandatory dependency), verify actual pixels/dimensions before MATLAB
    # (catches header-spoofed decompression bombs). Absent Pillow ->
    # header-only validation above (honest, documented).
    try:
        from PIL import Image as _PILImage  # type: ignore
        import io as _io
        try:
            with _PILImage.open(_io.BytesIO(bytes(data))) as _im:
                _im.verify()
            with _PILImage.open(_io.BytesIO(bytes(data))) as _im2:
                _w, _h = _im2.size
                if _w <= 0 or _h <= 0 or _w > MAX_IMAGE_DIMENSION or _h > MAX_IMAGE_DIMENSION:
                    _log("screen", rid=rid, route="/screen", status="rejected", category="baddims",
                         ms=_elapsed_ms())
                    raise HTTPException(status_code=400, detail=SAFE_ERROR_DETAILS["baddims"])
        except HTTPException:
            raise
        except Exception:
            pass  # undecodable here -> decoder-gated downstream, not a silent grade
    except ImportError:
        pass
    size = len(data)
    # Extension is cosmetic only (routing/decoding never trusts it);
    # traversal-hardened to a fixed allowlist shape.
    suffix = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png",
              ".gif": ".gif", ".bmp": ".bmp"}.get(
        (Path(image.filename or "upload.jpg").suffix or ".jpg").lower(), ".jpg")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False,
                                     dir=_BRIDGE_TMPDIR) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    _secure_chmod(tmp_path)
    del data

    acquired = _engine_lock.acquire(blocking=False)
    if not acquired:
        _secure_unlink(tmp_path)
        _log("screen", rid=rid, route="/screen", status="busy", size=size, ms=_elapsed_ms())
        raise HTTPException(status_code=429, detail=SAFE_ERROR_DETAILS["busy"])
    try:
        import asyncio as _asyncio
        t_engine = time.monotonic()
        result = await _asyncio.to_thread(_do_screen, tmp_path)
        dt = (time.monotonic() - t0) * 1000
        engine_ms = (time.monotonic() - t_engine) * 1000
        _log("screen", rid=rid, route="/screen", status=result.get("status"),
             category="ok", size=size, ms=f"{dt:.0f}", detail=f"engine_ms={engine_ms:.0f}")
        result["requestId"] = rid
        return JSONResponse(result)
    except BridgeSerializationError as e:
        _log("screen", rid=rid, route="/screen", status="error",
             category="serialization", size=size, ms=_elapsed_ms())
        raise HTTPException(status_code=500, detail=SAFE_ERROR_DETAILS["failed"])
    except HTTPException as e:
        _log("screen", rid=rid, route="/screen", status="engine-failure",
             category="error", size=size, ms=_elapsed_ms())
        # Engine-failure detail stays server-side; client gets the safe code.
        raise HTTPException(status_code=e.status_code, detail=SAFE_ERROR_DETAILS.get(
            "engine", "MATLAB engine unavailable."))
    except Exception:
        _log("screen", rid=rid, route="/screen", status="error", category="error", size=size,
             ms=_elapsed_ms())
        raise HTTPException(status_code=500, detail=SAFE_ERROR_DETAILS["failed"])
    finally:
        _engine_lock.release()
        _secure_unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
