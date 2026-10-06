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

WHAT THIS DOES NOT DO: database/auth/CORS architecture (later stage);
frontend hosting (static file served separately).
"""

import asyncio
import atexit
import tempfile
import threading
import time
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bridge_schema import (
    API_VERSION, MAX_UPLOAD_BYTES, ALLOWED_CONTENT_PREFIX,
    to_null_number as safe_num, to_null_str as safe_str,
    to_str_list as safe_str_list, to_null_bool as safe_bool,
    check_size_ok,
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

# FastAPI 413 for oversized bodies is handled manually below (streaming
# read with early abort) so an oversized payload is never fully retained.
MAX_UPLOAD_READ = MAX_UPLOAD_BYTES + 1

app = FastAPI(title="NetraSetu screening bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # SECURITY-GATED: tighten to the console origin before any real deployment
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_engine = None
_engine_lock = threading.Lock()  # serializes ALL engine screening calls


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
    try:
        yield get_engine()
    except HTTPException:
        raise
    except Exception as e:
        _engine_invalidate()
        raise HTTPException(status_code=503, detail=f"MATLAB engine failed: {e}")


def _screen_once(tmp_path: str) -> dict:
    with _locked_engine() as eng:
        m_result = eng.screenOneImage(tmp_path, nargout=1)
        return matlab_result_to_dict(m_result)


def _do_screen(tmp_path: str) -> dict:
    """Runs one screening on the worker thread with crash recovery: on
    failure the stale engine is invalidated and the call retried exactly
    once on a fresh engine. Lock MUST be held by the caller (released in
    finally at the call site)."""
    try:
        return _screen_once(tmp_path)
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

    evidence = m_result["evidence"]
    if isinstance(evidence, str):
        evidence_list = [evidence]
    else:
        try:
            evidence_list = [str(x) for x in list(evidence)]
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
        "status": str(m_result["status"]),
        "errorMessage": str(m_result["errorMessage"]),
        "focus": safe_num(m_result["focus"]),
        "entropy": safe_num(m_result["entropy"]),
        "roiPassed": safe_bool(m_result["roiPassed"]),
        "dl": safe_num(m_result["dl"]),
        "conf": safe_num(m_result["conf"]),
        "rule": safe_num(m_result["rule"]),
        "evidence": evidence_list,
        "nv": safe_bool(m_result["nv"]),
        "gradCamOnDisc": safe_bool(m_result["gradCamOnDisc"]),
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


@app.get("/health")
def health():
    """Bridge process alive + contract version. Does NOT imply MATLAB or
    model readiness - use /health/deep for that."""
    return {"status": "ok", "apiVersion": API_VERSION,
            "engine_started": _engine is not None}


@app.get("/health/deep")
def health_deep():
    """Readiness: engine startable, project files on path, model assets
    present by NAME only (no filesystem details leak). MATLAB-GATED."""
    acquired = _engine_lock.acquire(blocking=False)
    if not acquired:
        return JSONResponse(status_code=429, content={"status": "busy"})
    try:
        try:
            eng = get_engine()
            eng.eval("1+1;", nargout=0)
            engine_ok = True
        except Exception as e:
            _engine_invalidate()
            return JSONResponse(status_code=503,
                                content={"status": "engine-unavailable", "detail": str(e)[:200]})
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
        return {"status": "ready" if ready else "degraded",
                "apiVersion": API_VERSION, "engine": engine_ok,
                "projectFiles": on_path, "models": assets}
    finally:
        _engine_lock.release()


@app.post("/screen")
async def screen(image: UploadFile = File(...)):
    """Screens one uploaded fundus photo. Serialized: concurrent callers
    get HTTP 429 immediately (no unsafe concurrent engine use, no event
    loop blocking - the engine call runs in a worker thread)."""
    t0 = time.monotonic()
    if not image.content_type or not image.content_type.startswith(ALLOWED_CONTENT_PREFIX):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")
    # Streaming size enforcement (never rely on Content-Length alone;
    # oversized payload is not retained).
    data = bytearray()
    while True:
        chunk = await image.read(1024 * 1024)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_UPLOAD_BYTES:
            print("screen rejected: oversized upload", flush=True)
            raise HTTPException(status_code=413, detail="Image exceeds 15 MB limit.")
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    if len(suffix) > 8 or "/" in suffix or "\\" in suffix:
        suffix = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    del data

    acquired = _engine_lock.acquire(blocking=False)
    if not acquired:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        print("screen rejected: engine busy", flush=True)
        raise HTTPException(status_code=429, detail="Screening engine busy; retry shortly.")
    try:
        import asyncio as _asyncio
        result = await _asyncio.to_thread(_do_screen, tmp_path)
        dt = (time.monotonic() - t0) * 1000
        print(f"screen ok status={result.get('status')} dl={result.get('dl')} "
              f"rule={result.get('rule')} ms={dt:.0f}", flush=True)
        return JSONResponse(result)
    except HTTPException as e:
        print(f"screen engine-failure: {e.status_code}", flush=True)
        raise
    except Exception as e:
        print(f"screen failed: {type(e).__name__}", flush=True)
        raise HTTPException(status_code=500, detail="Screening failed.")
    finally:
        _engine_lock.release()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
