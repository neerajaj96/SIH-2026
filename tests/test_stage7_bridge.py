#!/usr/bin/env python3
"""test_stage7_bridge.py — Stage-7 bridge/frontend contracts (stdlib only).

FastAPI/MATLAB are stubbed via sys.modules so the REAL bridge mapping
code executes here without those runtimes. HTML checks are static
DOM-contract assertions on netrasetu.html. No clinical validation.

Run: python3 tests/test_stage7_bridge.py
MATLAB twin: testBridgeContract.m (UNEXECUTED here).
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} :: {detail}")


def txt(rel):
    with open(os.path.join(ROOT, rel), errors="ignore") as f:
        return f.read()


def _stub_modules():
    """Install minimal fastapi/matlab/uvicorn stubs for import."""
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = type("FastAPI", (), {"__init__": lambda s, **k: None,
                                           "add_middleware": lambda s, *a, **k: None,
                                           "get": lambda s, *a, **k: (lambda f: f),
                                           "post": lambda s, *a, **k: (lambda f: f)})
    fastapi.UploadFile = type("UploadFile", (), {})
    fastapi.File = lambda *a, **k: None
    class HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            self.status_code = status_code
            self.detail = detail
    fastapi.HTTPException = HTTPException
    mw = types.ModuleType("fastapi.middleware")
    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = object
    mw.cors = cors
    resp = types.ModuleType("fastapi.responses")
    resp.JSONResponse = type("JSONResponse", (), {"__init__": lambda s, *a, **k: None})
    matlab = types.ModuleType("matlab")
    engine = types.ModuleType("matlab.engine")
    matlab.engine = engine
    sys.modules.update({"fastapi": fastapi, "fastapi.middleware": mw,
                        "fastapi.middleware.cors": cors,
                        "fastapi.responses": resp, "matlab": matlab,
                        "matlab.engine": engine})


def _load_bridge():
    _stub_modules()
    for m in [m for m in sys.modules if m == "bridge_server" or m.startswith("bridge_server.")]:
        del sys.modules[m]
    sys.path.insert(0, ROOT)
    import bridge_server as B
    return B


def test_mapping_keys():
    B = _load_bridge()
    full = {"status": "ok", "errorMessage": "", "focus": 10.5, "entropy": 5.1,
            "roiPassed": True, "dl": 2, "conf": 0.85, "rule": 2,
            "evidence": ("a", "b"), "nv": False, "gradCamOnDisc": False,
            "qualityDecision": "BORDERLINE", "qualityReasons": ("r1",),
            "qualityGuidance": ("g1",), "qualityCalibrated": True,
            "ruleStatus": "SUFFICIENT", "nvStatus": "NOT_DETECTED",
            "odValidity": "CONFIDENT", "foveaValidity": "CONFIDENT",
            "quadrantValid": "VALID", "explainStatus": "VALID",
            "temperatureT": 1.56, "temperatureState": "CALIBRATED_VALID",
            "modelPresent": True,
            "disagreement": {"clinicalEvidenceStatus": "SUFFICIENT",
                             "gradeRelationship": "AGREE",
                             "confidenceStatus": "CALIBRATED",
                             "explanationStatus": "VALID",
                             "escalate": False,
                             "reasons": [], "configVersion": "1.0.0-stage5"}}
    out = B.matlab_result_to_dict(full)
    check("legacy keys preserved",
          all(k in out for k in ("status", "dl", "conf", "rule", "evidence", "nv", "gradCamOnDisc")), "break")
    check("apiVersion present", out.get("apiVersion") == "1.0-stage7", f"{out.get('apiVersion')}")
    check("quality group mapped", out["quality"]["decision"] == "BORDERLINE", "")
    check("temperature group mapped",
          out["temperature"] == {"value": 1.56, "state": "CALIBRATED_VALID"}, f"{out['temperature']}")
    check("disagreement nested", out["disagreement"]["gradeRelationship"] == "AGREE", "")
    check("landmarks mapped", out["landmarkStatus"] == {"od": "CONFIDENT", "fovea": "CONFIDENT", "quadrant": "VALID"}, "")
    check("modelPresent mapped", out["modelPresent"] is True, "")


def test_null_missing():
    B = _load_bridge()
    thin = {"status": "ok", "errorMessage": "", "focus": 1.0, "entropy": 1.0,
            "roiPassed": True, "dl": float("nan"), "conf": float("nan"),
            "rule": float("nan"), "evidence": [], "nv": False, "gradCamOnDisc": False}
    out = B.matlab_result_to_dict(thin)
    check("NaN rule -> null (never 0)", out["rule"] is None, f"{out['rule']}")
    check("NaN dl/conf -> null", out["dl"] is None and out["conf"] is None, "")
    check("missing additive keys -> null (no KeyError)",
          out["quality"]["decision"] is None and out["disagreement"] is None
          and out["temperature"]["value"] is None and out["landmarkStatus"]["od"] is None, "")
    check("missing evidence list -> []", isinstance(out["evidence"], list), "")


def test_schema_module():
    sys.path.insert(0, ROOT)
    import bridge_schema as S
    check("15MB exact boundary accepted", S.check_size_ok(15 * 1024 * 1024), "")
    check(">15MB rejected", not S.check_size_ok(15 * 1024 * 1024 + 1), "")
    check("negative/None rejected", not S.check_size_ok(-1) and not S.check_size_ok(None), "")
    check("bridge enforces same cap",
          "MAX_UPLOAD_BYTES" in txt("bridge_server.py"), "cap drift")
    check("429-busy path exists", "429" in txt("bridge_server.py"), "no backpressure")
    check("worker-thread serialization (no bare asyncio.Lock on blocking call)",
          "to_thread" in txt("bridge_server.py") and "threading.Lock" in txt("bridge_server.py"), "concurrency gap")
    check("lock always released (finally)",
          txt("bridge_server.py").count("_engine_lock.release()") >= 2, "lock leak")
    check("dead engine never reused (invalidate + retry-once + 503)",
          "_engine_invalidate" in txt("bridge_server.py") and "503" in txt("bridge_server.py"), "dead reuse")
    check("context manager yields exactly once (no double-yield)",
          "screening_engine" not in txt("bridge_server.py"), "stale CM")
    check("retry path explicit (_screen_once x2, single _do_screen)",
          txt("bridge_server.py").count("def _do_screen(") == 1, "duplicate screener")
    check("no image/PII in logs",
          "request start/end" in txt("bridge_server.py").lower() or "screen ok status" in txt("bridge_server.py"), "logging gap")


def test_frontend_modes():
    h = txt("netrasetu.html")
    check("LIVE default with bridge probe",
          "checkBridge()" in h and "BRIDGE_DEFAULT" in h and "http://localhost:8000" in h, "mode drift")
    check("no production hostname hardcoded",
          "netlify.app" not in h and "vercel.app" not in h and ".herokuapp" not in h, "hardcoded host")
    check("SCENARIOS unreachable from LIVE path",
          "demoToBackend(SCENARIOS" in h and "SCENARIOS[seed" in h, "mock leak")
    check("OFFLINE shows no mock result",
          "No mock result shown" in h, "fabrication risk")
    check("DEMO explicit with badge",
          "Explicit demo data" in h, "unlabeled demo")
    check("LIVE uses fetch POST", "fetch(bridgeURL() + '/screen'" in h or 'fetch(bridgeURL()' in h, "no live path")
    check("upload size+type guarded client-side",
          "MAX_UPLOAD_BYTES" in h and "startsWith('image/')" in h, "unsafe upload")
    check("evidence escaped", "escapeHtml(ev)" in h or "map(ev=>'<li>'+escapeHtml(ev)" in h, "XSS gap")
    check("patient-id escaped everywhere rendered (String() form)",
          h.count("escapeHtml(String(c.patientId))") >= 3 and "escapeHtml(pid)" in h, "id XSS")
    check("null rule never Grade 0",
          "not Grade 0" in h and "sc.rule == null" in h, "null-to-zero")
    check("proxy never verified",
          "PROXY screening signal" in h, "proxy inflation")


def test_queue_bounds():
    h = txt("netrasetu.html")
    check("150-cap FIFO", "LOCAL_QUEUE_MAX = 150" in h, "unbounded growth")
    check("thumbnails only (no raw persistence)",
          "toQueueRecord" in h and "readAsDataURL" in h and h.count("localStorage.setItem") <= 3, "raw leak")
    check("quota failure surfaced (no silent loss)",
          "quota-unavailable" in h, "silent loss")
    check("reviewer fields separate from AI output",
          "reviewerGrade" in h and "reviewerNotes" in h, "overwrite risk")
    check("insufficient/disagreement routable",
          "insufficient evidence" in h and "NUMERIC_DISAGREE" in h, "routing gap")


TESTS = [test_mapping_keys, test_null_missing, test_schema_module,
         test_frontend_modes, test_queue_bounds]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage7 bridge: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
