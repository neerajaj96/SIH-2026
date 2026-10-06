"""
bridge_server.py — REST bridge between a web/mobile frontend and the
real MATLAB pipeline, via MATLAB Engine for Python.

STATUS: written against the documented MATLAB Engine for Python API,
but UNTESTED — there is no MATLAB installation in the environment this
was written in, so the actual engine call, the struct-to-dict field
access, and the cell-array-to-list conversion for `evidence` have not
been run for real. Test with `python bridge_server.py` against a real
image before wiring the published web console to this.

WHAT THIS DOES NOT DO YET: the published NetraSetu console
(netrasetu.html) is currently fully self-contained on client-side mock
data — it does not call this server. Wiring it up is a separate step:
replace the `SCENARIOS`-based mock logic in its `runBtn` click handler
with a `fetch('http://<this-server>:8000/screen', {method:'POST',
body: formData})` call, and map the JSON response (same field names:
dl, conf, rule, evidence, nv, gradCamOnDisc, status) into the same
`renderResult()` function that's already there.

SETUP:
    pip install fastapi uvicorn python-multipart matlabengine
    (or: cd "matlabroot/extern/engines/python" && python setup.py install,
    if `pip install matlabengine` doesn't match your MATLAB version)
    python bridge_server.py
    # serves on http://localhost:8000, POST an image to /screen

Requires: MATLAB installed on this machine, with this project's .m
files (screenOneImage.m and everything it calls) on the MATLAB path,
and the trained .mat files in the working directory MATLAB starts in
(or adjust PROJECT_DIR below and cd to it before starting the engine).
"""

import atexit
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

app = FastAPI(title="NetraSetu screening bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to the console's actual origin before any real deployment
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_engine = None


def get_engine():
    """Starts (once) and reuses one MATLAB engine process. Starting an
    engine takes several seconds — this deliberately happens once at
    first request, not per-request, or every screening call would pay
    MATLAB startup latency."""
    global _engine
    if _engine is None:
        print("Starting MATLAB engine (first call only, takes a few seconds)...")
        _engine = matlab.engine.start_matlab()
        _engine.cd(PROJECT_DIR)
        _engine.addpath(PROJECT_DIR)
        print(f"MATLAB engine ready, cwd = {PROJECT_DIR}")
    return _engine


@atexit.register
def _shutdown_engine():
    global _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass


def matlab_result_to_dict(m_result) -> dict:
    """Converts screenOneImage.m's returned struct into a plain Python
    dict with JSON-safe types. UNTESTED: MATLAB Engine for Python
    generally returns a struct as something dict-like (`m_result['dl']`
    style access), and a MATLAB cell array of char as a tuple of str —
    that is the documented, expected behavior this function assumes,
    but confirm it against a real call before trusting this in
    production; adjust the field-by-field access below if your engine
    version behaves differently.
    """
    def safe_num(v):
        try:
            f = float(v)
            return None if f != f else f  # NaN -> None, JSON has no NaN
        except (TypeError, ValueError):
            return None

    evidence = m_result["evidence"]
    if isinstance(evidence, str):
        evidence_list = [evidence]
    else:
        try:
            evidence_list = list(evidence)
        except TypeError:
            evidence_list = []

    return {
        "status": str(m_result["status"]),
        "errorMessage": str(m_result["errorMessage"]),
        "focus": safe_num(m_result["focus"]),
        "entropy": safe_num(m_result["entropy"]),
        "roiPassed": bool(m_result["roiPassed"]),
        "dl": safe_num(m_result["dl"]),
        "conf": safe_num(m_result["conf"]),
        "rule": safe_num(m_result["rule"]),
        "evidence": evidence_list,
        "nv": bool(m_result["nv"]),
        "gradCamOnDisc": bool(m_result["gradCamOnDisc"]),
    }


@app.get("/health")
def health():
    """Cheap check that does NOT start the engine, so a load balancer
    or uptime monitor hitting this frequently doesn't force a MATLAB
    boot. Use /health/deep to actually verify the engine starts."""
    return {"status": "ok", "engine_started": _engine is not None}


@app.get("/health/deep")
def health_deep():
    try:
        eng = get_engine()
        eng.eval("1+1;", nargout=0)
        return {"status": "ok", "engine_started": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MATLAB engine not responding: {e}")


@app.post("/screen")
async def screen(image: UploadFile = File(...)):
    """Runs screenOneImage.m on the uploaded photo and returns its
    result as JSON, in the exact field shape netrasetu.html's mock
    SCENARIOS objects already use."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        eng = get_engine()
        m_result = eng.screenOneImage(tmp_path, nargout=1)
        return JSONResponse(matlab_result_to_dict(m_result))
    except matlab.engine.MatlabExecutionError as e:
        # A MATLAB-side error() call surfaces here, not inside
        # screenOneImage's own try/catch (that one only catches errors
        # from ITS OWN body, e.g. a bad segmentation call - a bad path
        # or a missing .m file on the MATLAB path throws before that).
        raise HTTPException(status_code=500, detail=f"MATLAB error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
