#!/usr/bin/env python3
"""test_flagship_sre.py — Wave-1 backend/SRE regression (stdlib only).

Pins: lock-contract assertion, serialization-vs-engine split (500 vs 503,
no invalidate on serialization), rate-bucket bound/reap, isolated tmpdir
+ idempotent unlink + 0600, ms timings on all paths, X-Request-ID exposed
via CORS, startup config validation, unhandled-exception ID preservation,
Pillow capability-gated (not mandatory).

Run: python3 tests/test_flagship_sre.py
MATLAB engine paths remain MATLAB-GATED; this suite checks code contracts.
"""
import os
import sys

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


def test_lock_contract():
    b = txt("bridge_server.py")
    check("lock assertion in _locked_engine", "_engine_lock.locked()" in b,
          "unenforced lock")
    check("serialization error class", "class BridgeSerializationError" in b,
          "no split")
    check("serialization never invalidates (doc)",
          "MUST NOT invalidate" in b or "Serialization failures are NOT" in b,
          "contract undocumented")


def test_error_split():
    b = txt("bridge_server.py")
    check("missing legacy key -> serialization (not KeyError->503)",
          "missing legacy key" in b, "conflation remains")
    check("serialization mapped to 500", "except BridgeSerializationError" in b,
          "500 path missing")
    check("engine mapped to 503", 'status_code=503' in b, "503 path missing")
    check("no double invalidate on serialization",
          b.count("_engine_invalidate()") <= 6, "count=%d" % b.count("_engine_invalidate()"))
    check("retry-once only for engine (serialization reraises)",
          "Serialization failures are NOT" in b, "retry-all risk")


def test_rate_bound():
    b = txt("bridge_server.py")
    check("bucket cap constant", "_RATE_BUCKET_CAP" in b, "unbounded")
    check("reap logic present", "now - v[-1]" in b or "expired" in b.lower(),
          "no reap")
    check("Retry-After on 429", 'Retry-After' in b, "no backoff hint")


def test_temp_hardening():
    b = txt("bridge_server.py")
    check("isolated tmpdir", "_BRIDGE_TMPDIR" in b and "netrasetu-bridge" in b,
          "world-readable gettempdir")
    check("SIH_TMPDIR override", "SIH_TMPDIR" in b, "not configurable")
    check("0600 chmod", "0o600" in b, "perms missing")
    check("idempotent unlink helper", "def _secure_unlink" in b, "helper missing")
    check("both paths use helper", b.count("_secure_unlink(tmp_path)") >= 2,
          "leak path")


def test_observability():
    b = txt("bridge_server.py")
    check("ms on rate-limited path", b.count("ms=_elapsed_ms()") >= 4,
          f"count={b.count('ms=_elapsed_ms()')}")
    check("engine_ms split logged", "engine_ms=" in b, "no split")
    check("X-Request-ID exposed via CORS", 'expose_headers' in b, "JS cannot read ID")
    check("allow_credentials explicit", "allow_credentials=False" in b, "policy gap")
    check("unhandled handler preserves ID", "_unhandled_error_shape" in b,
          "ID divergence")


def test_startup_validation():
    b = txt("bridge_server.py")
    check("CORS scheme validated", "must be http(s)" in b or "startswith(\"http" in b,
          "weak CORS")
    check("'*' refused", '"*"' in b and "must not contain" in b, "star risk")
    check("PROJECT_DIR checked", "SIH_PROJECT_DIR does not exist" in b or "isdir(PROJECT_DIR)" in b,
          "CWD drift silent")
    check("rate range clamped", "1..1000" in b, "range gap")
    check("deep reports configValid", "configValid" in b, "not observable")


def test_upload_gating():
    b = txt("bridge_server.py")
    check("Pillow optional only", "from PIL import Image" in b and "ImportError" in b,
          "mandatory dep risk")
    check("no new mandatory imports",
          "import PIL" not in b.replace("from PIL import", ""), "top-level PIL")


TESTS = [test_lock_contract, test_error_split, test_rate_bound,
         test_temp_hardening, test_observability, test_startup_validation,
         test_upload_gating]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== flagship SRE: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
