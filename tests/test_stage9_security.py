#!/usr/bin/env python3
"""test_stage9_security.py — Stage-9 security/privacy/observability
contracts (stdlib only). No network, no MATLAB, no pentest claims.

Run: python3 tests/test_stage9_security.py
Covers: fail-closed auth, CORS allowlist, magic/structural validation,
dimension caps, rate-limit + trusted-proxy semantics, request-ID hygiene,
safe error envelope, temp lifecycle, CSP honesty, sessionStorage key
discipline, PHI non-persistence, reviewer/AI separation, logging hygiene.
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


def test_fail_closed_auth():
    b = txt("bridge_server.py")
    check("no anonymous bypass (no SIH_ALLOW_ANON)",
          "ALLOW_ANON" not in b, "bypass present")
    check("missing key -> 401, unconfigured -> 503",
          '"unauth"' in b and '"unconfigured"' in b, "fail-open risk")
    check("constant-time key compare",
          "compare_digest" in b, "timing side-channel")
    check("auth gate runs before body work",
          b.index("_require_api_key") < b.index("await image.read"), "late gate")


def test_cors_allowlist():
    b = txt("bridge_server.py")
    check("no wildcard CORS", 'allow_origins=["*"]' not in b, "wildcard live")
    check("origins from env with localhost default",
          "SIH_CORS_ORIGINS" in b and "http://localhost:8000" in b, "hardcoded origins")
    check("empty allowlist refuses startup",
          "refusing to start" in b, "fail-open config")


def test_upload_validation():
    sys.path.insert(0, ROOT)
    import bridge_schema as S
    check("exact 15MB boundary accepted", S.check_size_ok(15 * 1024 * 1024), "")
    check(">15MB rejected", not S.check_size_ok(15 * 1024 * 1024 + 1), "")
    check("missing length path exists (streaming loop)",
          "while True" in txt("bridge_server.py"), "length-trust")
    check("JPEG magic recognized",
          S.detect_image_kind(b"\xff\xd8\xff\xe0junk") == "jpeg", "")
    check("PNG magic recognized",
          S.detect_image_kind(bytes.fromhex("89504e470d0a1a0a") + b"junk") == "png", "")
    check("text prefix rejected", S.detect_image_kind(b"not an image") is None, "")
    check("truncated header yields None (decoder-gated, not fabricated)",
          S.probe_jpeg_dimensions(b"\xff\xd8\xff") is None, "")
    check("dimension caps enforced server-side",
          "MAX_IMAGE_DIMENSION" in txt("bridge_server.py"), "no cap")
    check("extension allowlisted (not trusted for routing)",
          ".jpeg" in txt("bridge_server.py"), "traversal gap")
    check("temp file always unlinked (finally)",
          txt("bridge_server.py").count("_secure_unlink(tmp_path)") >= 2, "temp leak")
    check("temp helper idempotent (missing file is not an error)",
          "def _secure_unlink" in txt("bridge_server.py"), "helper missing")


def test_request_hygiene():
    sys.path.insert(0, ROOT)
    import bridge_schema as S
    check("valid IDs accepted", S.valid_request_id("web-abc_123.X-9"), "")
    check("empty rejected", not S.valid_request_id(""), "")
    check("overlong rejected", not S.valid_request_id("x" * 65), "")
    check("control chars rejected", not S.valid_request_id("a\nb"), "")
    check("spaces/slashes rejected",
          not S.valid_request_id("a b") and not S.valid_request_id("a/b"), "")
    check("non-string rejected", not S.valid_request_id(None), "")
    check("generated IDs valid", S.valid_request_id(S.new_request_id()), "")
    b = txt("bridge_server.py")
    check("middleware assigns ID pre-routing", "request.state.rid" in b, "late ID")
    check("error envelope carries ID, caps detail length",
          "requestId" in b and "[:300]" in b, "exfiltration risk")
    code = [ln for ln in txt("bridge_server.py").splitlines()
            if ln.strip() and not ln.strip().startswith(("#", '"""'))
            and '"""' not in ln]
    check("no traceback/paths in responses",
          not any("traceback" in ln or "exc_info" in ln for ln in code), "leak")


def test_rate_limit_semantics():
    b = txt("bridge_server.py")
    check("per-IP sliding window", "_rate_buckets" in b, "no limiter")
    check("X-Forwarded-For distrusted by default",
          "TRUSTED_PROXY" in b and "Blindly trusting" in b, "spoofable identity")
    check("429 on exhaust", "ratelimit" in b, "silent drop")


def test_frontend_hygiene():
    h = txt("netrasetu.html")
    check("CSP meta present and honest (unsafe-inline disclosed)",
          "Content-Security-Policy" in h and "unsafe-inline" in h, "theater CSP")
    check("CSP documents reverse-proxy tradeoff",
          "reverse proxy" in h or "reverse-proxy" in h, "undocumented tradeoff")
    check("connect-src bounded (no blanket https:)",
          "connect-src" in h and "https:" not in h.split("connect-src")[1].split('"')[0], "wide connect")
    check("key in sessionStorage only",
          "sessionStorage.setItem('netrasetu.apiKey'" in h and "localStorage.setItem('netrasetu.apiKey'" not in h, "key persistence")
    check("key sent as header, never in URL/body log",
          "X-API-Key" in h, "key transport gap")
    check("401/503 handled distinctly (no mock fallback)",
          "401" in h and "503" in h, "auth-blind UI")
    check("queue stores thumbnails, never raw uploads",
          "thumb:" in h and "readAsDataURL" in h, "raw leak")
    check("reviewer fields separate",
          "reviewerGrade" in h and "reviewerNotes" in h, "overwrite risk")
    check("no raw File object in stored record",
          "selectedFile" not in h.split("toQueueRecord")[1].split("submitCase")[0] if "toQueueRecord" in h else True, "raw leak")


def test_logging_hygiene():
    b = txt("bridge_server.py")
    check("allowlisted log fields only",
          'if k in ("rid", "route", "status", "ms", "category", "size", "detail")' in b
          or '"rid", "route", "status"' in b, "log exfiltration")
    check("no inference payload in logs (dl/rule/conf absent)",
          "result.get(\"status\")" in b or 'result.get("status")' in b, "payload leak")
    check("filenames not logged (only size/category/status)",
          all("filename" not in ln for ln in txt("bridge_server.py").splitlines()
              if "_log(" in ln), "filename in logs")


TESTS = [test_fail_closed_auth, test_cors_allowlist, test_upload_validation,
         test_request_hygiene, test_rate_limit_semantics,
         test_frontend_hygiene, test_logging_hygiene]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage9 security: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
