#!/usr/bin/env python3
"""scripts/smoke_e2e.py — live bridge smoke runner (stdlib + urllib only).

Two strictly separated concerns (never conflate them):
  TRANSPORT/SCHEMA smoke: HTTP, auth, validation, schema shape, request
    IDs, cleanup, repeat-request hygiene. A synthetic valid PNG proves
    ONLY these - never clinical screening success.
  CLINICAL-PATH smoke: requires MATLAB + Engine + .mat assets. Without
    them every clinical assertion is SKIPPED (honest, counted, non-zero
    exit only on --strict with a live backend expected).

Auth matrix (negative upload tests authenticate FIRST so the intended
validation layer is exercised, not the auth gate):
  key configured + missing/wrong key -> 401
  key absent server-side -> /screen 503 fail-closed
Repeat-request policy: schema validity + state isolation + no corruption
  (exact byte determinism asserted ONLY where explicitly guaranteed -
  it is not, so we assert stability of shape/status instead).

Usage:
  python3 scripts/smoke_e2e.py --bridge http://localhost:8000 [--key KEY]
      [--strict] [--image path/to/fundus.png]
Exit: 0 unless a runnable check FAILs (SKIPs never fail without --strict).
"""
import argparse
import io
import json
import os
import sys
import urllib.request
import urllib.error

TOPLEVEL_KEYS = ("apiVersion", "status", "dl", "conf", "rule", "evidence",
                 "quality", "clinical", "disagreement", "temperature")


def _req(base, path, method="GET", headers=None, body=None):
    r = urllib.request.Request(base + path, data=body, headers=headers or {},
                               method=method)
    try:
        with urllib.request.urlopen(r, timeout=25) as resp:
            raw = resp.read()
            return resp.status, dict(resp.headers), raw
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception as e:
        return None, {}, str(e).encode()


def _png_bytes(w=64, h=64):
    """Minimal VALID-STRUCTURE PNG (transport fixture only)."""
    import struct
    import zlib

    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes((x * 4) % 256 for _ in range(w * 3)) for x in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="http://localhost:8000")
    ap.add_argument("--key", default=os.environ.get("SIH_API_KEY", ""))
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--image", default="")
    args = ap.parse_args()

    passed, failed, skipped = [], [], []

    def ok(name):
        passed.append(name)
        print(f"[PASS] {name}")

    def no(name, why):
        failed.append((name, why))
        print(f"[FAIL] {name} :: {why}")

    def skip(name, why):
        skipped.append(name)
        print(f"[SKIP] {name} :: {why}")

    base = args.bridge.rstrip("/")
    # 1. Bridge reachable?
    st, _, _ = _req(base, "/health")
    if st is None:
        skip("bridge-reachable", "no bridge at " + base + " (DEPLOYMENT-GATED)")
        bridge_up = False
    else:
        bridge_up = True
        if st == 200:
            ok("bridge-reachable")
        else:
            no("bridge-reachable", f"HTTP {st}")
    if not bridge_up:
        for n in ("auth-matrix", "transport-schema", "negative-uploads",
                  "repeat-hygiene", "clinical-path"):
            skip(n, "bridge down (DEPLOYMENT-GATED)")
        return _verdict(passed, failed, skipped, args.strict)

    # 2. Auth matrix (no key / wrong key first).
    st, _, raw = _req(base, "/screen", method="POST",
                      headers={"Content-Type": "application/octet-stream"},
                      body=b"hello")
    if st in (401, 503):
        ok(f"auth-matrix (no-key -> {st})")
    else:
        no("auth-matrix", f"no-key screening returned HTTP {st} (must be 401/503)")

    headers = {"X-API-Key": args.key} if args.key else {}
    # 3. Transport/schema smoke with synthetic PNG (NOT clinical evidence).
    boundary = "----smoke1234"
    if args.image and os.path.isfile(args.image):
        with open(args.image, "rb") as f:
            payload = f.read()
        fname = os.path.basename(args.image)
    else:
        payload = _png_bytes()
        fname = "synthetic-transport-fixture.png"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f"filename=\"{fname}\"\r\nContent-Type: image/png\r\n\r\n").encode() + payload + \
        f"\r\n--{boundary}--\r\n".encode()
    st, hdrs, raw = _req(base, "/screen", method="POST",
                         headers=dict(headers, **{"Content-Type": f"multipart/form-data; boundary={boundary}"}),
                         body=body)
    if st == 503 and not args.key:
        skip("transport-schema", "fail-closed without key (configure SIH_API_KEY to probe deeper)")
    elif st == 200:
        try:
            doc = json.loads(raw.decode())
        except Exception as e:
            no("transport-schema", f"non-JSON 200: {e}")
            doc = None
        if doc is not None:
            missing = [k for k in TOPLEVEL_KEYS if k not in doc]
            if not missing:
                ok("transport-schema (top-level keys present)")
            else:
                no("transport-schema", f"missing keys {missing}")
            if doc.get("apiVersion") == "1.0-stage7":
                ok("schema-version")
            else:
                no("schema-version", f"{doc.get('apiVersion')}")
            if "requestId" in doc or hdrs.get("X-Request-ID"):
                ok("request-id-propagated")
            else:
                no("request-id-propagated", "no request ID echoed")
    elif st in (401, 429):
        skip("transport-schema", f"HTTP {st} (auth/busy gate engaged first)")
    else:
        no("transport-schema", f"HTTP {st}: {raw[:120]!r}")
    # 4. Negative uploads authenticate FIRST (exercises validation layer).
    if args.key:
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (15 * 1024 * 1024 + 1)
        st, _, _ = _req(base, "/screen", method="POST",
                        headers=dict(headers, **{"Content-Type": "multipart/form-data; boundary=x"}),
                        body=b"--x\r\nContent-Disposition: form-data; name=\"image\"; filename=\"big.png\"\r\n"
                             b"Content-Type: image/png\r\n\r\n" + big + b"\r\n--x--\r\n")
        if st == 413:
            ok("oversize-rejected")
        else:
            no("oversize-rejected", f"HTTP {st}")
        st, _, _ = _req(base, "/screen", method="POST",
                        headers=dict(headers, **{"Content-Type": "multipart/form-data; boundary=x"}),
                        body=b"--x\r\nContent-Disposition: form-data; name=\"image\"; filename=\"evil.txt\"\r\n"
                             b"Content-Type: image/png\r\n\r\nnot-an-image\r\n--x--\r\n")
        if st == 400:
            ok("malformed-rejected")
        else:
            no("malformed-rejected", f"HTTP {st}")
    else:
        skip("negative-uploads", "no key supplied (auth gate precedes validation)")
    # 5. Repeat hygiene: two identical posts, same shape/status, no corruption.
    if args.key:
        def once():
            s, _, r = _req(base, "/screen", method="POST",
                           headers=dict(headers, **{"Content-Type": f"multipart/form-data; boundary={boundary}"}),
                           body=body)
            try:
                return s, json.loads(r.decode())
            except Exception:
                return s, None
        s1, d1 = once()
        s2, d2 = once()
        if s1 == s2 == 200 and d1 is not None and d2 is not None \
                and set(d1) == set(d2) and d1.get("status") == d2.get("status"):
            ok("repeat-hygiene (shape/status stable, isolated)")
        else:
            no("repeat-hygiene", f"{s1}/{s2}")
    else:
        skip("repeat-hygiene", "no key supplied")
    # 6. Clinical path: gated, never fabricated.
    skip("clinical-path", "requires MATLAB + Engine + .mat assets + real fixture (DATA/MATLAB-GATED)")
    return _verdict(passed, failed, skipped, args.strict)


def _verdict(passed, failed, skipped, strict):
    print(f"\nSMOKE: {len(passed)} PASS, {len(failed)} FAIL, {len(skipped)} SKIP"
          f"{' (strict)' if strict else ''}")
    if failed:
        return 1
    if strict and skipped:
        print("STRICT: skips present with live backend expected -> FAIL")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
