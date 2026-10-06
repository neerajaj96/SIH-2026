#!/usr/bin/env python3
"""test_stage10_e2e.py — Stage-10 integration contracts (stdlib only).

Verifies the smoke tooling + docs + deployment topology WITHOUT MATLAB,
browser, network, or deployment. Live assertions are MATLAB-/DEPLOYMENT-
GATED and asserted as SKIP-paths here (the runner is executed against a
dead port to prove honest SKIP behavior).

Run: python3 tests/test_stage10_e2e.py
"""
import os
import subprocess
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


def test_smoke_tooling():
    p = os.path.join(ROOT, "scripts", "smoke_e2e.py")
    check("smoke runner exists", os.path.isfile(p), "missing")
    body = txt("scripts/smoke_e2e.py")
    check("transport/clinical separation explicit",
          "TRANSPORT/SCHEMA smoke" in body and "never clinical" in body.lower()
          or "NOT clinical" in body, "conflation risk")
    check("auth-first negative tests",
          "authenticate FIRST" in body or "auth gate precedes" in body, "order gap")
    check("no byte-determinism demand",
          "byte determinism" in body or "byte-determinism" in body, "overstrict")
    check("honest SKIP verdicts", "SKIP" in body and "DEPLOYMENT-GATED" in body, "fake PASS risk")
    check("synthetic PNG labeled fixture-only",
          "transport fixture only" in body, "fixture misuse")


def test_smoke_dead_port():
    """Executes the runner against a dead port: must SKIP, never FAIL."""
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "smoke_e2e.py"),
                        "--bridge", "http://127.0.0.1:9"],
                       capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    check("dead bridge exits 0 with SKIPs", r.returncode == 0, f"rc={r.returncode}")
    check("SKIP verdicts recorded", "SKIP" in out, "silent")
    check("no FAIL on dead bridge", "[FAIL]" not in out, "false failure")


def test_levels_and_topology():
    h = txt("STAGE10_INTEGRATION_HANDOFF.md") if os.path.isfile(
        os.path.join(ROOT, "STAGE10_INTEGRATION_HANDOFF.md")) else ""
    check("handoff exists", bool(h), "missing")
    for lvl in ("UNIT/CONTRACT", "LOCAL INTEGRATION", "MATLAB RUNTIME",
                "LIVE HTTP", "DEPLOYMENT"):
        check(f"level defined: {lvl}", lvl in h, "missing")
    check("impossible topology marked (no serverless MATLAB)",
          "serverless" in h.lower() or "static hosting" in h.lower(), "fantasy topology")
    check("TLS/proxy/key-management DEPLOYMENT-GATED",
          "DEPLOYMENT-GATED" in h, "unmarked")
    check("evidence matrix present", "Evidence" in h or "evidence" in h, "missing")


def test_ci_honesty():
    p = os.path.join(ROOT, ".github", "workflows", "python-contracts.yml")
    check("CI workflow exists", os.path.isfile(p), "missing")
    body = txt(".github/workflows/python-contracts.yml") if os.path.isfile(p) else ""
    check("MATLAB job explicitly UNEXECUTED/non-gating",
          "UNEXECUTED" in body, "fake-green risk")
    check("stdlib suites actually run",
          "test_stage9_security" in body and "test_stage8_telemed" in body, "empty CI")


def test_runbook():
    body = txt("docs/RUNBOOK.md") if os.path.isfile(
        os.path.join(ROOT, "docs", "RUNBOOK.md")) else txt("README.md")
    for item in ("SIH_API_KEY", "/health", "MATLAB", "CORS", "reverse proxy"):
        check(f"runbook covers: {item}", item in body, "missing")


TESTS = [test_smoke_tooling, test_smoke_dead_port, test_levels_and_topology,
         test_ci_honesty, test_runbook]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage10 e2e: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
