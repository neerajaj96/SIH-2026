#!/usr/bin/env python3
"""test_stage11_runtime.py — Stage-11 harness contracts (stdlib only).

Verifies the runtime-verification infrastructure WITHOUT executing it:
tri-state accounting (skips never PASS), simulated/trained separation,
normalization-layer comparison policy, evidence-dir hygiene, benchmark
fields, failure matrix, docs. MATLAB twin: testStage11Runtime.m
(UNEXECUTED here - no MATLAB, assets, Engine, or deployment).

Run: python3 tests/test_stage11_runtime.py
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


def test_tristate_accounting():
    body = txt("testStage11Runtime.m")
    check("separate passed/failed/unexecuted lists",
          "rep.passed" in body and "rep.unexecuted" in body, "binary accounting")
    check("prereq gate records UNEXECUTED (never PASS)",
          "UNEXECUTED" in body and "never PASS" in body.lower() or "NEVER increments PASS" in body, "skip-as-pass risk")
    check("summary prints all three counts",
          "PASS, %d FAIL, %d UNEXECUTED" in body, "hidden skips")


def test_sim_trained_separation():
    body = txt("testStage11Runtime.m")
    check("simulated checks labeled SIMULATED-adjacent (no weights needed)",
          "cSimQuality" in body and "cSimMalformed" in body, "missing")
    check("trained checks gated on assets (needNets)",
          body.count("needNets(hasNets)") >= 5, "ungated trained claims")
    check("no-asset prerequisite message names the .mat files",
          "unet_*.mat" in body, "vague gate")


def test_bridge_comparison_policy():
    body = txt("bridge_server.py")
    check("NaN->null conversion centralized",
          "to_null_number" in body or "safe_num" in body, "conversion drift")
    check("nested disagreement converted (not dropped)",
          "disagreement" in body, "schema loss")
    check("smoke runner separates transport from clinical",
          "transport fixture only" in txt("scripts/smoke_e2e.py"), "conflation")


def test_evidence_hygiene():
    for rel in ("scripts/matlab_env_report.m", "testStage11Runtime.m",
                "scripts/benchmarkStage11.m"):
        check(f"{rel} defaults evidence to tempdir (never repo)",
              "tempdir" in txt(rel), "repo pollution")
    check("no generated evidence tracked",
          not any(p.endswith((".mat", ".csv", ".png")) and "report" in p.lower()
                  for p in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, p))
                  and p.startswith(("matlab_env_", "clinical_rule_eval_"))), "pollution")


def test_benchmark_fields():
    body = txt("scripts/benchmarkStage11.m")
    for f in ("coldSec", "warmSecMean", "warmSecStd", "memoryBefore",
              "memoryAfter", "artifactBytes", "hardware", "UNEXECUTED"):
        check(f"benchmark records {f}", f in body, "missing")


def test_failure_matrix():
    body = txt("testStage11Runtime.m")
    check("failure/recovery check exists", "cFailure" in body, "missing")
    check("error carries message", "errorMessage" in body, "silent error")
    docs = txt("STAGE11_RUNTIME_VALIDATION_HANDOFF.md") if os.path.isfile(
        os.path.join(ROOT, "STAGE11_RUNTIME_VALIDATION_HANDOFF.md")) else ""
    check("failure matrix documented", "Failure" in docs or "failure" in docs, "missing")


TESTS = [test_tristate_accounting, test_sim_trained_separation,
         test_bridge_comparison_policy, test_evidence_hygiene,
         test_benchmark_fields, test_failure_matrix]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage11 runtime: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
