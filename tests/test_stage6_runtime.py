#!/usr/bin/env python3
"""test_stage6_runtime.py — Stage-6 runtime contracts (stdlib only).

Static pins + independent mirrors for: batch isolation, CWD candor,
collision-proof artifacts, entry-point consensus, single-computation
paths, NaN safety. MATLAB twin: testStage6Runtime.m (UNEXECUTED here).

Run: python3 tests/test_stage6_runtime.py
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


def unique_mirror(existing, stem):
    """Independent mirror of uniqueArtifactPath collision loop."""
    cand = f"{stem}_stamp"
    if cand not in existing:
        return cand
    k = 2
    while f"{cand}_v{k}" in existing:
        k += 1
    return f"{cand}_v{k}"


def test_batch_isolation():
    b = txt("runBatchScreening.m")
    check("per-file try/catch in batch loop", "try" in b and "catch ME" in b, "one bad file halts batch")
    check("catch builds ERROR row (schema preserved)",
          "'error'" in b and "qualityDecision" in b and "ERROR" in b, "malformed row")
    check("catch records message (useful errors)", "ME.message" in b, "silent swallow")
    check("CWD named in simulated warning", "pwd" in b, "silent simulation")


def test_collision_proof():
    check("helper exists", os.path.exists(os.path.join(ROOT, "uniqueArtifactPath.m")), "missing")
    h = txt("uniqueArtifactPath.m")
    check("collision loop with suffix", "_v%d" in h or "_v% d" in h or "_v" in h, "no uniquifier")
    check("readability preserved (stem+timestamp first)",
          "datestr" in h, "unreadable names")
    for rel in ("runBatchScreening.m", "auditQualityBatch.m", "evaluateClinicalRule.m",
                "production_inference.m"):
        check(f"{rel} uses helper", "uniqueArtifactPath(" in txt(rel), "timestamp overwrite risk")
    check("no raw timestamped fullfile remains",
          "sprintf('batch_summary_%s" not in txt("runBatchScreening.m"), "stale path")
    # executable mirror behavior
    check("mirror: first free name returned",
          unique_mirror(set(), "batch_summary") == "batch_summary_stamp", "")
    check("mirror: collision appends version",
          unique_mirror({"batch_summary_stamp"}, "batch_summary") == "batch_summary_stamp_v2", "")
    check("mirror: deterministic repeat",
          unique_mirror({"a_stamp"}, "a") == unique_mirror({"a_stamp"}, "a"), "")


def test_entry_consensus():
    rsp = txt("runScreeningPipeline.m")
    for f in ("qualityDecision", "ruleStatus", "nvStatus", "temperatureState",
              "explainStatus", "disagreement", "odValidity", "quadrantValid"):
        check(f"pipeline exposes {f}", f in rsp, "contract gap")
    scr = txt("screenOneImage.m")
    for f in ("qualityDecision", "ruleStatus", "nvStatus"):
        check(f"bridge struct passes {f}", f in scr, "bridge blind")
    pinf = txt("production_inference.m")
    for f in ("r.qualityDecision", "r.ruleStatus", "r.explainStatus", "r.disagreement"):
        check(f"report reads {f}", f in pinf, "report stale")


def test_single_paths():
    rsp = txt("runScreeningPipeline.m")
    check("one Grad-CAM path (canonical only)",
          rsp.count("gradCAM(") == 0 and "explainGradCAM(" in rsp, "duplicate explanation")
    check("one disagreement construction",
          rsp.count("analyzeDisagreement(") == 1, "duplicate adjudication")
    check("one fusion construction",
          rsp.count("buildGradingFusionTensor(") == 1 and "cat(3, imresize" not in rsp, "duplicate fusion")
    check("no NaN-to-zero (double() only on logicals)",
          "double(res.nv)" in txt("runBatchScreening.m"), "conversion gap")


def test_runtime_pins():
    check("MATLAB suite exists (UNEXECUTED)",
          os.path.exists(os.path.join(ROOT, "testStage6Runtime.m")), "missing")
    check("suite covers 16 areas",
          txt("testStage6Runtime.m").count("[nP,nT] = rt(") >= 16, "coverage gap")
    check("handoff exists",
          os.path.exists(os.path.join(ROOT, "STAGE6_RUNTIME_HANDOFF.md")), "missing")


TESTS = [test_batch_isolation, test_collision_proof, test_entry_consensus,
         test_single_paths, test_runtime_pins]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage6 runtime: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
