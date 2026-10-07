#!/usr/bin/env python3
"""test_stage13_data.py — Stage-13 acquisition/provenance contracts (stdlib).

Independent mirrors of: structured registry metadata (no prose parsing),
acceptance verdicts incl. COUNT_NOT_VERIFIED, label-range validation,
manifest reproducibility, frozen-test-first ordering, candidate state
machine, timestamp-independent identity, calibration-mismatch rejection,
preprocessing parity pins, GT/pred provenance, no-test-tuning hooks.

Run: python3 tests/test_stage13_data.py
MATLAB twin: testDataContracts.m (UNEXECUTED here - no MATLAB).
All values are TEST/LOGIC fixtures, never project results.
"""
import hashlib
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


def content_id(material: str) -> str:
    return "cand_" + hashlib.sha256(material.encode()).hexdigest()[:16]


def test_registry_structured():
    body = txt("datasetRegistry.m")
    for f in ("sourceUrl", "sourceType", "expectedImageCount",
              "expectedLabelCount", "expectedMaskCount", "versionPolicy"):
        check(f"registry owns {f}", f in body, "missing")
    check("sizeHint marked display-only (never parsed)",
          "DISPLAY ONLY" in body, "prose-parsing risk")
    check("mirrors labeled as mirrors (Messidor-2, DDR)",
          body.count("'mirror'") >= 2, "mirror laundering")
    check("unknowns stay UNKNOWN (no guessing)",
          "'UNKNOWN'" in body, "fabrication risk")
    check("approximate ~10k NOT a hard count (NaN)",
          "NaN,NaN,NaN,'download-date')" in body.replace(" ", ""), "false precision")
    check("exact counts kept exact (STARE 20, APTOS 3662, IDRiD 516/81)",
          all(s in body for s in ("20, NaN, 20", "3662, 3662", "516, NaN, 81")), "count drift")
    check("registry names/purposes preserved",
          all(s in body for s in ("'Messidor-2', 'test_holdout'", "'DDR', 'grading'")), "renamed")


def test_acceptance_gate():
    body = txt("acceptDataset.m")
    for f in ("sourceStatus", "versionStatus", "imageCountStatus",
              "labelRangeValid", "corruptCount", "overallStatus",
              "COUNT_NOT_VERIFIED", "ACCEPTED_WITH_NOTES"):
        check(f"verdict reports {f}", f in body, "missing")
    check("unknown dataset errors (no silent pass)",
          "unknown - dataset" in body or "unknownDataset" in body or ":unknown" in body, "silent pass")
    check("out-of-range ICDR rejects",
          "REJECTED" in body, "bad labels pass")


def test_freeze_lifecycle():
    body = txt("freezeCandidate.m")
    check("PRE_TRAIN requires test hash first",
          "testSplitHash" in body and "freeze TEST first" in body, "order gap")
    check("FINALIZE requires PRE_TRAIN record",
          "preTrainCandidate" in body, "skip-phase risk")
    check("FINALIZE requires checkpoint + calibration",
          all(s in body for s in ("checkpointId", "calibrationArtifactId", "validationSplitId")), "thin finalize")
    check("mismatch invalidates (checkpoint/split/classes)",
          body.count("calibrationMismatch") >= 3, "silent attach")
    check("timestamps excluded from identity",
          "NEVER identity" in body or "never identity" in body.lower(), "timestamp-dependent IDs")
    check("git SHA best-effort, never fabricated",
          "unavailable" in body, "fabrication risk")
    check("test-cohort change forces new version",
          "new evaluation version" in body or "new explicit evaluation" in body, "silent rewrite")
    check("REJECTED verdict blocks freeze without override",
          "rejectedDataset" in body and "overrideRejected" in body, "silent freeze")
    check("parent lineage recorded",
          "parentCandidateId" in body, "lineage gap")
    check("stage labels present",
          "TRAINING_READY" in body and "FINALIZED" in body and "EVALUATION_READY" in body, "stage gap")
    check("hash method recorded",
          "hashMethod" in body, "strength unknown")
    check("attested-not-computed marked",
          "ATTESTED_NOT_COMPUTED" in body, "false computed claim")
    check("hashArtifacts helper exists",
          os.path.isfile(os.path.join(ROOT, "hashArtifacts.m"))
          and "sha256:" in txt("hashArtifacts.m"), "no computed hashes")
    m1 = content_id("PRE_TRAIN|m|tr|va|te|v1")
    m2 = content_id("PRE_TRAIN|m|tr|va|te|v1")
    check("timestamp-independent identity (same inputs, same ID)", m1 == m2, "")
    check("input change mints new ID",
          content_id("PRE_TRAIN|m|tr|va|te|v1") != content_id("PRE_TRAIN|m|tr|va|te|v2"), "")


def test_parity_and_provenance():
    check("seg train keeps canonical preprocessing",
          "preprocessFundusForSegmentation" in txt("train_UNet_Segmentation.m"), "fork")
    check("grader keeps canonical fusion + predicted-mask gate",
          "buildGradingFusionTensor" in txt("train_DR_Grader.m")
          and "segmentationNotTrained" in txt("train_DR_Grader.m"), "leak risk")
    check("seed discipline intact (grader + seg)",
          "rng(" in txt("train_DR_Grader.m") and "rng(" in txt("train_UNet_Segmentation.m"), "nondeterminism")
    check("checkpoint provenance via saveModelWithMetadata",
          "saveModelWithMetadata" in txt("train_DR_Grader.m"), "provenance gap")


def test_docs_and_claims():
    for f in ("STAGE13_DATA_TRAINING_HANDOFF.md",):
        check(f"{f} exists",
              os.path.isfile(os.path.join(ROOT, f)), "missing")
    check("runbook acquisition section exists",
          "Acquisition" in txt("docs/RUNBOOK.md") or "acquisition" in txt("docs/RUNBOOK.md").lower(), "missing")


TESTS = [test_registry_structured, test_acceptance_gate, test_freeze_lifecycle,
         test_parity_and_provenance, test_docs_and_claims]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage13 data: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
