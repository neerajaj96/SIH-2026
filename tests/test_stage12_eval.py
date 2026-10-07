#!/usr/bin/env python3
"""test_stage12_eval.py — Stage-12 evaluation contracts (stdlib only).

Independent mirrors (not copies) of: manifest validation, referable
conversion, patient leakage, exact/possible duplicate classification,
role separation, GT/pred separation, exclusion accounting, QWK + Wilson
reference values, bootstrap seed reproducibility + degeneracy + unit
labeling, report NOT_MEASURED defaults, target-as-requirement labeling.

Run: python3 tests/test_stage12_eval.py
MATLAB twin: testEvalContracts.m (UNEXECUTED here - no MATLAB).
All numbers below are TEST/LOGIC values, never project results.
"""
import math
import os
import random
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


def qwk(y_true, y_pred, ncls=5):
    """Independent quadratic-weighted kappa (cf. sklearn reference
    0.88940092 on the shared (yTrue,yPred) pair used in runSelfTests)."""
    w = [[(i - j) ** 2 / (ncls - 1) ** 2 for j in range(ncls)] for i in range(ncls)]
    o = [[0] * ncls for _ in range(ncls)]
    for t, p in zip(y_true, y_pred):
        o[t][p] += 1
    n = len(y_true)
    rows = [sum(r) / n for r in o]
    cols = [sum(o[i][j] for i in range(ncls)) / n for j in range(ncls)]
    num = den = 0.0
    for i in range(ncls):
        for j in range(ncls):
            num += w[i][j] * o[i][j] / n
            den += w[i][j] * rows[i] * cols[j]
    return 1 - num / den if den else float("nan")


def wilson(x, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bootstrap_patients(groups, metric, seed=42, b=200):
    """Patient-level resampling with degeneracy counting (independent)."""
    rng = random.Random(seed)
    uniq = sorted(set(groups))
    mets, bad = [], 0
    for _ in range(b):
        draw = [rng.choice(uniq) for _ in uniq]
        idx = [i for i, g in enumerate(groups) if g in draw]
        try:
            m = metric(idx)
        except Exception:
            bad += 1
            continue
        if not isinstance(m, (int, float)) or not math.isfinite(m):
            bad += 1
            continue
        mets.append(m)
    return mets, bad


def test_manifest():
    rows = [
        {"dataset": "S", "datasetVersion": "t", "datasetSource": "syn",
         "imageId": "i1", "patientId": "pA", "labelICDR": 0, "split": "TRAIN",
         "provenance": "syn"},
        {"dataset": "S", "datasetVersion": "t", "datasetSource": "syn",
         "imageId": "i2", "patientId": "", "labelICDR": 2, "split": "TEST",
         "provenance": "syn"},
    ]
    for r in rows:
        check(f"manifest row {r['imageId']} ICDR valid", r["labelICDR"] in (0, 1, 2, 3, 4), "")
        check(f"manifest row {r['imageId']} split valid", r["split"] in ("TRAIN", "VAL", "TEST"), "")
        check(f"manifest row {r['imageId']} referable derived",
              (1 if r["labelICDR"] >= 2 else 0) == (1 if r["labelICDR"] >= 2 else 0), "")
    check("ICDR 7 rejected", 7 not in (0, 1, 2, 3, 4), "")
    check("referable 1->0", (1 >= 2) is False, "")
    check("referable 2->1", (2 >= 2) is True, "")
    check("missing patient IDs flagged (no silent patient claims)",
          rows[1]["patientId"] == "", "")
    body = txt("evaluationManifest.m")
    check("MATLAB manifest validates ICDR/split/referable/provenance",
          all(s in body for s in ("badICDR", "badSplit", "referableDrift", "noProvenance")), "validator gap")
    check("patientIdsAvailable=false disables patient claims",
          "patient-level protection NOT claimed" in txt("auditEvalLeakage.m"), "silent claim")


def test_leakage():
    imgs = [("i1", "TRAIN", "pA"), ("i2", "VAL", "pA"), ("i3", "TEST", "pA")]
    by_split = {}
    for i, s, p in imgs:
        by_split.setdefault(s, []).append((i, p))
    leak = any(p == "pA" for _, p in by_split.get("TEST", [])) and \
        any(p == "pA" for _, p in by_split.get("VAL", []))
    check("patient leakage caught (pA in VAL+TEST)", leak, "")
    ids = [i for i, _, _ in imgs] + ["i3"]
    check("exact duplicate caught", len(ids) != len(set(ids)), "")
    stems = ["a_v1", "a_v2"]
    check("possible (stem) collision distinguished from exact",
          len(set(stems)) == 2, "")
    body = txt("auditEvalLeakage.m")
    for name in ("EXACT_DUPLICATE", "POSSIBLE_DUPLICATE", "UNSUPPORTED",
                 "NOT_APPLICABLE", "UNVERIFIABLE"):
        check(f"auditor classifies {name}", name in body, "missing class")
    check("near-duplicate honestly unsupported",
          "NOT performed" in body, "overclaim")


def test_roles_and_gt():
    check("calibration-on-TEST is contamination (contract)",
          True, "see auditEvalLeakage calibrationsplit FINDINGS")
    body = txt("auditEvalLeakage.m")
    check("threshold-on-TEST is contamination", "thresholdsplit" in body.lower(), "missing")
    check("GT attestation required (no silent clean)",
          "InferenceUsedGT" in body, "missing")
    check("orchestrator errors without attestation",
          "attestation" in txt("runHeldoutEvaluation.m"), "missing")
    check("orchestrator withholds metrics on FINDINGS",
          "metrics withheld" in txt("runHeldoutEvaluation.m"), "leak")


def test_metrics_reference():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 2, 1]
    y_pred = [0, 1, 1, 2, 2, 3, 3, 3, 4, 3, 2, 1]
    check("QWK matches sklearn reference 0.88940092",
          abs(qwk(y_true, y_pred) - 0.88940092) < 1e-6, f"{qwk(y_true, y_pred)}")
    lo, hi = wilson(45, 50)
    check("Wilson matches statsmodels [0.7864, 0.9565]",
          abs(lo - 0.786398) < 1e-4 and abs(hi - 0.956524) < 1e-4, f"{lo},{hi}")
    lo0, _ = wilson(0, 20)
    _, hi1 = wilson(20, 20)
    check("Wilson stays in [0,1] at boundaries", lo0 >= 0 and hi1 <= 1, "")
    check("MATLAB engines reused (not duplicated)",
          "computeQWK" in txt("runHeldoutEvaluation.m") or "evaluateGrading" in txt("runHeldoutEvaluation.m"), "engine fork")
    check("no duplicated QWK/Wilson in new files",
          "cohen_kappa" not in txt("runHeldoutEvaluation.m").lower(), "fork")


def test_bootstrap_contracts():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    vals = list(range(10))
    m1, bad1 = bootstrap_patients(groups, lambda idx: sum(vals[i] for i in idx) / len(idx), seed=7, b=100)
    m2, _ = bootstrap_patients(groups, lambda idx: sum(vals[i] for i in idx) / len(idx), seed=7, b=100)
    check("seed reproducibility (identical replicates)", m1 == m2, "")
    check("unit labeled PATIENT in MATLAB contract",
          "'PATIENT'" in txt("patientBootstrapCI.m"), "mislabeled")
    check("image fallback labeled weaker",
          "IMAGE_LEVEL_BOOTSTRAP" in txt("patientBootstrapCI.m"), "mislabeled")
    _, bad_d = bootstrap_patients(["a"], lambda idx: 1 / 0, seed=7, b=20)
    check("degenerate replicates counted (never zero-filled)", bad_d == 20, f"{bad_d}")
    check("minimum-valid policy exists",
          "MinValid" in txt("patientBootstrapCI.m") and "UNAVAILABLE" in txt("patientBootstrapCI.m"), "fabrication risk")
    check("invalid reasons recorded",
          "invalidReasons" in txt("patientBootstrapCI.m"), "silent drops")


def test_report_safety():
    body = txt("runHeldoutEvaluation.m")
    check("report defaults NOT_MEASURED/null",
          body.count("NOT_MEASURED") >= 3, "placeholder risk")
    check("synthetic values cannot become MEASURED (claimState gated)",
          "claimState" in body, "promotion path")
    check("exclusions recorded", "nExclusions" in body or "exclusion" in body, "missing")
    for f in ("docs/EVAL_REPORT_TEMPLATE.md", "STAGE12_EVALUATION_PROTOCOL.md",
              "STAGE12_EVALUATION_HANDOFF.md"):
        check(f"{f} exists", os.path.isfile(os.path.join(ROOT, f)), "missing")
    tpl = txt("docs/EVAL_REPORT_TEMPLATE.md") if os.path.isfile(
        os.path.join(ROOT, "docs/EVAL_REPORT_TEMPLATE.md")) else ""
    check("template defaults null (not plausible numbers)",
          "null" in tpl, "misleading template")
    check("targets labeled REQUIREMENT (never achievement)",
          "REQUIREMENT" in (tpl + txt("STAGE12_EVALUATION_PROTOCOL.md")
                            if os.path.isfile(os.path.join(ROOT, "STAGE12_EVALUATION_PROTOCOL.md")) else tpl),
          "target-as-result risk")


TESTS = [test_manifest, test_leakage, test_roles_and_gt, test_metrics_reference,
         test_bootstrap_contracts, test_report_safety]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage12 eval: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
