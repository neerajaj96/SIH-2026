#!/usr/bin/env python3
"""test_flagship_eval.py — Waves 4/5/6 evaluation+provenance regression.

Pins: computed lineage (hashArtifacts), REJECTED-block, parent/stage/hash
conventions, manifest structured provenance, RandStream split (no global
rng side-effect) + stratification option, telemed single-source
referableFraction, UNVERIFIABLE-with-limitations policy, Wilson NaN mapping,
bootstrap multiplicity (mirror), Toolbox labeling honesty.

Run: python3 tests/test_flagship_eval.py
MATLAB execution remains MATLAB-GATED; these are contract pins.
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


def test_provenance_chain():
    h = txt("hashArtifacts.m")
    check("manifest hashing", "'manifest'" in h and "imageId" in h, "not computed")
    check("file hashing reads bytes", "fread" in h, "not computed")
    check("method recorded", "sha256:" in h and "fnv1a:" in h, "strength hidden")
    check("timestamps never hashed", "NEVER" in h or "never" in h.lower(), "nondeterminism")
    check("missing file errors (no silent ATTESTED)", "does not exist" in h, "silent gap")
    f = txt("freezeCandidate.m")
    check("ATTESTED_NOT_COMPUTED disclosed", f.count("ATTESTED_NOT_COMPUTED") >= 2, "false lineage")
    check("REJECTED blocks", "rejectedDataset" in f, "silent freeze")
    check("override audited", "overrideRejected" in f and "OVERRIDE" in f, "unaudited")
    check("stage map documented", "EVALUATION_READY" in f, "stage gap")


def test_manifest_split():
    m = txt("evaluationManifest.m")
    check("graderCount structured", "graderCount" in m, "free-text only")
    check("adjudicationRule structured", "adjudicationRule" in m, "free-text only")
    check("labelDate structured", "labelDate" in m, "free-text only")
    check("unknown stays NaN (no fabrication)", "NaN" in m, "fabrication risk")
    s = txt("buildPatientLevelSplit.m")
    check("RandStream (no global rng)", "RandStream('mt19937ar'" in s, "global side-effect")
    check("no bare rng(seed)", "rng(seed)" not in s, "side-effect remains")
    check("stratified option", "labels" in s and "stratified" in s.lower(), "starvation risk")
    check("unstratified limitation printed", "unstratified" in s.lower(), "silent skew")
    check("class balance reported", "label %g" in s or "train /" in s, "blind split")


def test_eval_semantics():
    r = txt("runHeldoutEvaluation.m")
    check("MEASURED wrapped", "localMeasured" in r, "bare values")
    check("UNVERIFIABLE limitation", "UNVERIFIABLE" in r, "silent policy")
    check("FINDINGS withhold", "withheld" in r, "leak then measure")
    check("nExclusions filled", "nExclusions = sum" in r or "nExclusions=sum" in r.replace(" ", ""), "missing")
    w = txt("wilsonScoreInterval.m")
    check("n==0 NaN (UNAVAILABLE)", "if n == 0" in w and "NaN" in w, "crash")
    check("toolbox labeled", "norminv" in w, "unlabeled dep")
    p = txt("patientBootstrapCI.m")
    check("with-replacement rows", "unitRows{draw(k)}" in p or "unitRows" in p, "dedup bias")
    check("invalid accounting", "undefined-domain" in p, "silent NaN")
    check("min-valid gate", "minValid" in p, "fabricated band")


def test_telemed_single_source():
    c = txt("telemedConfig.m")
    check("referable derived", "sum(cfg.severityMix(3:5))" in c, "magic fraction")
    check("SCENARIO tagged", "SCENARIO" in c, "unlabeled assumption")
    o = txt("optimizeResourceAllocation.m")
    check("optimizer reads config default", "telemedConfig().referableFraction" in o, "forked default")
    check("no staffing overclaim", "SCENARIO" in o or "sensitivity" in o.lower(), "field claim")


TESTS = [test_provenance_chain, test_manifest_split, test_eval_semantics,
         test_telemed_single_source]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== flagship eval: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
