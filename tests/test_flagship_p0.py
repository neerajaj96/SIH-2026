#!/usr/bin/env python3
"""test_flagship_p0.py — Wave-0 P0 regression (stdlib only).

Pins the six P0 blocker fixes so they cannot silently regress:
 A1 netrasetu.html tail/IIFE structure + JS syntax-critical shape
 A2 runHeldoutEvaluation localMeasured + splitRoles guards
 A3 patientBootstrapCI with-replacement multiplicity (no ismember dedup)
 A4 wilsonScoreInterval n==0 -> [NaN NaN] (no crash)
 A5 bridge forwards nvStatus (legacy nv preserved)
 A6 CI gates the previously-omitted suites

All values are TEST/LOGIC fixtures. No clinical/MATLAB execution claimed.
Run: python3 tests/test_flagship_p0.py
"""
import math
import os
import random
import re
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


def test_a1_tail():
    h = txt("netrasetu.html")
    m = re.search(r"<script>(.*)</script>", h, re.S)
    check("A1. script block present", m is not None, "no inline script")
    if not m:
        return
    js = m.group(1)
    check("A1. single outer IIFE open", js.count("(function(){") == 1,
          f"opens={js.count('(function(){')}")
    # one close for async-arrow dbReady IIFE + one for outer IIFE
    check("A1. balanced IIFE closes", js.count("})();") == 2,
          f"closes={js.count('})();')}")
    check("A1. no corrupt double-init tail", "})();initQueue();" not in js,
          "corrupt tail still present")
    check("A1. startup calls present",
          "checkBridge();" in js and "initQueue();" in js, "missing startup")
    check("A1. tail ends cleanly", js.strip().endswith("})();"),
          repr(js.strip()[-60:]))
    nob = js.count("{")
    ncb = js.count("}")
    check("A1. braces balanced", nob == ncb, "open=%d close=%d" % (nob, ncb))
    nop = js.count("(")
    ncp = js.count(")")
    check("A1. parens balanced", nop == ncp, "open=%d close=%d" % (nop, ncp))


def test_a2_heldout():
    m = txt("runHeldoutEvaluation.m")
    check("A2. localMeasured defined", "function m = localMeasured" in m,
          "missing helper")
    check("A2. MEASURED wrapped with source",
          "'MEASURED'" in m and "source" in m, "state not explicit")
    check("A2. splitRoles guarded", "isfield(predictions, 'splitRoles')" in m,
          "unguarded splitRoles")
    check("A2. imageId/icdr guarded", "predictions.imageId" in m and "isfield(predictions, 'imageId')" in m,
          "unguarded predictions")
    check("A2. UNVERIFIABLE policy present", "UNVERIFIABLE" in m,
          "silent UNVERIFIABLE")
    check("A2. nExclusions populated", "nExclusions" in m and "exclusion" in m,
          "exclusions dropped")
    check("A2. no bare localMeasured call without def",
          m.count("localMeasured(") >= 3 and "function m = localMeasured" in m,
          "call/def mismatch")


def test_a3_bootstrap():
    m = txt("patientBootstrapCI.m")
    check("A3. dedup removed",
          "sel = ismember(unitIds, draw)" not in m, "biased dedup still present")
    check("A3. per-unit row index", "unitRows" in m, "no multiplicity structure")
    check("A3. draws expanded with repetition",
          "unitRows{draw(k)}" in m or "unitRows(draw" in m, "no expansion")
    check("A3. seeded RandStream preserved", "RandStream('mt19937ar'" in m,
          "seed behavior changed")
    check("A3. invalid-replicate accounting kept",
          "undefined-domain" in m and "metric-error" in m, "accounting lost")
    check("A3. minimum-valid policy kept", "minValid" in m and "UNAVAILABLE" in m,
          "CI gate lost")
    # Python mirror of the fixed resampling: multiplicity preserved
    unit_ids = [1, 1, 2, 3, 3, 3]  # AAxBCCC layout
    unit_rows = {u: [i for i, v in enumerate(unit_ids) if v == u] for u in (1, 2, 3)}
    draw = [1, 1, 3]  # A drawn twice, C once
    rows = []
    for d in draw:
        rows.extend(unit_rows[d])
    check("A3. duplicate draw keeps multiplicity",
          sorted(rows) == [0, 0, 1, 1, 3, 4, 5], f"rows={rows}")
    check("A3. dedup would lose rows",
          len(rows) == 7 and len(set(rows)) == 5, "mirror sanity")
    # seeded determinism mirror
    r1 = random.Random(42)
    r2 = random.Random(42)
    s1 = [r1.randint(1, 3) for _ in range(5)]
    s2 = [r2.randint(1, 3) for _ in range(5)]
    check("A3. same seed same sample", s1 == s2, f"{s1} vs {s2}")
    r3 = random.Random(43)
    s3 = [r3.randint(1, 3) for _ in range(200)]
    check("A3. different seed can differ", s3 != s1 * 40, "seed ignored")


def wilson_py(successes, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(center - half, 0.0), min(center + half, 1.0))


def test_a4_wilson():
    m = txt("wilsonScoreInterval.m")
    check("A4. n==0 returns NaN (no error)",
          "if n == 0" in m and "lo = NaN; hi = NaN" in m, "crash path remains")
    check("A4. negative n still errors",
          "invalidN" in m, "validation lost")
    check("A4. toolbox labeled", "norminv" in m, "dependency unlabeled")
    lo, hi = wilson_py(0, 0)
    check("A4. mirror n==0 is NaN (UNAVAILABLE)",
          math.isnan(lo) and math.isnan(hi), f"got {(lo, hi)}")
    lo2, hi2 = wilson_py(8, 10)
    check("A4. valid interval unchanged",
          0.0 <= lo2 <= hi2 <= 1.0 and abs(lo2 - 0.488) < 0.05,
          f"got {(lo2, hi2)}")
    g = txt("evaluateGrading.m")
    check("A4. grading relies on Wilson (no duplicate math)",
          "wilsonScoreInterval(tp" in g and "wilsonScoreInterval(tn" in g,
          "caller drift")


def test_a5_bridge():
    b = txt("bridge_server.py")
    check("A5. legacy nv bool preserved", '"nv"' in b and "safe_bool" in b,
          "legacy break")
    check("A5. nvStatus forwarded", '"nvStatus"' in b, "key missing")
    check("A5. qualityDecision parity", '"qualityDecision"' in b, "parity gap")
    check("A5. qualityReasons parity", '"qualityReasons"' in b, "parity gap")
    check("A5. qualityGuidance parity", '"qualityGuidance"' in b, "parity gap")
    check("A5. qualityCalibrated parity", '"qualityCalibrated"' in b, "parity gap")
    check("A5. ruleStatus parity", '"ruleStatus"' in b, "parity gap")
    check("A5. serialization uses safe_get", "safe_get" in b, "KeyError risk")
    s = txt("screenOneImage.m")
    check("A5. MATLAB emits nvStatus", "'nvStatus'" in s or "nvStatus" in s,
          "producer missing")


def test_a6_ci():
    y = txt(".github/workflows/python-contracts.yml")
    for suite in ("test_stage5_explain.py", "test_stage6_runtime.py",
                  "test_stage11_runtime.py", "test_stage12_eval.py",
                  "test_stage13_data.py"):
        check(f"A6. CI gates {suite}", suite in y, "suite omitted (hides red)")
    check("A6. MATLAB stays non-gating",
          "if: false" in y and "MATLAB" in y, "fake MATLAB green risk")
    check("A6. seg-gap documented", "segmentation" in y.lower() or "seg " in y.lower()
          or "seg-mirror" in y.lower() or "seg mirrors" in y.lower(),
          "gap undocumented")


TESTS = [test_a1_tail, test_a2_heldout, test_a3_bootstrap,
         test_a4_wilson, test_a5_bridge, test_a6_ci]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== flagship P0: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
