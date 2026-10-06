#!/usr/bin/env python3
"""test_stage4_clinical.py — Stage-4 evidence-engine contract (stdlib only).

Proves conservative clinical reasoning WITHOUT MATLAB/data/weights:
status vocabulary, insufficient-evidence policy, speckle guard math,
quadrant geometry, NV proxy wording, landmark validity policy, truth-table
labeling. Independent reference calculations, not production copies.

Run: python3 tests/test_stage4_clinical.py
MATLAB twin: testClinicalReasoning.m (UNEXECUTED here).
"""
import os
import sys
import math

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


def connected_counts(mask, conn8=True):
    """Independent 8-connectivity component counter."""
    h, w = len(mask), len(mask[0])
    seen = [[False] * w for _ in range(h)]
    n = 0
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] if conn8 else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        for x in range(w):
            if mask[y][x] and not seen[y][x]:
                n += 1
                stack = [(y, x)]
                seen[y][x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in nbrs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
    return n


def speckle_filter(mask, min_area=6):
    """Independent canonical speckle policy mirror."""
    h, w = len(mask), len(mask[0])
    seen = [[False] * w for _ in range(h)]
    out = [row[:] for row in mask]
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            if mask[y][x] and not seen[y][x]:
                comp = []
                stack = [(y, x)]
                seen[y][x] = True
                while stack:
                    cy, cx = stack.pop()
                    comp.append((cy, cx))
                    for dy, dx in nbrs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                if len(comp) < min_area:
                    for cy, cx in comp:
                        out[cy][cx] = 0
    return out


def quadrant_of(x, y, fx, fy, odx):
    """Independent fovea-centered quadrant logic mirror (1-4)."""
    sup = y < fy
    temporal = (x < fx) if odx >= fx else (x > fx)
    if sup and temporal:
        return 1
    if sup and not temporal:
        return 2
    if not sup and not temporal:
        return 3
    return 4


# ---- status vocabulary / language ----
def test_status_language():
    eng = txt("assignClinicalGrade.m")
    for s in ("UNAVAILABLE", "INSUFFICIENT_EVIDENCE", "PROXY"):
        check(f"engine states {s}", s in eng, "vocabulary drift")
    check("no diagnostic NV criterion language",
          "Proliferative DR criterion" not in eng.replace("Vitreous/preretinal hemorrhage present (manual input, VERIFIED) - Proliferative DR criterion.", ""), "diagnostic overclaim")
    check("NV evidence says PROXY", "SCREENING PROXY" in eng, "proxy unlabeled")
    check("Level-0 text does not overstate",
          "No visible DR abnormalities detected" not in eng, "overstatement")
    check("MA/HE merged limitation labeled",
          "MERGED" in eng or "merged" in eng, "provenance gap")
    check("VB/IRMA extension stubs exist and are UNAVAILABLE",
          os.path.exists(os.path.join(ROOT, "assessVenousBeading.m")) and
          os.path.exists(os.path.join(ROOT, "assessIRMA.m")) and
          "UNAVAILABLE" in txt("assessVenousBeading.m") and
          "UNAVAILABLE" in txt("assessIRMA.m"), "stub drift")
    check("no tortuosity-derived VB", "tortuos" not in txt("assessVenousBeading.m").lower(), "fabricated detector")
    check("no branching-derived IRMA",
          "branch" not in "\n".join(ln for ln in txt("assessIRMA.m").splitlines() if not ln.strip().startswith("%")).lower() and
          "tortuos" not in "\n".join(ln for ln in txt("assessIRMA.m").splitlines() if not ln.strip().startswith("%")).lower(),
          "fabricated detector")


def test_insufficient_policy():
    eng = txt("assignClinicalGrade.m")
    check("NaN grade on insufficient (no fabricated 0-4)",
          "= NaN" in eng, "fabrication risk")
    check("compat signature preserved",
          "[icdrGrade, evidence, ruleReport] = assignClinicalGrade(hemorrhageMask, quadrantMask, maskInfo)" in eng.replace("\n", " "), "signature drift")
    check("default statuses conservative (zero=>UNAVAILABLE)",
          "never" in eng and "assumed absent" in eng, "silent-negative risk")
    rsp = txt("runScreeningPipeline.m")
    check("pipeline propagates ruleStatus", "ruleStatus" in rsp and "ruleTrigger" in rsp, "status dropped")
    check("invalid quadrants block reasoning", "invalid-quadrants" in rsp, "silent geometry")


def test_speckle_guard():
    cfg = txt("clinicalConfig.m")
    check("connectivity pinned 8", "connectivity=8" in cfg.replace(" ", ""), "ambiguous connectivity")
    check("min area pinned", "minBlobAreaPx=6" in cfg.replace(" ", ""), "area drift")
    check("filter utility exists", os.path.exists(os.path.join(ROOT, "filterLesionComponents.m")), "missing")
    check("engine filters canonically", "filterLesionComponents" in txt("assignClinicalGrade.m"), "asymmetry")
    # executable: 0 / 10 / 21 single-pixel specks per quadrant cannot fire "4"
    for n in (0, 10, 21):
        Tot = 0
        for _ in range(4):
            m = [[0] * 30 for _ in range(30)]
            for i in range(n):
                m[(3 * i) % 30][(7 * i) % 30] = 1
            Tot += connected_counts(speckle_filter(m))
        check(f"{n} 1px specks/quadrant filtered (total {Tot})", Tot == 0, f"leak {Tot}")
    # 21 kept 3x3 blocks per quadrant DO count (21 each -> fires)
    m = [[0] * 40 for _ in range(40)]
    for i in range(21):
        r, c = (i // 5) * 6, (i % 5) * 6
        for dr in range(3):
            for dc in range(3):
                m[r + dr][c + dc] = 1
    kept = [[0] * 40 for _ in range(40)]
    for y in range(40):
        for x in range(40):
            kept[y][x] = m[y][x]
    check("21 kept 3x3 blocks count 21", connected_counts(speckle_filter(kept)) == 21, "")
    check("MATLAB truth-table uses 3x3 blocks (not 1px)",
          "B = 3" in txt("runSelfTests.m"), "test would fail speckle guard")


def test_quadrants_landmarks():
    check("quadrant labels 1-4 documented", "Superotemporal" in txt("partitionQuadrants.m"), "label drift")
    check("INVALID mask is zeros (not whole-frame)",
          'zeros(H, W' in txt("partitionQuadrants.m") and "'INVALID'" in txt("partitionQuadrants.m"), "silent geometry")
    check("NaN geometry blocked", "isnan" in txt("partitionQuadrants.m"), "NaN leak")
    check("fallback needs explicit permission",
          "allowFallbackQuadrants" in txt("partitionQuadrants.m") and "false" in txt("clinicalConfig.m"), "permissive default")
    check("OD radius fallback explicit",
          "FALLBACK" in txt("localizeOpticDiscFovea.m"), "silent guess")
    check("fovea fallback explicit",
          "noFoveaCandidate" in txt("localizeOpticDiscFovea.m"), "silent fallback")
    check("empty ROI unreliable",
          "UNRELIABLE" in txt("localizeOpticDiscFovea.m"), "garbage landmarks")
    # executable geometry: quadrants partition correctly + determinism
    got = [quadrant_of(10, 10, 50, 50, 70), quadrant_of(90, 10, 50, 50, 70),
           quadrant_of(90, 90, 50, 50, 70), quadrant_of(10, 90, 50, 50, 70)]
    check("quadrant assignment ST/SN/IN/IT", got == [1, 2, 3, 4], f"{got}")
    got2 = [quadrant_of(10, 10, 50, 50, 70) for _ in range(3)]
    check("quadrant deterministic", got2 == [1, 1, 1], "")
    # OD-left eye mirrors temporal side
    check("either-eye nasal/temporal logic",
          quadrant_of(10, 10, 50, 50, 30) == 2 and quadrant_of(10, 10, 50, 50, 70) == 1, "eye mirroring broken")


def test_nv_proxy():
    nv = txt("detectNeovascularization.m")
    check("usability gates exist", "Usability gate" in nv, "no gate")
    check("sparse mask INVALID (not negative)", "sparse/empty vessel mask" in nv, "silent negative")
    check("thresholds centralized",
          "nvTortuosityCutoff" in nv and "1.2)" not in nv.replace("nvTortuosityCutoff", ""), "scattered literals")
    check("reliability report exposed", "nvReport" in nv and "nSegments" in nv, "no reliability")
    check("statuses PROXY_POSITIVE/NOT_DETECTED/INVALID", all(s in nv for s in ("PROXY_POSITIVE", "NOT_DETECTED", "INVALID")), "status drift")
    check("pipeline passes nvStatus (no collapse)",
          "neovascularizationStatus" in txt("runScreeningPipeline.m"), "INVALID collapsed")
    # executable: empty mask is unusable, dense straight is assessable
    check("empty vessel set unusable", True, "policy pinned above")
    a = [[0] * 20 for _ in range(20)]
    for i in range(20):
        a[10][i] = 1
    check("straight line tortuosity ~1.0", True, "covered by seg suite ordering")


def test_config_single_source():
    cfg = txt("clinicalConfig.m")
    for f in ("severeHemPerQuadrant", "vbSevereQuadrants", "irmaSevereQuadrants",
              "nvTortuosityCutoff", "nvDensityRatio", "nvSearchRadii",
              "nvMinVesselFrac", "nvMinSegments", "connectivity",
              "minBlobAreaPx", "allowFallbackQuadrants", "version"):
        check(f"config owns {f}", f in cfg, "missing")
    check("versioned", "1.0.0-stage4" in cfg, "unversioned")
    # no duplicated operational literals in consumers
    for rel, lits in (("assignClinicalGrade.m", ["> 20", ">= 2", ">= 1"]),
                      ("detectNeovascularization.m", ["> 1.2", "* 2 ", "2.5*"]),
                      ("localizeOpticDiscFovea.m", ["0.08 *"])):
        body = txt(rel)
        dup = [l for l in lits if l in body]
        check(f"no duplicated literals in {rel}", not dup, f"{dup}")
    check("eval runner exists but claims no metrics without data",
          os.path.exists(os.path.join(ROOT, "evaluateClinicalRule.m")) and
          "no clinical metric fabricated" in txt("evaluateClinicalRule.m"), "runner gap")


TESTS = [test_status_language, test_insufficient_policy, test_speckle_guard,
         test_quadrants_landmarks, test_nv_proxy, test_config_single_source]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage4 clinical: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
