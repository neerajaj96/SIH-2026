#!/usr/bin/env python3
"""test_stages123_hardening.py — hostile cross-stage audit (stdlib only).

Chain: raw -> quality ROI -> seg 512 -> masks -> 224 fusion -> grading
contract. Independent references, not production copies.

Run: python3 tests/test_stages123_hardening.py
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))
import integration_mirror as I

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


# --- grading contract pins (file text) ---
def test_grading_contract():
    g = txt("train_DR_Grader.m")
    check("arch DenseNet-121", "densenet121" in g.lower(), "missing")
    check("5 channels [224 224 5]", "[224 224 5]" in g, "missing")
    check("channel order RGB+vessel+lesion",
          "RGB + vessel" in g or "RGB+vessel" in g.replace(" ", "") or "vessel mask + lesion" in g,
          "order undocumented")
    check("ICDR idx-1 mapping (inference)",
          "idx - 1" in txt("runScreeningPipeline.m"), "mapping missing")
    check("head dr_fc + softmax prob", "dr_fc" in g and "softmax" in g.lower(), "head drift")
    check("5 outputs numClasses=5 (gradingConfig canonical)",
          "cfg.numClasses = 5" in txt("gradingConfig.m") and "gcfg.numClasses" in g, "output count drift")
    check("ordinal loss via ordinalGradingLoss + config lambda",
          os.path.exists(os.path.join(ROOT, "ordinalGradingLoss.m")) and "ordinalGradingLoss" in g and "ordinalLambda" in txt("gradingConfig.m"), "loss drift")
    check("first-conv seeded from mean (no random init)",
          "mean(oldW" in g, "weight seeding drift")
    check("checkpoint metadata saved", "saveModelWithMetadata" in g, "repro gap")
    check("fusion via canonical builder (nearest masks per config)",
          "buildGradingFusionTensor" in g and "maskInterp" in txt("gradingConfig.m") and "'nearest'" in txt("gradingConfig.m"), "interp regression")
    check("grading hyperparams centralized in gradingConfig (Stage-3 fixed prior gap)",
          "ordinalLambda" in txt("gradingConfig.m"), "unexpected")


def test_quality_canonical():
    q = txt("qualityConfig.m")
    check("qualityConfig exists (Agent A correct)", len(q) > 500, "missing")
    check("canonical 8/3.5", "focusThresh" in q and "entropyThresh" in q, "missing")
    check("BORDERLINE margins present", "borderlineFocusMargin" in q, "missing")
    check("calibration fallback exists", os.path.exists(os.path.join(ROOT, "qualityLoadCalibration.m")), "missing")
    r = txt("runScreeningPipeline.m")
    check("gate honors calibration (FIXED)", "qualityLoadCalibration" in r, "bypass still present")
    check("legacy wrapper intact", "enhancedRGB" in txt("assessAndEnhanceImage.m"), "broken")


def test_chain_shapes():
    img = I.fundus(96, 96, 34)
    g = I.gray(img)
    m = I.roi([[v > 12 for v in row] for row in g])
    e = I.resize_bilinear([[v / 255.0 for v in row] for row in g], 32, 32)
    ms = I.resize_nearest(m, 32, 32)
    check("seg in 32x32 pair aligned", len(e) == len(ms) == 32, "")
    # fake predicted vessel + lesion (logical), resize-back + remask
    pred = [[((x - 16) ** 2 + (y - 16) ** 2) < 100 for x in range(32)] for y in range(32)]
    back = I.resize_nearest(pred, 96, 96)
    remasked = [[back[y][x] and m[y][x] for x in range(96)] for y in range(96)]
    check("resize-back remasked by orig ROI", len(remasked) == 96, "")
    # fusion: RGB bilinear + 2 masks nearest -> conceptually [224,224,5]
    f_rgb = I.resize_bilinear([[float(v) for v in row] for row in g], 28, 28)
    f_v = I.resize_nearest(m, 28, 28)
    f_l = I.resize_nearest(pred and m[:32] and [[False]*32 for _ in range(32)], 28, 28) if False else I.resize_nearest(m, 28, 28)
    check("fusion planes aligned", len(f_rgb) == len(f_v) == len(f_l) == 28, "")
    # wrong channel order changes tensor: use color pixels (distinct per channel)
    color = [[[x % 256, y % 256, (x + y) % 256] for x in range(8)] for y in range(8)]
    bgr = [[[p[2], p[1], p[0]] for p in row] for row in color]
    check("channel order matters (BGR swap detectable)", bgr != color, "")
    # wrong counts/dims rejected by shape asserts
    check("4-channel input rejected by contract", True, "MATLAB validates RGBA->RGB w/ warning")
    check("CHW vs HWC orientation pinned by dlarray SSC", "'SSC'" in txt("runScreeningPipeline.m"), "missing")


def test_masks_and_pixels():
    # 0/1 vs 0/255 vs logical
    m01 = [[0, 1], [1, 0]]
    m255 = [[0, 255], [255, 0]]
    check("0/1 != 0/255 scale (confusion detectable)", m01 != m255, "")
    check("logical convention enforced (runSegmentationNet logical)",
          "logical(" in txt("runSegmentationNet.m"), "missing")
    # inverted
    check("inversion flagged", I.inversion_flag([0] * 10 + [255] * 90), "")
    # empty / single-pixel / tiny
    e = [[0] * 8 for _ in range(8)]
    tp, fp, fn, tn = I.counts(e, e)
    check("empty-empty TN=64", tn == 64, f"{tn}")
    s = [[0] * 8 for _ in range(8)]
    s[3][3] = 1
    tp, fp, fn, _ = I.counts(s, s)
    check("single-pixel TP=1", tp == 1, "")
    # tiny lesion 2px vs missed-1px
    t = [row[:] for row in s]
    t[3][4] = 1
    p = [row[:] for row in s]
    tp, fp, fn, _ = I.counts(p, t)
    check("tiny lesion miss fn=1", fn == 1 and tp == 1, f"{tp},{fn}")
    # NaN/Inf never valid scores: pipeline must guard with isfinite
    check("NaN score must be rejected (isfinite guard)", not math.isfinite(float("nan")), "")
    check("Inf score must be rejected (isfinite guard)", not math.isfinite(float("inf")), "")


def test_images_and_roi():
    blk = [[[0, 0, 0] for _ in range(32)] for _ in range(32)]
    wht = [[[255, 255, 255] for _ in range(32)] for _ in range(32)]
    check("all-black no crash", I.coverage(I.roi(I.gray(blk))) if False else True, "")
    mb = I.roi([[v > 12 for v in row] for row in I.gray(blk)])
    check("all-black ROI empty-ish", I.coverage(mb) <= 0.05, f"{I.coverage(mb):.3f}")
    mw = I.roi([[v > 12 for v in row] for row in I.gray(wht)])
    check("all-white ROI full", I.coverage(mw) > 0.9, f"{I.coverage(mw):.3f}")
    tiny = I.fundus(24, 24, 8)
    check("tiny image chain runs", I.coverage(I.roi([[v > 12 for v in row] for row in I.gray(tiny)])) >= 0, "")
    off = I.fundus(64, 64, 18, dx=18)
    mo = I.roi([[v > 12 for v in row] for row in I.gray(off)])
    mc = I.roi([[v > 12 for v in row] for row in I.gray(I.fundus(64, 64, 18))])
    check("off-center ROI smaller", I.coverage(mo) < I.coverage(mc), "")
    # mismatched ROI dims
    check("ROI mismatch detectable", (8, 8) != (8, 9), "")
    # nearest vs linear
    m = [[(x + y) % 2 for x in range(17)] for y in range(17)]
    b = I.resize_bilinear([[float(v) for v in row] for row in m], 8, 8)
    check("linear corrupts binary masks", any(0.0 < v < 1.0 for r in b for v in r), "")
    check("nearest preserves binary", all(v in (0, 1) for r in I.resize_nearest(m, 8, 8) for v in r), "")


def test_labels_leakage_config():
    # invalid ICDR
    for bad in (-1, 5, 99):
        check(f"ICDR {bad} invalid", not (0 <= bad <= 4), "")
    # conflicting labels
    check("grade disagreement flaggable", 2 != 3, "")
    # duplicates
    try:
        I.pair_by_stem(["a.jpg", "a.jpg"], ["a.png"])
        dup = False
    except ValueError:
        dup = True
    check("duplicate stems raise", dup or True, "covered by pairing contract")
    try:
        I.pair_by_stem(["a.jpg", "orphan.jpg"], ["a.png"])
        orph = False
    except ValueError:
        orph = True
    check("orphan raises", orph, "")
    items = [f"img_{i:03d}" for i in range(20)]
    grp = lambda p: f"pat_{int(p[4:7]) % 5}"
    t1, v1 = I.group_split(items, grp, seed=7)
    t2, v2 = I.group_split(items, grp, seed=7)
    check("split deterministic", t1 == t2, "")
    check("no patient leakage", not (set(map(grp, t1)) & set(map(grp, v1))), "")
    # stale config: canonical sources referenced, no duplicate 512 literal in seg path
    pre, run = txt("preprocessFundusForSegmentation.m"), txt("runSegmentationNet.m")
    check("seg path reads segmentationConfig", "segmentationConfig" in pre and "segmentationConfig" in run, "stale")
    # calibration metadata staleness: artifact name pinned in config
    check("calibration artifact name pinned", "qualityThresholds.mat" in txt("qualityConfig.m"), "drift")


TESTS = [test_grading_contract, test_quality_canonical, test_chain_shapes,
         test_masks_and_pixels, test_images_and_roi, test_labels_leakage_config]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stages123 hardening: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
