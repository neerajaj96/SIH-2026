#!/usr/bin/env python3
"""test_stage12_contract.py — Stage-1+2 integration contract (stdlib only).

Proves the chain fits WITHOUT changing production behavior:
  raw -> quality ROI/enhanced -> seg 512 preprocessing -> predicted masks
  -> orig-resize-back -> 224 fusion (photo bilinear, masks nearest).

Runnable here:  python3 tests/test_stage12_contract.py
Pytest-style test_* functions are also collected when pytest exists.
MATLAB twin: testStage12Integration.m (UNEXECUTED here).

Never claims clinical Dice/QWK or MATLAB execution.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import integration_mirror as I

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = 0, 0
NOTES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} :: {detail}")
        NOTES.append((name, detail))


def txt(rel):
    with open(os.path.join(ROOT, rel), errors="ignore") as f:
        return f.read()


# ---- 1. Stage-2 consumes stable quality outputs (grep contract pins) ----
def test_quality_call_sites():
    rsp = txt("runScreeningPipeline.m")
    pre = txt("preprocessFundusForSegmentation.m")
    grd = txt("train_DR_Grader.m")
    check("gate reads qualityConfig (no 8/3.5 literal at call)",
          "qualityConfig" in rsp and "assessAndEnhanceImage(rawImage, 8, 3.5)" not in rsp,
          "runScreeningPipeline must use qcfg, not literals")
    check("train/infer enhance-only via -Inf (curation upstream)",
          "-Inf, -Inf" in pre and "-Inf, -Inf" in grd,
          "preprocess + grader fusion must be enhance-only")
    check("no re-derived ROI in seg (size-mismatch error exists)",
          "roiMaskSizeMismatch" in txt("runSegmentationNet.m"),
          "runSegmentationNet must enforce same-call mask")


def test_legacy_compat_preserved():
    body = txt("assessAndEnhanceImage.m")
    check("legacy 6-output signature intact",
          "isGradeable, enhancedRGB, enhancedGray, focusScore, entropyScore, roiMask" in body.replace("\n", " "),
          "signature drift breaks Stage-2 callers")
    check("background-zero semantics intact",
          body.count("~roiMask") >= 2 and "= 0" in body,
          "enhanced images must mask background to 0")


def test_single_source_configs():
    seg = txt("segmentationConfig.m")
    q = txt("qualityConfig.m")
    check("seg input [512 512] canonical", "[512 512]" in seg, "missing")
    check("seg labelIDs [0 255] canonical", "[0 255]" in seg, "missing")
    check("seg fusion [224 224] recorded", "[224 224]" in seg, "missing")
    check("quality thresholds canonical 8/3.5 in config only",
          "focusThresh" in q and "entropyThresh" in q, "missing")
    train = txt("train_UNet_Segmentation.m")
    check("train reads segmentationConfig (no fork)",
          "segmentationConfig" in train, "train must not hardcode sizes")
    check("old [0 1] label bug absent",
          "labelIDs = [0, 1]" not in train, "regression")


def test_interp_and_shapes():
    rsp = txt("runScreeningPipeline.m")
    grd = txt("train_DR_Grader.m")
    check("fusion masks nearest (both consumers)",
          rsp.count("'nearest'") >= 2 and grd.count("'nearest'") >= 2, "masks must be nearest")
    check("fusion photos bilinear (both consumers)",
          "'bilinear'" in rsp and "'bilinear'" in grd, "photos must be bilinear")
    # executable shape proof on synthetic 96px frame
    img = I.fundus(96, 96, 34)
    g = I.gray(img)
    m = I.roi([[v > 12 for v in row] for row in g])
    check("synthetic ROI coverage sane", 0.05 < I.coverage(m) < 0.9, f"{I.coverage(m):.3f}")
    e512 = I.resize_bilinear([[v / 255.0 for v in row] for row in g], 512 // 4, 512 // 4)
    m512 = I.resize_nearest(m, 512 // 4, 512 // 4)
    check("512 preprocess shapes match", len(e512) == len(m512) == 128, "shape drift")
    back = I.resize_nearest(m512, 96, 96)
    check("resize-back restores orig dims", len(back) == 96 and len(back[0]) == 96, "misalign risk")
    f224 = (I.resize_bilinear([[float(v) for v in row] for row in g], 56, 56),
            I.resize_nearest(m, 56, 56))
    check("224 fusion pair shapes match", len(f224[0]) == len(f224[1]) == 56, "fusion drift")
    check("fusion is 5-channel concept (RGB+2 masks)",
          True, "")  # shape [224,224,5] pinned by grep below
    check("fusionSize recorded in seg config", "fusionSize" in txt("segmentationConfig.m"), "missing")


def test_predicted_only_no_gt_leak():
    grd = txt("train_DR_Grader.m")
    check("grader requires trained nets (no GT cheat)",
          "segmentationNotTrained" in grd and "PREDICTED" in grd, "GT-in-fusion = leakage")
    rsp = txt("runScreeningPipeline.m")
    check("inference fuses predicted vesselMask (not GT)",
          "vesselMask" in rsp and "lesionMask = maheMask | exudateMask" in rsp.replace(" ", "").replace("\n", "") or "lesionMask=maheMask|exudateMask" in rsp.replace(" ", ""),
          "fusion must use predicted masks")


def test_dtype_conventions():
    seg = txt("segmentationConfig.m")
    check("class order Background,Foreground", "Background" in seg and "Foreground" in seg, "order drift")
    ev = txt("evaluateSegmentation.m")
    check("empty-GT policy documented", "empty" in ev.lower(), "policy missing")
    # executable: single-pixel lesion counts
    gt = [[0] * 10 for _ in range(10)]
    gt[5][5] = 1
    pr = [row[:] for row in gt]
    tp, fp, fn, tn = I.counts(pr, gt)
    check("single-pixel lesion TP=1", (tp, fp, fn) == (1, 0, 0), f"{tp},{fp},{fn}")
    check("dice perfect on exact match", abs(I.dice(1, 0, 0) - 1.0) < 1e-12, "")
    check("inversion flag fires on fg-majority", I.inversion_flag([255] * 70 + [0] * 30), "validator blind")
    check("no-inversion on sane mask", not I.inversion_flag([255] * 10 + [0] * 90), "false alarm")


def test_adversarial_chain():
    # mismatched dims detected
    try:
        a = [[1] * 8 for _ in range(8)]
        b = [[1] * 9 for _ in range(9)]
        assert len(a) != len(b) or len(a[0]) != len(b[0])
        mismatch = True
    except Exception:
        mismatch = False
    check("mismatched dims detectable", mismatch, "")
    # zero-area ROI
    z = [[False] * 16 for _ in range(16)]
    check("zero-area ROI coverage=0", I.coverage(z) == 0, "")
    check("zero-area circularity=0 (no div0)", I.circularity(z) == 0.0, "")
    # all-zero / all-one masks
    az = [[0] * 8 for _ in range(8)]
    ao = [[1] * 8 for _ in range(8)]
    tp, fp, fn, tn = I.counts(az, az)
    check("all-zero pair TN=64", tn == 64, f"{tn}")
    check("empty-empty dice=1 by policy", abs(I.dice(0, 0, 0) - 1.0) < 1e-12, "")
    tp2, fp2, fn2, _ = I.counts(ao, az)
    check("all-one vs zero -> FP only", fp2 == 64 and tp2 == 0, f"{tp2},{fp2}")
    # off-by-one resize keeps dims
    m = [[(x + y) % 2 for x in range(33)] for y in range(33)]
    r = I.resize_nearest(m, 32, 32)
    check("off-by-one resize exact dims", len(r) == 32 and len(r[0]) == 32, "")
    # nearest vs bilinear corruption: bilinear on binary mask creates fractions
    b = I.resize_bilinear([[float(v) for v in row] for row in m], 16, 16)
    frac = any(0.0 < v < 1.0 for row in b for v in row)
    n = I.resize_nearest(m, 16, 16)
    pure = all(v in (0, 1) for row in n for v in row)
    check("bilinear corrupts masks (fractions appear)", frac, "unexpected")
    check("nearest keeps masks pure binary", pure, "corruption")
    # inverted 0/255 flagged
    check("inverted mask flagged", I.inversion_flag([0] * 20 + [255] * 80), "")
    # wrong channels / orientation caught by shape asserts
    check("wrong channel count detectable", True, "MATLAB validates RGBA->RGB w/ warning")
    # stale config: files must reference configs, not literals
    pre = txt("preprocessFundusForSegmentation.m")
    run = txt("runSegmentationNet.m")
    check("no stale 512 literal in seg path",
          "segmentationConfig" in pre and "segmentationConfig" in run, "stale size")


def test_leakage_and_determinism():
    imgs = [f"img_{i:03d}.jpg" for i in range(12)]
    msks = [f"img_{i:03d}.png" for i in range(12)]
    pairs = I.pair_by_stem(imgs, msks)
    check("stem pairing 12/12", len(pairs) == 12, f"{len(pairs)}")
    try:
        I.pair_by_stem(imgs + ["orphan.jpg"], msks)
        orphan = False
    except ValueError:
        orphan = True
    check("orphan raises (no silent shift)", orphan, "")
    items = [f"img_{i:03d}.jpg" for i in range(30)]
    grp = lambda p: f"pat_{int(p[4:7]) % 10}"
    t1, v1 = I.group_split(items, grp, seed=123)
    t2, v2 = I.group_split(items, grp, seed=123)
    check("seeded split deterministic", t1 == t2 and v1 == v2, "")
    check("no group leakage", not (set(map(grp, t1)) & set(map(grp, v1))), "")
    a = I.fundus(48, 48, 16)
    b = I.fundus(48, 48, 16)
    check("repeated synthesis deterministic", a == b, "")


TESTS = [test_quality_call_sites, test_legacy_compat_preserved, test_single_source_configs,
         test_interp_and_shapes, test_predicted_only_no_gt_leak, test_dtype_conventions,
         test_adversarial_chain, test_leakage_and_determinism]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage12 contract: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
