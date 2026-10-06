"""Adversarial tests for the segmentation subsystem mirror.

Runs with: python3 -m pytest tests/ -v (no MATLAB, no numpy, no data).
Each test names the .m file + behavior it pins. Synthetic only - no
medical performance numbers claimed.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import seg_mirror as M


def _square(n=10, r0=2, r1=6, c0=2, c1=6):
    # Defaults mirror MATLAB gt(3:6,3:6) 1-indexed inclusive = 4x4 at
    # 0-indexed rows/cols 2..5 (range(2,6)). Do not "fix" to 3..6.
    m = [[0] * n for _ in range(n)]
    for r in range(r0, r1):
        for c in range(c0, c1):
            m[r][c] = 1
    return m


def test_hand_computed_10x10_matches_matlab_reference():
    # Mirrors runSelfTests.m localTestSegmentationMetrics + evaluateSegmentation.m:11-17.
    # GT 4x4 at MATLAB (3:6,3:6), pred 4x4 at (4:7,4:7) shifted by 1px -> 9px overlap.
    gt = _square()
    pred = _square(r0=3, r1=7, c0=3, c1=7)
    m = M.single_pair_metrics(pred, gt)
    assert abs(m["dice"] - 0.5625) < 1e-9
    assert abs(m["iou"] - 0.391304347826087) < 1e-9
    assert abs(m["sensitivity"] - 0.5625) < 1e-9
    assert abs(m["specificity"] - 0.9166666666666666) < 1e-9
    assert abs(m["precision"] - 0.5625) < 1e-9
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (9, 7, 7, 77)


def test_empty_empty_is_perfect_but_nan_sens():
    # evaluateSegmentation.m empty policy: dice/iou=1, sens/spec/prec=NaN.
    m = M.single_pair_metrics([[0, 0], [0, 0]], [[0, 0], [0, 0]])
    assert m["dice"] == 1.0 and m["iou"] == 1.0
    assert math.isnan(m["sensitivity"]) and math.isnan(m["precision"])
    assert m["nEmptyGT"] == 1


def test_single_pixel_false_alarm_hurts():
    # Empty GT + 1 FP pixel must score dice=0, not NaN/excluded.
    pred = [[0] * 10 for _ in range(10)]
    pred[0][0] = 1
    gt = [[0] * 10 for _ in range(10)]
    m = M.single_pair_metrics(pred, gt)
    assert m["dice"] == 0.0
    assert m["fp"] == 1 and m["tp"] == 0


def test_batch_macro_vs_pooled_divergence_flagged():
    # 1 empty-correct (dice 1) + 1 half-overlap image: macro mean is pulled
    # up by the empty image while pooled reflects pixel reality. Both must
    # be reported (evaluateSegmentationDataset relies on this).
    empty_pred = [[0] * 8 for _ in range(8)]
    empty_gt = [[0] * 8 for _ in range(8)]
    gt2 = _square(n=8, r0=2, r1=6, c0=2, c1=6)  # 16px
    pred2 = _square(n=8, r0=2, r1=6, c0=4, c1=8)  # shifted -> 8px overlap
    b = M.batch_metrics([(empty_pred, empty_gt), (pred2, gt2)])
    assert b["n"] == 2 and b["nEmpty"] == 1 and b["nEmptyCorrect"] == 1
    assert b["mean"]["dice"] > b["pooled"]["dice"]  # macro inflated by empty
    assert abs(b["mean"]["dice"] - (1.0 + 0.5) / 2) < 1e-9


def test_tversky_perfect_is_zero_total_miss_near_one():
    y_ok = [0.99, 0.01, 0.99, 0.01]
    t = [1, 0, 1, 0]
    assert M.tversky_loss(y_ok, t) < 0.05
    y_bad = [0.01, 0.99, 0.01, 0.99]
    assert M.tversky_loss(y_bad, t) > 0.9


def test_tversky_beta_penalizes_misses_more_than_alpha():
    # Same FN-heavy prediction: higher beta must give higher loss.
    y = [0.2, 0.2, 0.9, 0.9]  # misses on first two positives
    t = [1, 1, 0, 0]
    lo = M.tversky_loss(y, t, alpha=0.3, beta=0.3)
    hi = M.tversky_loss(y, t, alpha=0.3, beta=0.9)
    assert hi > lo


def test_tversky_guards_non_probabilities():
    import pytest
    with pytest.raises(AssertionError):
        M.tversky_loss([1.5, -0.2], [1, 0])


def test_pairing_stem_case_insensitive_sorted():
    imgs = ["d/STARE_01.JPG", "d/stare_02.jpg"]
    masks = ["m/stare_02.PNG", "m/STARE_01.png"]
    ip, mp = M.pair_files(imgs, masks)
    assert ip == sorted(imgs)
    assert M.stem(ip[0]) == M.stem(mp[0])


def test_pairing_orphan_errors():
    import pytest
    with pytest.raises(ValueError, match="orphan"):
        M.pair_files(["a/001.jpg", "a/002.jpg"], ["m/001.png"])


def test_pairing_duplicate_errors():
    import pytest
    with pytest.raises(ValueError, match="duplicate"):
        M.pair_files(["a/001.jpg"], ["m/001.png", "m2/001.png"])


def test_flatten_nested_cells():
    nested = [["a", ["b", ("c",)]], "d", None, ""]
    assert sorted(M.flatten_cells(nested)) == ["a", "b", "c", "d"]
    # localExpandSub pattern: {fullfile(cell,...)} double-wrap
    assert sorted(M.flatten_cells([["x/images", "y/images"]])) == ["x/images", "y/images"]


def test_value_range_detection():
    assert M.value_range([0, 255, 0, 255]) == "0-255"
    assert M.value_range([0, 1, 0, 1]) == "0-1"
    assert M.value_range([0, 128, 255]) == "other"
    assert M.value_range([0, 0, 0]) == "0-1"  # blank mask max<=1


def test_foreground_fraction_midpoint():
    assert abs(M.foreground_fraction([0] * 97 + [255] * 3) - 0.03) < 1e-9
    assert M.foreground_fraction([0, 0, 0]) == 0.0
    assert M.foreground_fraction([5, 5, 5]) == 1.0  # flat nonzero


def test_inversion_detection():
    assert M.check_inversion([0.02, 0.978]) == "error"  # classic inverted pair
    assert M.check_inversion([0.05, 0.25]) == "warn"
    assert M.check_inversion([0.05, 0.08]) == "ok"
    assert M.check_inversion([0.75]) == "error"  # single inverted source


def test_split_deterministic_no_leakage():
    paths = [f"img_{i:03d}.jpg" for i in range(60)]
    pat = lambda p: f"patient_{int(p[4:7]) % 20:02d}"  # 20 patients x3 (mirrors runSelfTests)
    tr1, va1 = M.patient_split(paths, pat, 0.2, 123)
    tr2, va2 = M.patient_split(paths, pat, 0.2, 123)
    assert tr1 == tr2 and va1 == va2  # seeded determinism
    g_tr = {pat(paths[i]) for i in tr1}
    g_va = {pat(paths[i]) for i in va1}
    assert not (g_tr & g_va)  # no patient crosses
    assert len(g_tr) + len(g_va) == 20


def test_split_counts_and_single_group_floor():
    paths = [f"i{i}.jpg" for i in range(10)]
    tr, va = M.patient_split(paths, None, 0.15, 42)
    assert len(tr) + len(va) == 10 and len(va) >= 1 and len(tr) >= 1


def test_config_parity_single_source_of_truth():
    # Pins the Stage-2 contract: 512 default, 0/255 IDs, all three .m files
    # reference segmentationConfig (no silent literal drift).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = open(os.path.join(root, "segmentationConfig.m")).read()
    assert "[512 512]" in cfg and "[0 255]" in cfg
    for f in ["train_UNet_Segmentation.m", "preprocessFundusForSegmentation.m", "runSegmentationNet.m"]:
        body = open(os.path.join(root, f)).read()
        assert "segmentationConfig" in body, f"{f} must read segmentationConfig"
    train = open(os.path.join(root, "train_UNet_Segmentation.m")).read()
    assert "labelIDs = [0, 1]" not in train  # old bug must stay fixed
    assert "splitEachLabel_manual" not in train  # old splitter must stay removed
    assert "buildSegmentationFileLists" in train and "splitSegmentationDataset" in train
    pipe = open(os.path.join(root, "runScreeningPipeline.m")).read()
    assert "'nearest'" in pipe  # fusion mask interp fix must stay
