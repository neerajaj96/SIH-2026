"""Adversarial Python tests for Stage-3 grading logic (no MATLAB/data).

Runs with: python3 -m pytest tests/test_grading_mirror.py -v
All synthetic - no clinical performance claimed. Each test names the .m
contract it pins. MATLAB/data execution stays UNVERIFIED.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import grading_mirror as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


# --- grade mapping (gradingConfig.m, train_DR_Grader.m) ---

def test_argmax_minus_one_and_referable_threshold():
    assert G.grade_from_probs([0.05, 0.05, 0.7, 0.1, 0.1]) == 2
    assert G.grade_from_probs([0.9, 0.02, 0.02, 0.03, 0.03]) == 0
    assert G.grade_from_probs([0.1, 0.1, 0.1, 0.1, 0.6]) == 4
    assert G.is_referable(1) is False
    assert G.is_referable(2) is True
    assert G.is_referable(4) is True
    assert G.one_hot(3) == [0, 0, 0, 1, 0]


def test_config_freezes_labels_and_channels():
    cfg = _read("gradingConfig.m")
    assert "channelOrder" in cfg
    for ch in ["'R'", "'G'", "'B'", "'vessel'", "'lesion'"]:
        assert ch in cfg, f"channel {ch} missing from gradingConfig"
    assert "[224 224]" in cfg
    assert "single(0:4)" in cfg
    assert "referableThreshold" in cfg
    for f in ["train_DR_Grader.m", "buildGradingFusionTensor.m", "ordinalGradingLoss.m", "evaluateGrading.m"]:
        assert "gradingConfig" in _read(f), f"{f} must read gradingConfig"


def test_fusion_matches_inference_interp_and_order():
    # Both training builder and inference path: photos bilinear, masks nearest.
    for f in ["buildGradingFusionTensor.m", "train_DR_Grader.m", "runScreeningPipeline.m"]:
        body = _read(f)
        assert "bilinear" in body and "nearest" in body, f"{f} must pin interp semantics"
    b = _read("buildGradingFusionTensor.m")
    assert "maheMask | exudateMask" in b or "maheMask|exudateMask" in b.replace(" ", "")
    assert "ground-truth" in b.lower() or "ground truth" in b.lower()


def test_no_gt_masks_in_fusion_training_path():
    t = _read("train_DR_Grader.m")
    assert "segmentationNotTrained" in t  # errors if U-Net .mats missing
    assert "runSegmentationNet" in t  # predicted masks, not GT
    assert "ordinalGradingLoss" in t  # audited loss, not inline duplicate
    assert "buildGradingFusionTensor" in t  # canonical builder


# --- ordinal loss (ordinalGradingLoss.m) ---

def test_ordinal_penalty_zero_when_confident_correct():
    p = [[0.02, 0.02, 0.9, 0.03, 0.03]]
    t = [[0, 0, 1, 0, 0]]
    loss, ce, pen = G.hybrid_loss(p, t)
    assert pen < 0.05
    assert abs(loss - (ce + 0.5 * pen)) < 1e-12


def test_extreme_error_penalized_more_than_adjacent():
    t = [[0, 0, 0, 0, 1]]  # true PDR
    p_adj = [[0.05, 0.05, 0.1, 0.7, 0.1]]  # mass on neighbor L3
    p_far = [[0.7, 0.1, 0.05, 0.05, 0.1]]  # same p_true, mass 4 away
    _, ce_a, pen_a = G.hybrid_loss(p_adj, t)
    _, ce_f, pen_f = G.hybrid_loss(p_far, t)
    assert abs(ce_a - ce_f) < 1e-9  # CE blind to distance ...
    assert pen_f > pen_a * 5  # ... penalty quadratic in distance
    assert G.hybrid_loss(p_far, t)[0] > G.hybrid_loss(p_adj, t)[0]


def test_lambda_scale_documented_and_bounded():
    p = [[0.7, 0.1, 0.05, 0.05, 0.1]]
    t = [[0, 0, 0, 0, 1]]
    _, ce, pen = G.hybrid_loss(p, t)
    # CE = -log p_true: ~0 confident-correct, 1.61 at uniform, unbounded
    # above when confident-wrong (here -log(.1)=2.30). Penalty in [0,16].
    assert abs(ce - 2.302585092994046) < 1e-9
    assert 0 <= pen <= 16
    lo, _, _ = G.hybrid_loss(p, t, lambda_=0.0)
    hi, _, _ = G.hybrid_loss(p, t, lambda_=2.0)
    assert hi > lo
    body = _read("ordinalGradingLoss.m")
    assert "classWeights" in body  # opt-in imbalance path present
    assert "notProbabilities" in body and "notNormalized" in body  # guards


def test_class_weights_upweight_rare_class():
    p = [[0.5, 0.2, 0.1, 0.1, 0.1]]
    t = [[0, 0, 0, 0, 1]]  # rare L4
    plain, _, _ = G.hybrid_loss(p, t)
    weighted, _, _ = G.hybrid_loss(p, t, class_weights=[1, 1, 1, 1, 5])
    assert weighted > plain


# --- QWK (computeQWK.m) ---

def test_qwk_matches_matlab_reference_value():
    # Reference from runSelfTests.m localTestQWK (sklearn quadratic kappa).
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 2, 1]
    y_pred = [0, 1, 1, 2, 2, 3, 3, 3, 4, 3, 2, 1]
    assert abs(G.qwk(y_true, y_pred) - 0.88940092) < 1e-6


def test_qwk_perfect_and_adversarial_orders():
    assert G.qwk([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == 1.0
    far = G.qwk([0, 0, 0, 4, 4, 4], [4, 4, 4, 0, 0, 0])  # maximal distance
    near = G.qwk([0, 0, 0, 4, 4, 4], [1, 1, 1, 3, 3, 3])  # adjacent
    assert near > far  # quadratic weights punish distance
    single = G.qwk([2, 2, 2], [2, 2, 2])
    assert single == 1.0


# --- confusion + per-class (evaluateGrading.m) ---

def test_confusion_layout_rows_true_cols_pred():
    C = G.confusion([0, 0, 1, 4], [0, 1, 1, 4])
    assert C[0][0] == 1 and C[0][1] == 1 and C[1][1] == 1 and C[4][4] == 1
    assert sum(sum(r) for r in C) == 4


def test_per_class_recall_specificity_and_zero_support_nan():
    C = G.confusion([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
    per = G.per_class_metrics(C)
    assert per[0]["support"] == 2 and per[3]["support"] == 0  # L3 absent
    assert math.isnan(per[3]["recall"])  # undefined, not 0
    assert per[2]["recall"] == 0.5
    assert 0 <= per[0]["specificity"] <= 1
    # Never-predicted-but-supported class gets F1 0, not NaN.
    C2 = G.confusion([1, 1, 0], [0, 0, 0])
    per2 = G.per_class_metrics(C2)
    assert per2[1]["f1"] == 0.0


def test_macro_vs_weighted_f1_divergence():
    # Rare severe class with poor F1: macro drops more than weighted.
    C = G.confusion([0] * 90 + [4] * 10, [0] * 90 + [0] * 10)
    per = G.per_class_metrics(C)
    assert G.macro_f1(per) < G.weighted_f1(per)


# --- referable + Wilson (wilsonScoreInterval.m) ---

def test_referable_mapping_and_counts():
    rc = G.referable_counts([0, 1, 2, 3, 4, 2], [0, 2, 2, 1, 4, 4])
    assert (rc["tp"], rc["fn"], rc["tn"], rc["fp"]) == (3, 1, 1, 1)


def test_wilson_matches_statsmodels_reference():
    # Reference bounds from runSelfTests.m (statsmodels wilson).
    cases = [(45, 50, (0.786398, 0.956524)), (93, 100, (0.862505, 0.965681)),
             (150, 175, (0.797616, 0.901327))]
    for x, n, (elo, ehi) in cases:
        lo, hi = G.wilson(x, n)
        assert abs(lo - elo) < 1e-4 and abs(hi - ehi) < 1e-4
    _, hi = G.wilson(20, 20)
    assert hi <= 1 and hi > 0.8
    lo, _ = G.wilson(0, 20)
    assert lo >= 0 and lo < 1e-9


def test_roc_auc_ordering_and_ties():
    assert abs(G.roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) < 1e-9
    assert G.roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert math.isnan(G.roc_auc([0.5, 0.6], [1, 1]))  # single-class undefined


def test_error_distance_adjacent_vs_severe():
    e = G.error_distance([0, 0, 0, 4], [0, 1, 3, 4])
    assert e["distHist"][0] == 2 and e["distHist"][1] == 1 and e["distHist"][3] == 1
    assert abs(e["adjacentRate"] - 0.5) < 1e-9
    assert abs(e["severeRate"] - 0.5) < 1e-9


# --- 5-channel tensor contract ---

def test_synthetic_fusion_shape_dtype_ranges():
    f = G.synthetic_fusion()
    assert G.check_fusion_tensor(f) == (8, 8, 5)
    assert G.CHANNEL_ORDER == ["R", "G", "B", "vessel", "lesion"]


def test_lesion_channel_is_mahe_or_exudate():
    import random
    rng = random.Random(3)
    mahe = [rng.random() < 0.1 for _ in range(200)]
    exud = [rng.random() < 0.1 for _ in range(200)]
    lesion = [1 if (a or b) else 0 for a, b in zip(mahe, exud)]
    assert all(v in (0, 1) for v in lesion)
    # Merged channel fires iff either source fires (never AND-only).
    assert any(l == 1 and a == 0 for l, a in zip(lesion, mahe))
    assert all((l == 1) == (a == 1 or b == 1) for l, a, b in zip(lesion, mahe, exud))


# --- leakage / provenance (buildGradingDatasets.m) ---

def test_dataset_gates_present_in_code():
    b = _read("buildGradingDatasets.m")
    assert "localAssertGrades" in b and "0..4" in b
    assert "localAssertDisjoint" in b  # train/val/test pairwise disjoint
    assert "duplicate" in b.lower()
    assert "localBuildManifest" in b  # provenance manifest
    assert "localWarnDDRSegOverlap" in b  # seg/grading ID overlap flagged
    assert "never tune" in b.lower()  # no-test-tuning rule in log
    assert "0.85, 0.15, 42" in b  # DDR split identity preserved
    assert "0.70, 0.15, 42" in b  # Messidor split identity preserved


def test_no_test_tuning_paths_in_training():
    t = _read("train_DR_Grader.m")
    assert "buildGradingDatasets" in t
    assert "ValidationData" in t  # tunes on val
    assert "testSet" not in t  # never touches test
    assert "calibrateTemperature" in t  # deferred to val-only step


def test_eval_is_held_out_only_and_uses_shared_qwk_wilson():
    e = _read("evaluateGrading.m")
    assert "HELD-OUT" in e.upper() or "held-out" in e.lower()
    assert "computeQWK" in e and "wilsonScoreInterval" in e
    assert "compareModels(" not in e  # no call into the paired-test entry point; standalone eval


def test_stage1_stage2_files_untouched():
    import subprocess
    out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout
    changed = {line[3:].strip() for line in out.splitlines() if line.strip()}
    forbidden = {"segmentationConfig.m", "preprocessFundusForSegmentation.m", "runSegmentationNet.m",
                 "train_UNet_Segmentation.m", "evaluateSegmentation.m", "evaluateSegmentationDataset.m",
                 "validateMaskConventions.m", "assessAndEnhanceImage.m", "calibrateQualityThresholds.m",
                 "tests/seg_mirror.py", "tests/test_seg_mirror.py"}
    touched = forbidden & changed
    assert not touched, f"Stage-1/2 files must stay untouched: {touched}"
