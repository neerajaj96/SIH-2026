"""grading_mirror.py - Pure-stdlib mirror of Stage-3 DR grading logic.

Source of truth remains the .m files (gradingConfig.m,
ordinalGradingLoss.m, buildGradingFusionTensor.m, evaluateGrading.m,
computeQWK.m, wilsonScoreInterval.m, buildGradingDatasets.m,
train_DR_Grader.m). This mirror exists because the box has no
MATLAB/Octave and no clinical data - logic verifiable here is verified
here; everything else is marked UNVERIFIED. No numpy (not installed).

Deliberately does NOT build a fake DenseNet to "prove" the model works.
"""
import math

N_CLASSES = 5
CLASS_VALUES = [0, 1, 2, 3, 4]
CHANNEL_ORDER = ["R", "G", "B", "vessel", "lesion"]
INPUT_SIZE = (224, 224)
REFERABLE_THRESHOLD = 2


# --- grade mapping (train_DR_Grader.m categorical/onehot/argmax-1) ---

def grade_from_probs(probs):
    """argmax - 1 with 0-indexed ICDR grades. probs: length-5 list."""
    assert len(probs) == 5
    best = max(range(5), key=lambda c: probs[c])
    return best  # grades already 0..4, so argmax index IS the grade


def is_referable(grade, threshold=REFERABLE_THRESHOLD):
    return grade >= threshold


def one_hot(grade):
    assert 0 <= grade <= 4 and grade == int(grade)
    return [1 if c == grade else 0 for c in range(5)]


# --- ordinal loss (ordinalGradingLoss.m) ---

def softmax(logits):
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy(probs_batch, targets_batch):
    """Mean -log p_true over batch. Batches are lists of length-5 lists."""
    eps = 1e-12
    total = 0.0
    for p, t in zip(probs_batch, targets_batch):
        true_c = t.index(1)
        total += -math.log(max(p[true_c], eps))
    return total / len(probs_batch)


def expected_grade(probs):
    return sum(p * c for p, c in zip(probs, CLASS_VALUES))


def ordinal_penalty(probs_batch, targets_batch):
    se = 0.0
    for p, t in zip(probs_batch, targets_batch):
        true_c = t.index(1)
        se += (expected_grade(p) - true_c) ** 2
    return se / len(probs_batch)


def hybrid_loss(probs_batch, targets_batch, lambda_=0.5, class_weights=None):
    ce = cross_entropy(probs_batch, targets_batch)
    pen = ordinal_penalty(probs_batch, targets_batch)
    if class_weights is not None:
        assert len(class_weights) == 5 and all(w > 0 for w in class_weights)
        eps = 1e-12
        total = 0.0
        for p, t in zip(probs_batch, targets_batch):
            true_c = t.index(1)
            total += class_weights[true_c] * -math.log(max(p[true_c], eps))
        ce = total / len(probs_batch)
    return ce + lambda_ * pen, ce, pen


# --- QWK (computeQWK.m) ---

def qwk(y_true, y_pred, num_classes=5):
    assert len(y_true) == len(y_pred) and len(y_true) > 0
    O = [[0] * num_classes for _ in range(num_classes)]
    for a, b in zip(y_true, y_pred):
        assert 0 <= a < num_classes and 0 <= b < num_classes
        O[a][b] += 1
    W = [[(i - j) ** 2 / (num_classes - 1) ** 2 for j in range(num_classes)]
         for i in range(num_classes)]
    n = len(y_true)
    row = [sum(r) for r in O]
    col = [sum(O[i][j] for i in range(num_classes)) for j in range(num_classes)]
    num = sum(W[i][j] * O[i][j] for i in range(num_classes) for j in range(num_classes))
    den = sum(W[i][j] * row[i] * col[j] / n for i in range(num_classes) for j in range(num_classes))
    if den == 0:
        return 1.0 if num == 0 else float("nan")
    return 1 - num / den


# --- confusion + per-class (evaluateGrading.m) ---

def confusion(y_true, y_pred, num_classes=5):
    C = [[0] * num_classes for _ in range(num_classes)]
    for a, b in zip(y_true, y_pred):
        C[a][b] += 1
    return C


def _safe_div(num, den, fallback=float("nan")):
    return num / den if den > 0 else fallback


def per_class_metrics(C):
    n = sum(sum(r) for r in C)
    out = []
    for c in range(len(C)):
        tp = C[c][c]
        fn = sum(C[c]) - tp
        fp = sum(C[i][c] for i in range(len(C))) - tp
        tn = n - tp - fn - fp
        rec = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        prec = _safe_div(tp, tp + fp)
        if math.isnan(rec) or math.isnan(prec) or (rec + prec) == 0:
            f1 = float("nan")
            if (tp + fn) > 0 and (tp + fp) == 0:
                f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        out.append({"grade": c, "recall": rec, "specificity": spec,
                    "precision": prec, "f1": f1, "support": tp + fn})
    return out


def macro_f1(per):
    vals = [m["f1"] for m in per if not math.isnan(m["f1"])]
    return sum(vals) / len(vals) if vals else float("nan")


def weighted_f1(per):
    num, den = 0.0, 0
    for m in per:
        if not math.isnan(m["f1"]):
            num += m["f1"] * m["support"]
            den += m["support"]
    return num / den if den > 0 else float("nan")


def referable_counts(y_true, y_pred, threshold=REFERABLE_THRESHOLD):
    tr = [1 if y >= threshold else 0 for y in y_true]
    pr = [1 if y >= threshold else 0 for y in y_pred]
    tp = sum(1 for a, b in zip(tr, pr) if a == 1 and b == 1)
    fn = sum(1 for a, b in zip(tr, pr) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(tr, pr) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(tr, pr) if a == 0 and b == 1)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def error_distance(y_true, y_pred):
    errs = [abs(int(b) - int(a)) for a, b in zip(y_true, y_pred)]
    n_err = sum(1 for e in errs if e > 0)
    hist = [sum(1 for e in errs if e == d) for d in range(5)]
    return {"meanAbsErr": sum(errs) / len(errs),
            "distHist": hist,
            "adjacentRate": (hist[1] / n_err) if n_err else float("nan"),
            "severeRate": (sum(hist[2:]) / n_err) if n_err else float("nan")}


# --- Wilson CI (wilsonScoreInterval.m) ---

def _norminv(p):
    """Acklam rational approximation of the standard normal quantile."""
    assert 0 < p < 1
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def wilson(successes, n, alpha=0.05):
    assert n > 0 and 0 <= successes <= n
    z = _norminv(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(center - half, 0.0), min(center + half, 1.0))


# --- ROC-AUC (evaluateGrading.m localROCAUC, Mann-Whitney + tied ranks) ---

def roc_auc(scores, labels):
    n1 = sum(1 for v in labels if v == 1)
    n0 = sum(1 for v in labels if v == 0)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    s = [scores[k] for k in order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1  # 1-based average rank
        i = j + 1
    r = [0.0] * len(scores)
    for pos, idx in enumerate(order):
        r[idx] = ranks[pos]
    return (sum(r[i] for i in range(len(scores)) if labels[i] == 1) - n1 * (n1 + 1) / 2) / (n1 * n0)


# --- synthetic 5-channel tensor (buildGradingFusionTensor.m contract) ---

def synthetic_fusion(h=8, w=8, vessel_frac=0.1, lesion_frac=0.05, seed=0):
    """Deterministic synthetic fused tensor as nested lists [h][w][5].

    ch1-3 photo ramp 0-255, ch4 vessel {0,255}, ch5 lesion {0,255} with
    lesion = mahe OR exudate (merged, Stage-2 contract). Range checks mirror
    the builder's loud validation."""
    import random
    rng = random.Random(seed)
    photo = [[[int(255 * (r / max(h - 1, 1))) for _ in range(3)] for _ in range(w)] for r in range(h)]
    mahe = [[1 if rng.random() < lesion_frac / 2 else 0 for _ in range(w)] for _ in range(h)]
    exud = [[1 if rng.random() < lesion_frac / 2 else 0 for _ in range(w)] for _ in range(h)]
    vessel = [[1 if rng.random() < vessel_frac else 0 for _ in range(w)] for _ in range(h)]
    fused = []
    for r in range(h):
        row = []
        for c in range(w):
            lesion = 1 if (mahe[r][c] or exud[r][c]) else 0
            row.append([photo[r][c][0], photo[r][c][1], photo[r][c][2],
                        255 * vessel[r][c], 255 * lesion])
        fused.append(row)
    return fused


def check_fusion_tensor(fused):
    h, w = len(fused), len(fused[0])
    assert all(len(row) == w and all(len(px) == 5 for px in row) for row in fused)
    for row in fused:
        for px in row:
            assert all(0 <= px[k] <= 255 for k in range(5)), "channels must be 0-255"
            assert px[3] in (0, 255) and px[4] in (0, 255), "mask channels must be {0,255}"
    return (h, w, 5)
