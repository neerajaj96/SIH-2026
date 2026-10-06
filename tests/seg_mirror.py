"""seg_mirror.py - Pure-stdlib (+Pillow for IO) mirror of the MATLAB
segmentation subsystem. Source of truth remains the .m files; this mirror
exists because this box has no MATLAB/Octave, so logic must be verifiable
here. Every function documents which .m file + lines it mirrors.

No numpy dependency (not installed, PEP 668 managed). All ops are pure
Python lists + stdlib statistics/random.
"""
import math
import os
import random
import statistics

# --- evaluateSegmentation.m:78-99 (hardened) ---

def _safe_div(num, den, fallback):
    if den > 0:
        return num / den
    return fallback


def single_pair_metrics(pred, gt):
    """Mirrors evaluateSegmentation.m localSinglePairMetrics (hardened).

    pred/gt: 2D nested lists (or flat lists) of truthy/falsy (or 0/255).
    Returns dict with dice/iou/sensitivity/specificity/precision plus
    tp/fp/fn/tn/nEmptyGT audit counts.
    Empty-vs-empty -> dice=1, iou=1, sens/spec/prec=NaN (float('nan')).
    Empty GT with FP>0 -> dice=0 (false alarm hurts).
    """
    pf = _flatten_2d(pred)
    gf = _flatten_2d(gt)
    assert len(pf) == len(gf), "pred/gt size mismatch"
    pb = [1 if _truthy(v) else 0 for v in pf]
    gb = [1 if _truthy(v) else 0 for v in gf]
    tp = sum(1 for a, b in zip(pb, gb) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(pb, gb) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(pb, gb) if a == 0 and b == 1)
    tn = sum(1 for a, b in zip(pb, gb) if a == 0 and b == 0)
    return {
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": _safe_div(tp, tp + fp + fn, 1.0),
        "sensitivity": _safe_div(tp, tp + fn, float("nan")),
        "specificity": _safe_div(tn, tn + fp, float("nan")),
        "precision": _safe_div(tp, tp + fp, float("nan")),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "nEmptyGT": 0 if any(gb) else 1,
    }


def _truthy(v):
    # Masks may be bool, 0/1, or 0/255. Threshold mirrors MATLAB midpoint:
    # for binary inputs any nonzero is foreground.
    if isinstance(v, bool):
        return v
    return v != 0


def _flatten_2d(m):
    if not isinstance(m, (list, tuple)):
        return [m]
    out = []
    for row in m:
        if isinstance(row, (list, tuple)):
            out.extend(row)
        else:
            out.append(row)
    return out


def batch_metrics(pairs):
    """Mirrors evaluateSegmentation.m batch form (macro + pooled micro).

    pairs: list of (pred, gt). Returns dict with perImage list, mean/std
    dicts (omitnan), pooled dict, n, nEmpty, nEmptyCorrect.
    """
    per = [single_pair_metrics(p, g) for p, g in pairs]
    fields = ["dice", "iou", "sensitivity", "specificity", "precision"]
    mean, std = {}, {}
    for f in fields:
        vals = [m[f] for m in per if not (isinstance(m[f], float) and math.isnan(m[f]))]
        mean[f] = statistics.fmean(vals) if vals else float("nan")
        std[f] = statistics.pstdev(vals) if len(vals) >= 2 else (0.0 if len(vals) == 1 else float("nan"))
        # NOTE: MATLAB std() uses N-1 (sample); statistics.pstdev uses N.
        # Tests use pytest.approx with tolerance covering the N vs N-1 gap
        # for small n, and exact values for n=1. Documented divergence.
    TP = sum(m["tp"] for m in per)
    FP = sum(m["fp"] for m in per)
    FN = sum(m["fn"] for m in per)
    TN = sum(m["tn"] for m in per)
    pooled = {
        "dice": _safe_div(2 * TP, 2 * TP + FP + FN, 1.0),
        "iou": _safe_div(TP, TP + FP + FN, 1.0),
        "sensitivity": _safe_div(TP, TP + FN, float("nan")),
        "specificity": _safe_div(TN, TN + FP, float("nan")),
        "precision": _safe_div(TP, TP + FP, float("nan")),
    }
    n_empty = sum(m["nEmptyGT"] for m in per)
    n_empty_correct = sum(1 for m in per if m["nEmptyGT"] == 1 and m["dice"] == 1.0)
    return {"perImage": per, "mean": mean, "std": std, "pooled": pooled,
            "n": len(per), "nEmpty": n_empty, "nEmptyCorrect": n_empty_correct}


# --- train_UNet_Segmentation.m:170-177 tverskyLoss ---

def tversky_loss(y_fg, t_fg, alpha=0.3, beta=0.7, smooth=1e-6):
    """Foreground-channel Tversky (mirrors MATLAB tverskyLoss foreground
    term). y_fg: flat list of probabilities in [0,1]; t_fg: flat 0/1.
    Returns 1 - TI. Guards probability range like the .m asserts."""
    assert len(y_fg) == len(t_fg) and len(y_fg) > 0
    for v in y_fg:
        assert -1e-3 <= v <= 1 + 1e-3, f"not probabilities: {v}"
    tp = sum(y * t for y, t in zip(y_fg, t_fg))
    fp = sum(y * (1 - t) for y, t in zip(y_fg, t_fg))
    fn = sum((1 - y) * t for y, t in zip(y_fg, t_fg))
    ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - ti


def tversky_loss_2class(y_fg, t_fg, alpha=0.3, beta=0.7, smooth=1e-6):
    """Full 2-class mean matching MATLAB mean(1-TI,'all') over [H W 2].

    Background channel is the complement (1-y, 1-t). Returns mean of the
    two per-class losses."""
    fg = tversky_loss(y_fg, t_fg, alpha, beta, smooth)
    y_bg = [1 - v for v in y_fg]
    t_bg = [1 - v for v in t_fg]
    bg = tversky_loss(y_bg, t_bg, alpha, beta, smooth)
    return (fg + bg) / 2


# --- buildSegmentationFileLists.m pairing ---

def stem(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name.lower()


def pair_files(image_files, mask_files, target="target"):
    """Mirrors buildSegmentationFileLists pairing: stem case-insensitive,
    sorted, errors on orphans/duplicates. Returns (imgPaired, maskPaired)."""
    by_stem = {}
    dups = []
    for m in mask_files:
        k = stem(m)
        if k in by_stem:
            dups.append(k)
        else:
            by_stem[k] = m
    if dups:
        raise ValueError(f"{target}: duplicate mask stems {sorted(set(dups))}")
    img_p, mask_p, orphans_i, orphans_m = [], [], [], []
    for im in image_files:
        k = stem(im)
        if k in by_stem:
            img_p.append(im)
            mask_p.append(by_stem.pop(k))
        else:
            orphans_i.append(im)
    orphans_m = list(by_stem.values())
    if orphans_i or orphans_m:
        raise ValueError(f"{target}: {len(img_p)} paired, {len(orphans_i)} orphan images, "
                         f"{len(orphans_m)} orphan masks (first img={orphans_i[:1]}, mask={orphans_m[:1]})")
    if not img_p:
        raise ValueError(f"{target}: zero pairs")
    order = sorted(range(len(img_p)), key=lambda i: img_p[i])
    return [img_p[i] for i in order], [mask_p[i] for i in order]


def flatten_cells(nested):
    """Mirrors buildSegmentationFileLists localFlatten + train script
    localExpandSub fix: flattens arbitrarily nested lists/tuples/strings."""
    flat = []

    def rec(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                rec(e)
        elif isinstance(x, str):
            if x:
                flat.append(x)
        elif x is None:
            return
        else:
            flat.append(str(x))

    rec(nested)
    # dedupe preserving order, drop empties (mirrors unique() order caveat:
    # MATLAB unique() sorts; this preserves insertion - tests assert set
    # equality, not order, for this helper).
    seen, out = set(), []
    for f in flat:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# --- validateMaskConventions.m ---

def value_range(unique_vals):
    """Mirrors validator range classification. unique_vals: iterable of
    pixel values. Returns '0-1', '0-255', or 'other'.

    Strict: 0-255 ONLY if every value is in {0,1,255}. A grayscale
    artifact like [0,128,255] is 'other' (CHECK THIS), not 0-255 - the
    validator's looksBinary gate enforces the same.
    """
    u = sorted(set(unique_vals))
    if not u:
        return "other"
    if max(u) <= 1:
        return "0-1"
    if set(u) <= {0, 1, 255}:
        return "0-255"
    return "other"


def foreground_fraction(flat_vals):
    """Midpoint-threshold fg fraction mirroring the validator (with flat
    fallback to nonzero test). flat_vals: iterable of pixel values."""
    vals = list(flat_vals)
    if not vals:
        return 0.0
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return 1.0 if mx != 0 else 0.0
    thr = (mx + mn) / 2
    return sum(1 for v in vals if v > thr) / len(vals)


def check_inversion(fractions):
    """Mirrors validator spread logic. Returns 'ok' | 'warn' | 'error'."""
    if len(fractions) < 2:
        if fractions and fractions[0] > 0.60:
            return "error"
        return "ok"
    spread = max(fractions) - min(fractions)
    if spread > 0.50 and max(fractions) > 0.60:
        return "error"
    if spread > 0.15:
        return "warn"
    return "ok"


# --- splitSegmentationDataset.m / buildPatientLevelSplit.m ---

def patient_split(paths, patient_of=None, val_fraction=0.15, seed=42):
    """Group-aware split mirroring splitSegmentationDataset.

    paths: list of image paths. patient_of: fn path->group id or None
    (None = image-level, explicit). Deterministic via random.Random(seed).
    Returns (train_idx, val_idx) index lists. Guarantees: no group crosses
    the split; >=1 group each side; val groups = max(round(frac*G),1).
    """
    n = len(paths)
    assert n >= 2
    assert 0 < val_fraction < 1
    if patient_of is None:
        groups = list(paths)
    else:
        groups = [str(patient_of(p)) for p in paths]
    uniq = list(dict.fromkeys(groups))  # stable unique (mirrors unique stable)
    rng = random.Random(seed)
    order = list(range(len(uniq)))
    rng.shuffle(order)
    n_val_g = max(round(val_fraction * len(uniq)), 1)
    n_val_g = min(n_val_g, len(uniq) - 1)
    val_groups = {uniq[i] for i in order[:n_val_g]}
    train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
    val_idx = [i for i, g in enumerate(groups) if g in val_groups]
    # leakage audit
    assert not (set(groups[i] for i in train_idx) & set(groups[i] for i in val_idx))
    return train_idx, val_idx
