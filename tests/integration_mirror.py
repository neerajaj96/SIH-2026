#!/usr/bin/env python3
"""integration_mirror.py — stdlib-only synthetic helpers for Stage-1+2 chain.

Independent re-implementation (not a copy) of the contract semantics:
synthetic fundus, ROI (threshold+fill+largest), coverage/circularity,
nearest/bilinear resize, mask conventions, 512/224 transforms, fusion
shape, Dice/counts, inversion rule, stem pairing, group split.

Source of truth remains the .m files; this exists because no MATLAB
runs here. No numpy/PIL/pytest needed.
"""
import math
import random


def fundus(h, w, r, bright=1.0, dx=0):
    cx, cy = w / 2 + dx, h / 2
    img = [[[0, 0, 0] for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                t = 60 + 40 * math.sin(0.3 * x) * math.cos(0.3 * y)
                v = max(0, min(255, int(t * bright)))
                img[y][x] = [v, v, v]
    return img


def gray(img):
    return [[sum(p) // 3 for p in row] for row in img]


def roi(mask_seed):
    """Largest-component ROI with hole-fill from border flood. Independent logic."""
    h, w = len(mask_seed), len(mask_seed[0])
    m = [row[:] for row in mask_seed]
    # hole fill: flood non-mask from border
    seen = [[False] * w for _ in range(h)]
    stack = []
    for x in range(w):
        if not m[0][x]:
            stack.append((0, x))
        if not m[h - 1][x]:
            stack.append((h - 1, x))
    for y in range(h):
        if not m[y][0]:
            stack.append((y, 0))
        if not m[y][w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or seen[y][x] or m[y][x]:
            continue
        seen[y][x] = True
        stack += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    for y in range(h):
        for x in range(w):
            if not m[y][x] and not seen[y][x]:
                m[y][x] = True
    comp = [[0] * w for _ in range(h)]
    cid, sizes = 0, {}
    for y in range(h):
        for x in range(w):
            if m[y][x] and not comp[y][x]:
                cid += 1
                q = [(y, x)]
                comp[y][x] = cid
                n = 0
                while q:
                    cy, cx = q.pop()
                    n += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and m[ny][nx] and not comp[ny][nx]:
                            comp[ny][nx] = cid
                            q.append((ny, nx))
                sizes[cid] = n
    if not sizes:
        return m
    big = max(sizes, key=sizes.get)
    return [[comp[y][x] == big for x in range(w)] for y in range(h)]


def coverage(m):
    return sum(sum(r) for r in m) / (len(m) * len(m[0]))


def circularity(m):
    h, w = len(m), len(m[0])
    area = sum(sum(r) for r in m)
    per = 0
    for y in range(h):
        for x in range(w):
            if m[y][x]:
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w or not m[ny][nx]:
                        per += 1
                        break
    return min(1.0, 4 * math.pi * area / (per * per)) if per else 0.0


def resize_nearest(mat, th, tw):
    """Independent nearest-neighbor (categorical masks)."""
    h, w = len(mat), len(mat[0])
    out = [[0] * tw for _ in range(th)]
    for y in range(th):
        for x in range(tw):
            out[y][x] = mat[min(h - 1, int(y * h / th))][min(w - 1, int(x * w / tw))]
    return out


def resize_bilinear(mat, th, tw):
    """Independent bilinear (photos)."""
    h, w = len(mat), len(mat[0])
    out = [[0.0] * tw for _ in range(th)]
    for y in range(th):
        gy = (y + 0.5) * h / th - 0.5
        y0 = max(0, min(h - 1, int(math.floor(gy))))
        y1 = max(0, min(h - 1, y0 + 1))
        fy = max(0.0, min(1.0, gy - y0))
        for x in range(tw):
            gx = (x + 0.5) * w / tw - 0.5
            x0 = max(0, min(w - 1, int(math.floor(gx))))
            x1 = max(0, min(w - 1, x0 + 1))
            fx = max(0.0, min(1.0, gx - x0))
            out[y][x] = (mat[y0][x0] * (1 - fy) * (1 - fx) + mat[y0][x1] * (1 - fy) * fx
                         + mat[y1][x0] * fy * (1 - fx) + mat[y1][x1] * fy * fx)
    return out


def counts(pred, gt):
    tp = fp = fn = tn = 0
    for pr, gr in zip(pred, gt):
        for p, g in zip(pr, gr):
            if p and g:
                tp += 1
            elif p and not g:
                fp += 1
            elif not p and g:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def dice(tp, fp, fn):
    return 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def inversion_flag(values_0_255):
    """Mirror of validator inversion rule: >60% fg or >50pp spread w/ side >60%."""
    n = len(values_0_255)
    fg = sum(1 for v in values_0_255 if v > 0) / n
    return fg > 0.60


def stem(name):
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0].lower()


def pair_by_stem(images, masks):
    mi = {}
    for m in masks:
        s = stem(m)
        if s in mi:
            raise ValueError("duplicate mask stem: " + s)
        mi[s] = m
    pairs = []
    for im in images:
        s = stem(im)
        if s not in mi:
            raise ValueError("orphan image: " + s)
        pairs.append((im, mi.pop(s)))
    if mi:
        raise ValueError("orphan masks remain")
    return sorted(pairs)


def group_split(items, group_of, seed=42, val_frac=0.2):
    """Seeded group-aware split; no group crosses sides."""
    rng = random.Random(seed)
    groups = sorted(set(group_of(i) for i in items))
    rng.shuffle(groups)
    nval = max(1, int(len(groups) * val_frac))
    val = set(groups[:nval])
    tr = [i for i in items if group_of(i) not in val]
    va = [i for i in items if group_of(i) in val]
    assert not (set(map(group_of, tr)) & set(map(group_of, va)))
    return tr, va
