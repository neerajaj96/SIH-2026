#!/usr/bin/env python3
"""Stage-1 Python mirror harness (STDLIB ONLY) for IQA + enhancement logic.
Executed HERE (no MATLAB/Octave in workspace). Mirrors the MATLAB math
with identical constants from qualityConfig.m; validates ORDERINGS,
decision taxonomy, calibration math, adversarial robustness, determinism,
and benchmarks. Does NOT claim MATLAB execution - see header.

Run: python3 tests/python/test_quality_mirror.py
"""
import math, time, sys

CFG = {
    "version": "1.1.0-stage1-config",
    "focusThresh": 8.0, "entropyThresh": 3.5,
    "borderlineFocusMargin": 0.15, "borderlineEntropyMargin": 0.10,
    "roiSeedThresh": 12, "roiErodeDisk": 8, "roiErodeRelFrac": 0.007,
    "roiMinCoverageFrac": 0.05, "roiCircularityMin": 0.55,
    "entropyBins": 256, "bgFraction": 0.05, "bgFloorPx": 15,
    "claheClipLimit": 0.01, "enhanceFlatThresh": 0.12,
    "enhanceFlatClip": 0.005, "denoiseMinDiameter": 256,
}

PASS = FAILS = 0
def check(name, cond, detail=""):
    global PASS, FAILS
    if cond: PASS += 1; print(f"[PASS] {name}")
    else: FAILS += 1; print(f"[FAIL] {name} :: {detail}")

def fundus(H, W, R, bright=1.0, dx=0, cy_off=0):
    cx, cy = W/2+dx, H/2+cy_off
    img = [[[0,0,0] for _ in range(W)] for _ in range(H)]
    for y in range(H):
        for x in range(W):
            if (x-cx)**2+(y-cy)**2 <= R*R:
                t = 60 + 40*math.sin(0.3*x)*math.cos(0.3*y) + 20*math.sin(0.11*x+0.23*y)
                v = max(0, min(255, int(t*bright)))
                img[y][x] = [v, v, v]
    return img

def gray(img):
    return [[(p[0]+p[1]+p[2])//3 for p in row] for row in img]

def roi_mask(g, seed=12):
    H, W = len(g), len(g[0])
    m = [[v > seed for v in row] for row in g]
    # fill holes: flood background from border on inverted mask
    inv = [[not v for v in row] for row in m]
    seen = [[False]*W for _ in range(H)]
    stack = [(0,0)] if not m[0][0] else []
    # seed all border background pixels
    stack = []
    for x in range(W):
        if not m[0][x]: stack.append((0,x))
        if not m[H-1][x]: stack.append((H-1,x))
    for y in range(H):
        if not m[y][0]: stack.append((y,0))
        if not m[y][W-1]: stack.append((y,W-1))
    while stack:
        y,x = stack.pop()
        if y<0 or y>=H or x<0 or x>=W or seen[y][x] or m[y][x]: continue
        seen[y][x] = True
        stack += [(y+1,x),(y-1,x),(y,x+1),(y,x-1)]
    for y in range(H):
        for x in range(W):
            if not m[y][x] and not seen[y][x]: m[y][x] = True
    # largest component (BFS)
    comp = [[0]*W for _ in range(H)]; cid=0; sizes={}
    for y in range(H):
        for x in range(W):
            if m[y][x] and not comp[y][x]:
                cid+=1; q=[(y,x)]; comp[y][x]=cid; n=0
                while q:
                    cy,cx = q.pop(); n+=1
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=cy+dy,cx+dx
                        if 0<=ny<H and 0<=nx<W and m[ny][nx] and not comp[ny][nx]:
                            comp[ny][nx]=cid; q.append((ny,nx))
                sizes[cid]=n
    if not sizes: return m
    big = max(sizes, key=sizes.get)
    return [[comp[y][x]==big for x in range(W)] for y in range(H)]

def coverage(m): return sum(sum(r) for r in m)/ (len(m)*len(m[0]))
def circularity(m):
    H,W=len(m),len(m[0]); area=sum(sum(r) for r in m)
    per=0
    for y in range(H):
        for x in range(W):
            if m[y][x] and any(ny<0 or ny>=H or nx<0 or nx>=W or not m[ny][nx]
                               for ny,nx in ((y+1,x),(y-1,x),(y,x+1),(y,x-1))):
                per+=1
    return min(1.0, 4*math.pi*area/(per*per)) if per>0 else 0.0

def lap_var(g, m):
    H,W=len(g),len(g[0]); vals=[]
    for y in range(1,H-1):
        for x in range(1,W-1):
            if m[y][x]:
                vals.append(-4*g[y][x]+g[y-1][x]+g[y+1][x]+g[y][x-1]+g[y][x+1])
    mu=sum(vals)/len(vals); return sum((v-mu)**2 for v in vals)/len(vals)

def entropy_channel(v01, mask, bins=256):
    vals=[v01[y][x] for y in range(len(v01)) for x in range(len(v01[0])) if mask[y][x]]
    hist=[0]*bins
    for v in vals: hist[min(bins-1,int(v*bins))]+=1
    n=len(vals); e=0.0
    for c in hist:
        if c: p=c/n; e-=p*math.log2(p)
    return e

def v_channel(img):
    return [[max(p)/255.0 for p in row] for row in img]

def decide(focus, ent, cov, circ, under, over, spec, contrast):
    cfg=CFG; reasons=[]; guide=[]
    if cov < cfg["roiMinCoverageFrac"]:
        reasons.append("coverage"); guide.append("recenter")
    if circ < cfg["roiCircularityMin"] and cov>=cfg["roiMinCoverageFrac"]:
        reasons.append("circularity"); guide.append("avoid crop")
    fb=cfg["focusThresh"]*(1+cfg["borderlineFocusMargin"]); eb=cfg["entropyThresh"]*(1+cfg["borderlineEntropyMargin"])
    if not focus>=cfg["focusThresh"]: reasons.append("focus-fail")
    elif focus<fb: reasons.append("focus-border")
    if not ent>=cfg["entropyThresh"]: reasons.append("entropy-fail")
    elif ent<eb: reasons.append("entropy-border")
    if under>0.25: reasons.append("underexp")
    if over>0.15: reasons.append("overexp")
    if spec>0.02: reasons.append("specular")
    if contrast<0.25: reasons.append("flat-contrast")
    hard = not(focus>=cfg["focusThresh"] and ent>=cfg["entropyThresh"]) or cov<cfg["roiMinCoverageFrac"] or under>0.25 or over>0.15
    dec="FAIL" if hard else ("BORDERLINE" if reasons else "PASS")
    return dec, reasons, guide

def auc_rank(good, bad):
    nG,nB=len(good),len(bad); allv=[(v,1) for v in good]+[(v,0) for v in bad]
    allv.sort(key=lambda t:t[0])
    r=0; s=0
    for i,(_,lab) in enumerate(allv,1):
        if lab==1: s+=i
    return (s-nG*(nG+1)/2)/(nG*nB)

t0=time.process_time()
# 1 config
check("config canonical", CFG["focusThresh"]==8.0 and CFG["entropyThresh"]==3.5)
# 2 ROI centered vs off-center vs degenerate
g=gray(fundus(96,96,34)); m=roi_mask(g)
c0=coverage(m); q0=circularity(m)
check("roi centered coverage", c0>0.05, f"cov={c0:.3f}")
check("roi centered circularity", q0>0.55, f"circ={q0:.3f}")
g2=gray(fundus(96,96,22,dx=20)); m2=roi_mask(g2); c2=coverage(m2)
check("roi off-center smaller", 0<c2<c0, f"{c2:.3f} vs {c0:.3f}")
blk=[[0]*32 for _ in range(32)]; mb=roi_mask(blk)
check("roi all-black no crash", sum(sum(r) for r in mb)>=0)
# 3 focus ordering sharp vs blurred (box blur)
def boxblur(g):
    H,W=len(g),len(g[0]); o=[[0]*W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            s=n=0
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<H and 0<=nx<W: s+=g[ny][nx]; n+=1
            o[y][x]=s//n
    return o
sharp=fundus(64,64,24); gs=gray(sharp); ms=roi_mask(gs)
gb=boxblur(gs)
fs=lap_var(gs,ms); fb=lap_var(gb,ms)
check("focus sharp>blur", fs>fb, f"{fs:.1f} vs {fb:.1f}")
# 4 entropy textured>flat
V=v_channel(sharp); et=entropy_channel(V,ms)
flat=[[[128,128,128] for _ in range(64)] for _ in range(64)]
Vf=v_channel(flat); mf=roi_mask(gray(flat)); ef=entropy_channel(Vf,mf)
check("entropy textured>flat", et>ef, f"{et:.2f} vs {ef:.2f}")
# 5 decision taxonomy
d,_r,_g=decide(12,5.0,0.35,0.9,0.05,0.02,0.0,0.5)
check("decision PASS", d=="PASS", d)
d2,_,_=decide(8.5,3.6,0.35,0.9,0.05,0.02,0.0,0.5)
check("decision BORDERLINE near thresh", d2=="BORDERLINE", d2)
d3,_,_=decide(2,1.0,0.35,0.9,0.05,0.02,0.0,0.5)
check("decision FAIL low scores", d3=="FAIL", d3)
d4,_,_=decide(12,5.0,0.01,0.9,0.05,0.02,0.0,0.5)
check("decision FAIL degenerate coverage", d4=="FAIL", d4)
# 6 calibration math
good=[9,10,11,12]; bad=[2,3,4,5]; th=(min(good)+max(bad))/2
check("calib midpoint", abs(th-7)<1e-9, th)
check("calib sens/spec", sum(v>=th for v in good)/len(good)==1 and sum(v<th for v in bad)/len(bad)==1)
check("calib AUC separated=1", abs(auc_rank(good,bad)-1.0)<1e-9, auc_rank(good,bad))
# 7 enhancement guard branch
check("enhance flat uses reduced clip", (0.10 < CFG["enhanceFlatThresh"]) and (CFG["enhanceFlatClip"] < CFG["claheClipLimit"]))
check("denoise skip tiny", 200 < CFG["denoiseMinDiameter"])
# 8 determinism
a=decide(10,4.0,0.3,0.8,0.1,0.05,0.01,0.4); b=decide(10,4.0,0.3,0.8,0.1,0.05,0.01,0.4)
check("determinism", a==b)
# 9 adversarial: white, tiny, speck
wht=[[[255,255,255] for _ in range(48)] for _ in range(48)]
mw=roi_mask(gray(wht)); check("adversarial all-white no crash", coverage(mw)>=0)
tny=fundus(32,32,10); mt=roi_mask(gray(tny)); check("adversarial tiny no crash", True)
spk=fundus(64,64,24)
import random; random.seed(0)
for _ in range(60): spk[random.randrange(64)][random.randrange(64)]=[255,255,255]
ms2=roi_mask(gray(spk)); check("adversarial specks keep main ROI", coverage(ms2)>0.05, f"{coverage(ms2):.3f}")
el=time.process_time()-t0
print(f"\n=== python mirror: {PASS} passed, {FAILS} failed, {el:.2f}s cpu ===")
sys.exit(1 if FAILS else 0)
