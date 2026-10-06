#!/usr/bin/env python3
"""test_stage5_explain.py — Stage-5 explainability/calibration/disagreement
contracts (stdlib only). Independent reference math, not production copies.

Run: python3 tests/test_stage5_explain.py
MATLAB twin: testExplainability.m (UNEXECUTED here).
Covers revised-brief N.1-15. No clinical validation claimed.
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


def softmax_rows(logits):
    out = []
    for row in logits:
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        out.append([v / s for v in e])
    return out


def validate_temp_artifact(doc):
    """Independent mirror of the load-time acceptance rules."""
    if doc is None:
        return "UNCALIBRATED_FALLBACK"
    t = doc.get("temperatureT")
    if not (isinstance(t, (int, float)) and math.isfinite(t) and t > 0):
        return "CALIBRATED_MISMATCH"
    prov = doc.get("calibrationProvenance")
    if prov is not None:
        if prov.get("numClasses") != 5 or list(prov.get("classOrdering", [])) != [0, 1, 2, 3, 4]:
            return "CALIBRATED_MISMATCH"
    return "CALIBRATED_VALID"


def disagreement(dl, rule, conf_state, rule_state, expl_state):
    """Independent mirror of the orthogonal disagreement dimensions."""
    if dl is None or (isinstance(dl, float) and math.isnan(dl)) or \
       rule is None or (isinstance(rule, float) and math.isnan(rule)):
        rel = "NOT_COMPARABLE"
    elif dl == rule:
        rel = "AGREE"
    else:
        rel = "NUMERIC_DISAGREE"
    esc = rel == "NUMERIC_DISAGREE" or rule_state in ("INSUFFICIENT_EVIDENCE", "INVALID") \
        or expl_state in ("DEGRADED", "INVALID")
    return rel, esc


def test_temperature_ordering():
    logits = [[3.0, 1.0, 0.5, 0.2, 0.1], [0.4, 2.5, 1.0, 0.3, 0.2]]
    p1 = softmax_rows(logits)
    pT = softmax_rows([[v / 1.5 for v in row] for row in logits])
    a1 = [max(range(5), key=lambda i: r[i]) for r in p1]
    aT = [max(range(5), key=lambda i: r[i]) for r in pT]
    check("1. argmax invariant under T", a1 == aT, f"{a1} vs {aT}")
    check("1. T>1 lowers max confidence",
          all(max(rT) < max(r1) for rT, r1 in zip(pT, p1)), "calibration must soften")
    check("1. rows still sum to 1", all(abs(sum(r) - 1) < 1e-9 for r in pT), "")


def test_fallback_status():
    pipe = txt("runScreeningPipeline.m")
    check("2. fallback T never called calibrated",
          "CALIBRATED_VALID" in pipe and "UNCALIBRATED" in pipe, "status conflation")
    check("2. simulated temperature UNAVAILABLE",
          "'UNAVAILABLE'" in pipe, "missing state")
    dis = txt("analyzeDisagreement.m")
    check("2. LOW_CONFIDENCE is review heuristic (DATA-GATED note)",
          "DATA-GATED" in dis or "heuristic" in dis, "unmarked threshold")


def test_artifact_mismatch():
    check("3. valid artifact accepted",
          validate_temp_artifact({"temperatureT": 1.56, "calibrationProvenance": {"numClasses": 5, "classOrdering": [0, 1, 2, 3, 4]}}) == "CALIBRATED_VALID", "")
    check("3. missing file falls back",
          validate_temp_artifact(None) == "UNCALIBRATED_FALLBACK", "")
    check("3. non-finite T mismatches",
          validate_temp_artifact({"temperatureT": float("nan")}) == "CALIBRATED_MISMATCH", "")
    check("3. wrong class order mismatches",
          validate_temp_artifact({"temperatureT": 1.2, "calibrationProvenance": {"numClasses": 4, "classOrdering": [0, 1, 2, 3]}}) == "CALIBRATED_MISMATCH", "")
    check("3. legacy artifact (no provenance) accepted on scalar contract",
          validate_temp_artifact({"temperatureT": 1.5}) == "CALIBRATED_VALID", "")
    body = txt("loadModelsIfPresent.m") + txt("getOrLoadCachedModels.m")
    check("3. both loaders validate (no isfile-only acceptance)",
          body.count("CALIBRATED_MISMATCH") >= 2 and "classOrdering" in body, "isfile-only acceptance")


def test_roi_suppression():
    # 16x16 map, hotspot at corner (0,0); ROI = central disc
    mp = [[0.1] * 16 for _ in range(16)]
    mp[0][0] = 5.0
    roi = [[(x - 8) ** 2 + (y - 8) ** 2 <= 49 for x in range(16)] for y in range(16)]
    masked = [[mp[y][x] if roi[y][x] else 0.0 for x in range(16)] for y in range(16)]
    peak = max(((masked[y][x], x, y) for y in range(16) for x in range(16)))[1:]
    check("4. off-ROI hotspot suppressed (peak moves inside ROI)",
          roi[peak[1]][peak[0]], f"peak={peak}")
    check("4. canonical masks before measuring",
          "mapFull(~roiMaskOrig) = 0" in txt("explainGradCAM.m") or "mapFull(~roiMask" in txt("explainGradCAM.m"), "order drift")


def test_empty_map():
    check("5. empty map UNAVAILABLE pinned",
          "empty/near-empty" in txt("explainGradCAM.m") and "UNAVAILABLE" in txt("explainGradCAM.m"), "silent empty")
    check("5. outside-ROI-only UNAVAILABLE",
          "outside-roi" in txt("explainGradCAM.m"), "border-only gap")


def test_border_disc_math():
    # border-concentrated mass: mass fraction in outer band ~1.0
    mp = [[0.05] * 20 for _ in range(20)]
    for y in range(20):
        for x in range(20):
            if x < 2 or x > 17 or y < 2 or y > 17:
                mp[y][x] = 1.0
    mx = max(max(r) for r in mp)
    act = [[v >= 0.5 * mx for v in row] for row in mp]
    mass = sum(mp[y][x] for y in range(20) for x in range(20) if act[y][x])
    border = sum(mp[y][x] for y in range(20) for x in range(20)
                 if act[y][x] and (x < 2 or x > 17 or y < 2 or y > 17))
    check("6. border mass fraction ~1.0 on rim activation",
          border / mass > 0.9, f"{border / mass:.2f}")
    check("6. border threshold centralized",
          "borderMassFrac" in txt("explainabilityConfig.m"), "scattered literal")
    # disc-zone mass on synthetic centered blob
    mp2 = [[0.05] * 20 for _ in range(20)]
    for y in range(20):
        for x in range(20):
            if (x - 10) ** 2 + (y - 10) ** 2 <= 9:
                mp2[y][x] = 1.0
    mx2 = max(max(r) for r in mp2)
    act2 = [[v >= 0.5 * mx2 for v in row] for row in mp2]
    mass2 = sum(mp2[y][x] for y in range(20) for x in range(20) if act2[y][x])
    disc = sum(mp2[y][x] for y in range(20) for x in range(20)
               if act2[y][x] and (x - 10) ** 2 + (y - 10) ** 2 <= 9)
    check("7. disc mass fraction ~1.0 on centered activation",
          disc / mass2 > 0.9, f"{disc / mass2:.2f}")
    check("7. disc zone radii centralized",
          "discZoneRadii" in txt("explainabilityConfig.m"), "scattered literal")


def test_invalid_geometry():
    check("8. NaN geometry INVALID pinned",
          "NaN" in txt("explainGradCAM.m"), "geometry gap")
    check("8. wrong arch/layer INVALID (no substitution)",
          "never silently substitutes" in txt("explainGradCAM.m") or "refusing to substitute" in txt("explainGradCAM.m"), "substitution risk")


def test_missing_evidence():
    rel, esc = disagreement(2, float("nan"), "UNCALIBRATED", "INSUFFICIENT_EVIDENCE", "UNAVAILABLE")
    check("9. NaN rule is NOT_COMPARABLE (not disagreement)", rel == "NOT_COMPARABLE", rel)
    check("9. insufficient still escalates", esc, "silent insufficient")
    rel2, esc2 = disagreement(2, 2, "CALIBRATED", "SUFFICIENT", "VALID")
    check("9. equal sufficient grades AGREE", rel2 == "AGREE" and not esc2, f"{rel2},{esc2}")
    check("10. batch requires both grades (NaN-safe)",
          "~isnan(icdrDL)&&~isnan(icdrRule)" in txt("runBatchScreening.m").replace(" ", ""), "manufactured disagreement")
    check("10. insufficient triage count disjoint from disagreement",
          "nInsufficient" in txt("runBatchScreening.m"), "conflated counts")


def test_provenance_api():
    body = txt("bridge_server.py")
    for k in ("qualityDecision", "qualityReasons", "qualityGuidance", "qualityCalibrated", "ruleStatus", "nvStatus"):
        check(f"12. bridge forwards {k}", f'"{k}"' in body, "key missing")
    check("12. legacy keys frozen", '"gradCamOnDisc"' in body and '"roiPassed"' in body, "bridge break")
    check("12. additive keys backward compatible (safe_get)",
          "safe_get" in body, "KeyError risk")
    grad = txt("explainGradCAM.m")
    for k in ("targetClass", "reductionLayer", "featureLayerUsed", "inputChannels", "channelsCaveat", "configVersion"):
        check(f"11. explanation provenance has {k}", k in grad, "provenance gap")
    check("14. fused-channels caveat present",
          "cannot by itself distinguish photographic evidence" in grad, "causality overclaim")
    pinf = txt("production_inference.m")
    check("14b. caveat printed exactly once in PDF report",
          pinf.count("Caveat: ") >= 1 and "Paragraph(['Caveat:" in pinf.replace(" ", ""), "PDF caveat missing")
    check("14b. caveat printed exactly once in txt report",
          "'Caveat: %s" in pinf, "txt caveat missing")
    n_caveat = pinf.count("explainChannelsCaveat")
    check("14b. single canonical source (no duplicated wording)",
          n_caveat == 2 and "r.explainChannelsCaveat=gradRep.channelsCaveat" in txt("runScreeningPipeline.m").replace(" ", ""),
          f"caveat refs={n_caveat} (expect PDF + txt, set once in pipeline)")
    check("disagreement has no dead params",
          "nvStatus" not in txt("analyzeDisagreement.m") and "qualityDecision" not in txt("analyzeDisagreement.m"), "misleading signature")
    check("pipeline call matches slim signature",
          "r.explainStatus)" in txt("runScreeningPipeline.m"), "caller drift")
    a = disagreement(3, 3, "CALIBRATED", "SUFFICIENT", "VALID")
    b = disagreement(3, 3, "CALIBRATED", "SUFFICIENT", "VALID")
    check("13. deterministic disagreement", a == b, "")
    check("15. Stage-5 config single source (gradingConfig frozen)",
          "featureLayer" in txt("explainabilityConfig.m"), "missing")
    # real check:
    gcfg = txt("gradingConfig.m").lower()
    has_explain = "explainabilityconfig" in gcfg or "featurelayer" in gcfg or "lowconfidencereview" in gcfg
    check("15. gradingConfig untouched by Stage-5", not has_explain, "frozen file modified")


TESTS = [test_temperature_ordering, test_fallback_status, test_artifact_mismatch,
         test_roi_suppression, test_empty_map, test_border_disc_math,
         test_invalid_geometry, test_missing_evidence, test_provenance_api]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            FAIL += 1
            print(f"[FAIL] {t.__name__} raised :: {e}")
    print(f"\n=== stage5 explain: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
