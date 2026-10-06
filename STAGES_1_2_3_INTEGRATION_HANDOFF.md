# STAGES 1–2–3 Integration Handoff (AUTHORITATIVE)

This file is the single authoritative Stage 1–3 contract. Stage-scoped
docs remain as references: `QUALITY_CONTRACT.md` (Stage-1 detail),
`SEGMENTATION_HANDOFF.md` (Stage-2 detail), `GRADING_CONTRACT.md`
(Stage-3 detail; its "Stage-1 files do not exist" note was
origin-relative and is superseded here), `STAGE12_INTEGRATION_HANDOFF.md`
(prior merge record). On any conflict, THIS file wins.

## 0. A/B discrepancy + P1+P2+P3 merge — reconciled against the repo

- Agent A CORRECT on integrated `main`: `qualityConfig.m`,
  `assessFundusQuality.m`, `qualityLoadCalibration.m`,
  `QUALITY_CONTRACT.md` all exist.
- Agent B described `origin/main` pre-merge (Stage-2/3 only; Stage-1 had
  never been pushed there), hence "Stage-1 files do not exist".
- REMOTE BASE: `origin/main 341c4f3` (Stage-3 grading, 6 commits).
  LOCAL BASE: `9fbf613` (Stage-1 ×8 + Stage-2 merge + integration).
  Merge-base: `6a69a74`. Merged clean (no conflicts) — Stage-3 files are
  purely additive (`gradingConfig`, `buildGradingFusionTensor`,
  `ordinalGradingLoss`, `evaluateGrading`, `runGradingSelfTests`,
  grading mirrors + hardened `train_DR_Grader`/`buildGradingDatasets`/
  `train_Baseline_ResNet50`).
- MERGE RESULT: this branch. Nothing from any stage deleted.

## 1. Actual architecture (as built — FINAL after release hardening)

`raw fundus (RGB/gray/uint8/double, HxW[x3], RGBA truncated w/ warning)`
→ Stage-1 `assessFundusQuality` (CANONICAL struct + SINGLE core:
  validation → ROI → focus/entropy → guarded enhancement → extended
  metrics → PASS/BORDERLINE/FAIL). Legacy `assessAndEnhanceImage`
  6-output wrapper DELEGATES here (legacy boolean = threshold-only
  comparison; old error IDs preserved). `qualityConfig` sole source
  (`8`/`3.5`, 15%/10% BORDERLINE bands from EFFECTIVE thresholds);
  `qualityThresholds.mat + .meta.json` override via
  `qualityLoadCalibration` (pwd convention, loud fallback; explicit
  thresholds always win; `[]` skips calibration for enhance-only) →
  `enhancedRGB (HxWx3 uint8, bg=0)`, `enhancedGray (HxW uint8, bg=0)`,
  `roiMask (HxW logical)`
→ `runScreeningPipeline` consumes the canonical report directly:
  exposes `qualityDecision/reasons/guidance`, effective
  `focusThresh/entropyThresh`, `qualityCalibrated/Source`; FAIL
  hard-rejects (`ungradeable`), BORDERLINE proceeds gradeable-with-warning
  (decision + reasons preserved, never coerced); `screenOneImage` passes
  fields through (additive, bridge ignores unknown keys);
  `production_inference` reads effective values from `r` (no second load —
  gate/report divergence structurally closed)
→ Stage-2 `preprocessFundusForSegmentation` (bilinear + `single/255` →
  `[512 512]` from `segmentationConfig`) → `runSegmentationNet` per
  target (bilinear down, argmax class 2 = Foreground, nearest
  resize-back + re-mask by ORIGINAL roi) → `vessel / mahe / exudate`
  logical + `lesionMask = mahe|exudate`
→ Stage-3 `buildGradingFusionTensor` (SINGLE implementation, training AND
  inference): `single(cat(3, enhancedRGB bilinear, vessel*255 nearest,
  lesion*255 nearest))` = `[224 224 5]`, order `[R,G,B,vessel,lesion]`,
  loud validation (NaN/Inf, size mismatch, fractional masks, shape)
→ DenseNet-121 (`imagePretrainedNetwork`, head `dr_fc` + `prob`
  softmax, 5 outputs; first-conv expanded from `mean(oldW,3)`
  deterministically) + `ordinalGradingLoss` (`CE + 0.5*E-grade-MSE`,
  audited guards) → ICDR `dlGrade = idx-1 ∈ {0..4}`, referable `>= 2`.
  Fusion input `0-255 single` matches `imageInputLayer([224 224 5],
  Normalization='zscore')` by design (zscore normalizes; builder warns on
  0-1 input as out-of-distribution).

## 2. Exact contracts

- Quality: new code `rep = assessFundusQuality(img)`; legacy 6-output
  frozen. Thresholds from `qualityConfig` (`8`/`3.5`, 15%/10%
  BORDERLINE bands); `qualityThresholds.mat + .meta.json` override via
  `qualityLoadCalibration` (pwd convention, loud fallback). Legacy
  `-Inf,-Inf` enhance-only calls INTENTIONAL (preprocess, grader
  fusion build, eval runner, visualizer, calibrator scorer, tests).
  `runScreeningPipeline` gate is the sole enforcing call.
- Segmentation: 3 binary targets; `segmentationConfig` sole source
  (`[512 512]`, `labelIDs [0 255]`, class 2 = Foreground, Tversky,
  `seed 42`, `fusionSize [224 224]`). Masks logical / `0/255` disk
  (`0/1` converts w/ warning, mixed = ERROR, inversion = ERROR).
  Photos bilinear, masks nearest EVERYWHERE. ROI never re-derived
  (`roiMaskSizeMismatch`). Train/infer parity. Predicted masks ONLY
  (`segmentationNotTrained` gate; GT-in-fusion = leakage).
- Grading: `gradingConfig` sole source (`inputSize [224 224]`,
  `photoInterp bilinear`, `maskInterp nearest`, `numClasses 5`,
  `channelOrder R,G,B,vessel,lesion`, `ordinalLambda 0.5`,
  `referableThreshold 2`, `seed 42`). First-conv expanded from
  `mean(oldW,3)` deterministically. Checkpoints per-model + resume +
  rich metadata (`saveModelWithMetadata`). Datasets: DDR 85/15 seed 42
  + Messidor-2 held out (or Messidor-only 70/15/15 patient-level);
  grade gates, provenance, disjoint asserts, duplicate tripwire,
  `DDR_seg ⊂ DDR` overlap warning, manifest, no-test-tuning rule.
- Orchestration: `runScreeningPipeline` (single core, canonical
  quality fields), `screenOneImage` lean struct + additive quality
  passthrough, `runBatchScreening` CSV unchanged (continues on bad
  image), `production_inference` report (thresholds/decision from `r`),
  `getOrLoadCachedModels` persistent all-or-nothing nets +
  temperature hot-reload on `.mat` mtime (nets never reloaded).
  Field names consistent. No silent failure.

## 3. Leakage map

Seg lists stem-paired (orphans/duplicates raise); seeded group-aware
splits with leak asserts. Grader trains on PREDICTED masks only.
`DDR_seg ⊂ DDR` overlap → memorize risk, dedupe before
cross-population claims. DDR grading has no patient IDs (honest
image-level split). Calibration sets must be val-only, never test.
No test tuning in code. All VERIFIED (static + synthetic).

## 4. Test results (executed here, Python 3.12, no MATLAB/data/weights)

- `tests/test_stages123_hardening.py`: **66/66 PASS** (grading + quality
  + chain + adversarial + release-hardening pins: BORDERLINE
  non-coercion, single-source calibration, hot-reload, zscore/0-255,
  validation IDs, loud failures).
- `tests/test_stage12_contract.py`: **49/49 PASS** (canonical core,
  single fusion builder, configs).
- `tests/python/test_quality_mirror.py`: **20/20 PASS**.
- `tests/test_grading_mirror.py` manual runner: **23/23 PASS**.
- Seg mirror manual: 14/17 (3 need `pytest` pkg; logic covered above).

## 5. MATLAB-GATED (written, UNEXECUTED — never claim as run)

`runSelfTests` (14 checks), `runGradingSelfTests`,
`testQualitySubsystem`, `testStage12Integration`, `trainnet`/resume/
`pixelLabelDatastore`, `initialize(net)` semantics, eval/bench suites,
PLUS new hardening surface: wrapper/canonical numeric parity run,
builder-unified inference run, temperature hot-reload run, BORDERLINE
end-to-end run.

## 6. DATA-GATED (no datasets/weights here)

Real Dice/QWK/sens/spec/AUC; per-camera calibration; DDR overlap
dedupe; DIARETDB1 inspection; 512v768; temperature run.

## 7. Known limitations / REMAINING

- Grading hyperparams live in `gradingConfig` (Stage-3 centralized;
  no action).
- GT-vs-predicted is a caller guarantee the builder cannot verify
  pixel-wise (documented at call sites; enforced by
  `segmentationNotTrained` gate + review discipline).
- Calibration/temperature `.mat` files resolve via pwd convention
  (same as model files); nonstandard launch dirs need explicit paths.
- `initialize(net)` post-weight-set semantics need a MATLAB check.
- VB/IRMA absent; NV proxy; UI/backend/DB/Simulink/Grad-CAM/4-2-1
  untouched per scope.

## 8. Stage-4 prerequisites

1. MATLAB: `runSelfTests; runGradingSelfTests; testQualitySubsystem; testStage12Integration`
   + new-surface runs (wrapper/canonical parity, unified fusion, hot-reload, BORDERLINE e2e).
2. Calibrate quality per camera; honor BORDERLINE in curation.
3. Train 3 seg nets → `evaluateSegmentationDataset` per target/dataset.
4. Train grader on PREDICTED masks; resolve DDR overlap wording.
5. No clinical reasoning until 1–4 measured. STAGE-4 READY: NO.
