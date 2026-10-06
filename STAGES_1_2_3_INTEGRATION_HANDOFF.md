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

## 1. Actual architecture (as built)

`raw fundus (RGB/gray/uint8/double, HxW[x3])`
→ Stage-1 `assessAndEnhanceImage` (legacy 6-output, frozen) /
  `assessFundusQuality` (canonical struct: PASS/BORDERLINE/FAIL +
  reasons/guidance) via `qualityConfig` + `qualityThresholds.mat`
  override (gate honors calibration — fixed in integration) →
  `enhancedRGB (HxWx3 uint8, bg=0)`, `enhancedGray (HxW uint8, bg=0)`,
  `roiMask (HxW logical)`
→ Stage-2 `preprocessFundusForSegmentation` (bilinear + `single/255` →
  `[512 512]` from `segmentationConfig`) → `runSegmentationNet` per
  target (bilinear down, argmax class 2 = Foreground, nearest
  resize-back + re-mask by ORIGINAL roi) → `vessel / mahe / exudate`
  logical + `lesionMask = mahe|exudate`
→ Stage-3 `buildGradingFusionTensor` (training; inference inlines
  byte-identical semantics by documented decision):
  `single(cat(3, enhancedRGB bilinear, vessel*255 nearest,
  lesion*255 nearest))` = `[224 224 5]`, order `[R,G,B,vessel,lesion]`
→ DenseNet-121 (`imagePretrainedNetwork`, head `dr_fc` + `prob`
  softmax, 5 outputs) + `ordinalGradingLoss` (`CE + 0.5*E-grade-MSE`,
  audited guards) → ICDR `dlGrade = idx-1 ∈ {0..4}`, referable `>= 2`.

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
- Orchestration: `runScreeningPipeline` (single core), `screenOneImage`
  lean struct, `runBatchScreening` CSV (continues on bad image),
  `production_inference` report, `getOrLoadCachedModels` persistent
  all-or-nothing (temperature cached at first load — restart bridge
  after calibrating). Field names consistent. No silent failure.

## 3. Leakage map

Seg lists stem-paired (orphans/duplicates raise); seeded group-aware
splits with leak asserts. Grader trains on PREDICTED masks only.
`DDR_seg ⊂ DDR` overlap → memorize risk, dedupe before
cross-population claims. DDR grading has no patient IDs (honest
image-level split). Calibration sets must be val-only, never test.
No test tuning in code. All VERIFIED (static + synthetic).

## 4. Test results (executed here, Python 3.12, no MATLAB/data/weights)

- NEW/UPDATED `tests/test_stages123_hardening.py`: **49/49 PASS**.
- `tests/test_stage12_contract.py`: **44/44 PASS** (updated for
  builder indirection).
- `tests/python/test_quality_mirror.py`: **20/20 PASS**.
- `tests/test_grading_mirror.py` manual runner: **23/23 PASS**.
- Seg mirror manual: 14/17 (3 need `pytest` pkg; logic covered above).

## 5. MATLAB-GATED (written, UNEXECUTED — never claim as run)

`runSelfTests` (14 checks), `runGradingSelfTests`,
`testQualitySubsystem`, `testStage12Integration`, `trainnet`/resume/
`pixelLabelDatastore`, `initialize(net)` semantics, eval/bench suites.

## 6. DATA-GATED (no datasets/weights here)

Real Dice/QWK/sens/spec/AUC; per-camera calibration; DDR overlap
dedupe; DIARETDB1 inspection; 512v768; temperature run.

## 7. Known limitations / REMAINING

- Inference fusion inlines builder semantics (documented byte-identical,
  deliberate — do not "fix" stylistically).
- Temperature cached at first load (restart after calibrating).
- Stage-2/3 call legacy quality API (compatible; struct API adds
  metadata grading doesn't consume — migration deferred, not drift).
- VB/IRMA absent; NV proxy; UI/backend/DB/Simulink/Grad-CAM/4-2-1
  untouched per scope.

## 8. Stage-4 prerequisites

1. MATLAB: `runSelfTests; runGradingSelfTests; testQualitySubsystem; testStage12Integration`.
2. Calibrate quality per camera; honor BORDERLINE in curation.
3. Train 3 seg nets → `evaluateSegmentationDataset` per target/dataset.
4. Train grader on PREDICTED masks; resolve DDR overlap wording.
5. No clinical reasoning until 1–4 measured. STAGE-4 READY: NO.
