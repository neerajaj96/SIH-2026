# STAGES 1–2–3 Integration Handoff (AUTHORITATIVE)

This file is the single authoritative Stage 1–3 contract. Stage-scoped
docs remain as references: `QUALITY_CONTRACT.md` (Stage-1 detail),
`SEGMENTATION_HANDOFF.md` (Stage-2 detail), `STAGE12_INTEGRATION_HANDOFF.md`
(prior merge record). On any conflict, THIS file wins.

## 0. A/B discrepancy — reconciled against the repo (authoritative)

- Agent A CORRECT on current `main` (`662c366` + merge `60975e7`):
  `qualityConfig.m`, `assessFundusQuality.m`, `qualityLoadCalibration.m`,
  `QUALITY_CONTRACT.md` all exist.
- Agent B described `origin/main` (`6a69a74`), which contains Stage-2 only
  — Stage-1 was never pushed there, so B saw pre-Stage-1 files
  (`assessAndEnhanceImage.m` + `calibrateQualityThresholds.m` only).
- Graph: base `5d00f50` → Stage-1 line (8 commits) + Stage-2 line
  (12 commits on origin) → merged here (`60975e7`) + integration
  (`662c366`). Nothing deleted; both lines preserved. `runSelfTests.m`
  conflict resolved by keeping BOTH suites (14 checks).

## 1. Actual architecture (as built)

`raw fundus (RGB/gray/uint8/double, HxW[x3])`
→ Stage-1 `assessAndEnhanceImage` (legacy 6-output, frozen) /
  `assessFundusQuality` (canonical struct: PASS/BORDERLINE/FAIL +
  reasons/guidance) via `qualityConfig` + `qualityThresholds.mat`
  override → `enhancedRGB (HxWx3 uint8, bg=0)`,
  `enhancedGray (HxW uint8, bg=0)`, `roiMask (HxW logical)`
→ Stage-2 `preprocessFundusForSegmentation` (bilinear + `single/255` →
  `[512 512]`, mask nearest) → `runSegmentationNet(net, enhancedGray,
  roiMask)` per target (bilinear down, argmax class 2 = Foreground,
  nearest resize-back + re-mask by ORIGINAL roi) → `vessel / mahe /
  exudate` logical + `lesionMask = mahe|exudate`
→ Stage-3 fusion `single(cat(3, imresize(enhancedRGB,[224 224],
  'bilinear'), imresize(uint8(vessel)*255,[224 224],'nearest'),
  imresize(uint8(lesion)*255,[224 224],'nearest')))` = `[224 224 5]`
  → DenseNet-121 (`imagePretrainedNetwork("densenet121")`, head
  `dr_fc` + `softmax prob`, 5 outputs) + hybrid
  `CE + 0.5*ordinalPenalty` loss → ICDR `dlGrade = idx-1 ∈ {0..4}`.

## 2. Exact contracts

- Quality API: new code `rep = assessFundusQuality(img)`; legacy
  `[isGradeable,enhancedRGB,enhancedGray,focus,entropy,roiMask] =
  assessAndEnhanceImage(img,ft,et)` frozen. Thresholds: `qualityConfig`
  canonical (`8`/`3.5` + 15%/10% BORDERLINE bands); calibrated
  `qualityThresholds.mat + .meta.json` via `calibrateQualityThresholds`
  (`OutputMat`, sens/spec/AUC + overlap warning), loaded by
  `qualityLoadCalibration` (pwd convention, loud fallback). FIXED HERE:
  `runScreeningPipeline` gate now honors calibration (was canonical-only
  while `production_inference` displayed calibrated values).
- Segmentation: exactly 3 binary targets (vessel, MA/HE, exudate);
  `segmentationConfig` sole source (`inputSize/imageSize [512 512(/1)]`,
  `classNames [Background,Foreground]`, `labelIDs [0 255]`, Tversky
  per-target, `seed 42`, `fusionSize [224 224]`). Masks logical in
  memory, `0/255` PNG on disk (`0/1` auto-converts w/ warning, mixed =
  ERROR, inversion = ERROR). Photos bilinear, masks nearest EVERYWHERE.
  ROI never re-derived (`roiMaskSizeMismatch` error). Train/infer parity
  (bilinear + single/255, nearest mask). Predicted masks ONLY downstream
  (`train_DR_Grader:segmentationNotTrained` gate; GT-in-fusion = leak).
- Grading: DenseNet-121, 5 channels ordered
  `[R,G,B,vessel,lesion]` (lesion = mahe|exudate), ICDR 0–4, canonical
  helper `buildFusionTensor` used in training; inference mirrors it in
  `runScreeningPipeline`. First-conv expanded deterministically from
  `mean(oldW,3)` (no rng). Checkpoint: `saveModelWithMetadata` (lambda +
  validationStory + sizes/seed/counts). KNOWN GAP: grading hyperparams
  (`lambda 0.5`, layer names) inline in `train_DR_Grader.m`, NOT
  centralized — documented, not redesigned here.
- Orchestration: `runScreeningPipeline` (single core: quality →
  seg → landmarks/quadrants → rule grade → DL grade), `screenOneImage`
  (lean JSON struct), `runBatchScreening` (CSV, continues on bad image),
  `production_inference` (report), `getOrLoadCachedModels` (persistent,
  all-or-nothing 4 nets; temperature cached at first load — restart
  bridge after calibrating), `loadModelsIfPresent` (pwd convention).
  Field names consistent (`dlGrade/dl`, `confidence/conf`,
  `ruleGrade/rule`, `nvFlagged/nv`). No silent failure (status/errorMessage).

## 3. Leakage map

- Seg file lists: stem-paired (case-insensitive, sorted), orphans/
  duplicates raise; `0/1→255` converted. Split: seeded (`seed 42`),
  group-aware, file-list-first, leak assert. DDR_seg ⊂ DDR overlap
  (memorization risk for grader fusion) — DATA-GATED, dedupe before
  cross-population claims. DDR grading: NO patient IDs → honest
  image-level split. Messidor-2: patient/eye-paired split. Calibration
  sets (`data/gradeable`) must be val-only, never test — DATA-GATED.
- No GT masks in fusion; no test tuning in code.

## 4. Test results (executed here, Python 3.12, no MATLAB/data/weights)

- NEW `tests/test_stages123_hardening.py`: **49/49 PASS** (grading pins,
  quality canonical + calibration fix, chain shapes, 0/1/255 + logical +
  inversion, empty/single-px/tiny, black/white/tiny/off-center, mismatch,
  nearest-vs-linear, NaN/Inf guards, ICDR validity, duplicates/orphans,
  seeded leak-free split, stale-config pins, determinism).
- `tests/test_stage12_contract.py`: **44/44 PASS** (no regression).
- `tests/python/test_quality_mirror.py`: **20/20 PASS** (no regression).
- Seg mirror manual (no pytest pkg): 14/17 PASS; 3 fail only on
  `import pytest` inside test bodies (logic replicated in contract suite).

## 5. MATLAB-GATED (written, UNEXECUTED — never claim as run)

`runSelfTests` (14 checks), `testQualitySubsystem`, `testStage12Integration`,
`trainnet` runs, checkpoint resume, `pixelLabelDatastore [0 255]`,
`initialize(net)` weight-preservation after first-conv expansion,
`evaluateSegmentationDataset`/bench on real nets.

## 6. DATA-GATED (no datasets/weights here)

Real Dice/IoU/sens/spec/QWK/AUC/accuracy; per-camera calibration;
`DDR_seg ⊂ DDR` dedupe; DIARETDB1 confidence-marking inspection;
`512 vs 768` VRAM benchmark; temperature calibration run.

## 7. Known limitations / REMAINING

- Grading config not centralized (`lambda`, layer names inline).
- Temperature cached at first `getOrLoadCachedModels` (stale until restart).
- `initialize(net)` semantics after manual weight set need MATLAB check.
- VB/IRMA detectors absent; NV is screening proxy; UI/backend/DB/Simulink/Grad-CAM/4-2-1 untouched per scope.

## 8. Stage-4 prerequisites

1. MATLAB run: `runSelfTests; testQualitySubsystem; testStage12Integration`.
2. Calibrate quality per camera; honor BORDERLINE in curation.
3. Train 3 seg nets → `evaluateSegmentationDataset` per target/dataset +
   worst-K review + inference bench (macro + pooled + empty audits).
4. Train grader on PREDICTED masks only; resolve DDR overlap wording.
5. Do not start clinical reasoning until 1–4 are measured. STAGE-4 READY: NO.
