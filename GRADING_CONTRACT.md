# Grading Contract — Stage-3 DR Severity Grading (Contributor B)

Source files (actual current tree; `qualityConfig.m`,
`assessFundusQuality.m` and `QUALITY_CONTRACT.md` named in the mission do
**not** exist - Stage-1 is `assessAndEnhanceImage.m` +
`calibrateQualityThresholds.m`, consumed via its outputs, never redesigned):

- Owned: `gradingConfig.m`, `ordinalGradingLoss.m`,
  `buildGradingFusionTensor.m`, `evaluateGrading.m`,
  `runGradingSelfTests.m`, `train_DR_Grader.m`, `buildGradingDatasets.m`,
  `tests/grading_mirror.py`, `tests/test_grading_mirror.py`, this file.
- Supported (minimal additive edits only): `train_Baseline_ResNet50.m`
  (per-model checkpoints + metadata; training semantics unchanged).
- Untouched: `computeQWK.m`, `wilsonScoreInterval.m`,
  `calibrateTemperature.m`, `compareModels.m`, `runScreeningPipeline.m`,
  `production_inference.m`, `loadModelsIfPresent.m`,
  `getOrLoadCachedModels.m`, every Stage-1/Stage-2 file and test
  (verified via `test_stage1_stage2_files_untouched`).

## 1. Frozen five-channel input (P0)

Order `gradingConfig.channelOrder = {R,G,B,vessel,lesion}`, `single`
`[224 224 5]`, dlarray `'SSC'`, `imageInputLayer([224 224 5],
Normalization='zscore', Name='fusion_input')`. Channels 1-3 are enhanced
RGB `0-255` (never raw, except the deliberate baseline ablation which
never calls the fusion builder); channels 4-5 are predicted masks
`{0,255}` with `lesion = maheMask | exudateMask` (Stage-2 contract - the
mission's "vessel + MA/HE + exudate" is this merged lesion channel, NOT a
6-channel redesign). Photos `bilinear`, masks `nearest` (Stage-2 fix);
fractional mask values error loudly. Labels ICDR `0..4`
(`categorical(grades,0:4)`, one-hot, `pred = argmax-1`), head `dr_fc` 5
outputs + `prob` softmax, referable `>= 2`. GT masks in fusion = leakage;
`train_DR_Grader.m` errors without the three `unet_*.mat` files.

## 2. Datasets (P1)

`buildGradingDatasets.m` split identity preserved (DDR `85/15 seed 42` +
Messidor-2 held out; Messidor-only `70/15/15` patient-level via
`pairs.csv`) so pipeline vs baseline splits stay identical for the paired
McNemar test. Added: grade-range gates, `.patientId`/`.source`
provenance, pairwise disjoint asserts, duplicate tripwire, per-split class
histograms + imbalance ratio, `DDR_seg` overlap warning (seg-seen IDs get
memorized fusion channels), 5th-output `manifest`, no-test-tuning rule in
the log. `buildPatientLevelSplit.m` untouched.

## 3. Architecture (P2) and loss (P3)

DenseNet-121 via `imagePretrainedNetwork` (+legacy fallback), stem-conv
detection by graph order (first 3-channel `Convolution2DLayer`, not name
substring), RGB-mean init for the 2 extra mask channels (pretrained RGB
stays meaningful, deterministic under `rng(42)`), guarded `dr_fc/prob`
head with exactly-5-output assert, headless-safe `plot`, per-model
`checkpoints/grader_densenet121/` + resume. Loss keeps the hybrid idea
(`CE + lambda*E-grade-MSE`, `lambda=0.5`) with shape/probability guards,
scale audit (`CE=-log p`, penalty `[0,16]`, quadratic in distance so far
misses dominate), bounded gradients, and opt-in median-frequency class
weights (logged, default off, val-justified only).

## 4. Imbalance, eval, baseline, metadata (P4-P7)

Imbalance is reported (counts + weights logged) not silently fixed.
`evaluateGrading.m` (held-out only) reports QWK (shared `computeQWK`),
macro/weighted F1, per-class recall/specificity/precision, 5x5 confusion,
referable sens/spec + Wilson CIs, error-distance (adjacent vs severe),
optional referable/macro-OVR AUC; `compareModels.m` stays the paired
baseline-vs-pipeline gate (McNemar). Saved `.meta.json` now records
channel order, input size, class mapping, loss/optimizer/seed, train/val
counts + hists, `validationStory` + manifest, toolbox versions.

## 5. Verification status (acceptance mapping)

- MEASURED (this box): `python3 -m pytest tests/ -q` → **40/40 pass**
  (17 Stage-2 + 23 Stage-3), stdlib+Pillow only, no MATLAB/data.
- PYTHON-SYNTHETIC: grade mapping, ordinal math (far>adjacent at equal
  CE), QWK `0.88940092`, Wilson statsmodels bounds, confusion/F1,
  referable counts, AUC ordering, fusion shape/dtype/OR-merge, manifest
  gates - all synthetic, never clinical accuracy.
- MATLAB-GATED (UNVERIFIED): `trainnet` runs, checkpoint resume,
  `pixelLabelDatastore`/DenseNet/`train_DR_Grader`/`runGradingSelfTests`
  (4 checks) - run in MATLAB before trusting weights.
- DATA-GATED (UNVERIFIED): QWK/sensitivity/specificity/AUC on DDR or
  Messidor-2 - report only via `evaluateGrading` on held-out data, never
  fabricate (`>90%/>85%`, FDA/CE, "diagnosis" claims forbidden).
- NOT IMPLEMENTED: VB/IRMA detectors, NVD validation, Simulink/CORS/DB
  work, 512-vs-768 seg benchmark, lambda/weight val-tuning itself.
