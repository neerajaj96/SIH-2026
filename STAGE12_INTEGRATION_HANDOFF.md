# Stage-1 + Stage-2 Integration Handoff — SUPERSEDED

> Authoritative record: `STAGES_1_2_3_INTEGRATION_HANDOFF.md` (THIS file is
> retained for history only; on any conflict the 1-2-3 handoff wins).
> Canonical core is `assessFundusQuality` (`assessAndEnhanceImage` is a
> legacy wrapper). Point-in-time counts below are stale; see the 1-2-3
> handoff for current suite tallies.

Merged line: Stage-1 (8 local commits: qualityConfig, assessFundusQuality,
ROI/enhancement guards, calibrator .mat, dual harness) + Stage-2
(origin/main 12 commits: segmentationConfig, paired lists, validator,
leak-free split, parity, fusion-nearest, eval, DDR merge, bench) via
merge commit `60975e7`. `runSelfTests.m` conflict resolved by keeping
BOTH suites (now 14 checks: 8 original + 2 quality + 4 seg).

## Dependency / contract map (verified on merged tree)

`raw RGB/gray` → `assessAndEnhanceImage` (legacy 6-output, defaults from
`qualityConfig`; gate in `runScreeningPipeline` reads `qcfg`, never
`8/3.5` literals; train/infer call with `-Inf,-Inf` = enhance-only) →
`enhancedGray/enhancedRGB/roiMask` (background-zero, same-size enforced
by `runSegmentationNet:roiMaskSizeMismatch`, never re-derived) →
`preprocessFundusForSegmentation` (bilinear image + `single/255`,
nearest mask → `[512 512]` from `segmentationConfig`) →
`runSegmentationNet` (bilinear down, `nearest` resize-back + re-mask by
ORIGINAL roi) → `vessel/mahe/exudate` logical + `lesionMask=mahe|exudate`
→ `imresize [224 224]` (photos `bilinear`, masks `nearest` in BOTH
`runScreeningPipeline` and `train_DR_Grader`) → 5-channel
`[224 224 5]` fusion of PREDICTED masks only (`train_DR_Grader` errors
`segmentationNotTrained` without `unet_*.mat`; GT-in-fusion would be
leakage). Masks on disk `0/255` (`labelIDs [0 255]`, class 2 =
Foreground); `0/1` auto-converts with warning; inversion = ERROR.

## Issue classification

- VERIFIED (code + executable Python): legacy compat intact; no ROI
  re-derivation; spatial compat; background-zero; single-source configs
  (no stale `512`/`[0 1]` literals); `224` nearest-masks/bilinear-photos
  in both consumers; predicted-only fusion; train/infer parity;
  pairing orphans raise; seeded split leak-free; determinism.
- FIXED (this task): `runSelfTests.m` merge conflict (both suites kept);
  NOTHING else in production touched (shared files read-only).
- MATLAB-GATED (written, UNEXECUTED here): `testStage12Integration.m`,
  `testQualitySubsystem.m` image groups, 14-check `runSelfTests` —
  require MATLAB + Image/Stats toolboxes. Never quoted as run.
- DATA-GATED: no clinical Dice/QWK/sens/spec (no data/weights);
  thresholds still placeholders until per-camera calibration;
  `DDR_seg ⊂ DDR` overlap noted by Stage-2 (dedupe before
  cross-population claims); DIARETDB1 confidence markings uninspected.
- ENV-GATED (pre-existing): 3/17 seg-mirror fns need `pytest`
  (`pytest.raises`); logic replicated without pytest in
  `test_stage12_contract.py` (44/44). `pytest`/`PIL` absent here.

## Known drift / recommendation (no production change made)

Stage-2 calls legacy `assessAndEnhanceImage`, not the newer
`assessFundusQuality` struct API (branches diverged; outputs identical,
so compatible). Per shared-file rule this was left as-is. Stage-3
should adopt `assessFundusQuality` for `BORDERLINE`/`reasons`/`guidance`
while keeping the 6-output wrapper frozen.

## Files changed (this task only)

- `tests/integration_mirror.py` (new), `tests/test_stage12_contract.py`
  (new, 44 checks), `testStage12Integration.m` (new, UNEXECUTED),
  `STAGE12_INTEGRATION_HANDOFF.md` (new), `runSelfTests.m` (conflict
  resolution only — both suites kept).

## Tests executed here

- `python3 tests/test_stage12_contract.py` → **44/44 PASS**.
- `python3 tests/python/test_quality_mirror.py` → **20/20 PASS**.
- Seg mirror manual runner → **14/17 PASS**, 3 fail only on
  `import pytest` (missing package, logic covered in contract suite).

## Not executable here

- Any `.m` test (`runSelfTests`, `testQualitySubsystem`,
  `testStage12Integration`); anything needing MATLAB toolboxes, real
  fundus data, or `unet_*.mat` weights.

## Remaining risks → Stage-3 assumptions

1. Run `runSelfTests; testQualitySubsystem; testStage12Integration` in
   MATLAB before trusting integration numbers.
2. Calibrate quality per camera (`calibrateQualityThresholds` → `.mat`);
   treat `BORDERLINE` as gradeable-with-warning in curation.
3. Train seg nets first (`train_UNet_Segmentation`), then
   `evaluateSegmentationDataset` per target/dataset; report macro +
   pooled + empty audits, never pooled alone.
4. Only then train grader on PREDICTED masks; respect `DDR_seg ⊂ DDR`
   overlap caveat for validation-story wording.
