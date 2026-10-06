# Stage-2 Handoff — Retinal Vessel + Lesion Segmentation

Ownership: `segmentationConfig.m`, `train_UNet_Segmentation.m`,
`preprocessFundusForSegmentation.m`, `runSegmentationNet.m`,
`evaluateSegmentation.m`, `validateMaskConventions.m` +
`buildSegmentationFileLists.m`, `splitSegmentationDataset.m`,
`mergeDDRSegMasks.m`, `evaluateSegmentationDataset.m`,
`visualizeSegmentationFailures.m`, `benchmarkSegmentationInference.m`,
`tests/seg_mirror.py`, `tests/test_seg_mirror.py`.
Stage-1 (`assessAndEnhanceImage.m`, `calibrateQualityThresholds.m`)
untouched except via its stable contract below.

## 1. Achieved system

- **Single config** (`segmentationConfig.m`): `inputSize [512 512]`,
  `imageSize [512 512 1]`, `classNames [Background,Foreground]`,
  `labelIDs [0 255]` (fixes old `[0 1]` vs on-disk `0/255` mismatch),
  Tversky `vessels .3/.7, mahe .3/.7, exudate .3/.6`, `seed 42`,
  `valFraction .15`. Train, preprocess and inference all read it.
- **Paired training data** (`buildSegmentationFileLists.m`): flattens
  nested cells (old `fullfile(cell)` bug), pairs by basename stem
  case-insensitive + sorted, errors on orphans/duplicates, auto-converts
  `0/1 → 0/255` copies, supports `.png/.jpg/.jpeg/.tif/.tiff/.bmp/.gif`.
- **Always-on validator** (`validateMaskConventions.m`): runs even for
  single source; `ERROR` on inversion (`>60% fg` or `>50pp` spread with a
  side `>60%`) and on mixed `0/1 + 0/255`; strict `0-255 ⟺ ⊆{0,1,255}`
  (grayscale/confidence → `CHECK THIS`); blank/full counts; `.jpeg/.bmp`
  coverage; optional image-dim note.
- **Leakage-safe split** (`splitSegmentationDataset.m`): seeded
  (`rng(seed+i)` per target), group-aware (`PatientIdFcn`, image-level
  default logged explicitly), no `subset()` fragility (splits file lists
  first), leakage `assert`, source breakdown report.
- **Parity train/inference**: both `bilinear` image + `single/255`,
  `nearest` mask, `logical` outputs, override warnings if sizes diverge.
  Fusion contract fixed in both consumers
  (`runScreeningPipeline.m`, `train_DR_Grader.m`): photos `bilinear`,
  masks `nearest` (was default bicubic inventing fractional mask values).
- **Training** (`train_UNet_Segmentation.m`): per-target
  `checkpoints/seg_<Target>/` + resume-from-latest, Tversky probability
  guards, synchronized flip augmentation (same geometry to image+mask),
  channel-dim guard (`HxW→HxWx1`), extended `.meta.json`
  (sizes, seed, counts, split level, class/label IDs).
- **Eval** (`evaluateSegmentation.m` + `evaluateSegmentationDataset.m`):
  counts audit (`tp/fp/fn/tn`), macro mean/std + pooled micro, empty-GT
  policy documented (`empty-empty=1`, `empty+FP=0`, `sens/spec/prec=NaN`
  when undefined), held-out runner using **predicted** masks only
  (GT scored after, never fed in), CSV + worst-K overlays
  (`visualizeSegmentationFailures.m`: green TP / red FP / blue FN),
  runtime bench (`benchmarkSegmentationInference.m`) at
  DRIVE/portable/IDRiD orig sizes.
- **DDR merge** (`mergeDDRSegMasks.m`): layout-tolerant `MA|HE→masks_maho`,
  `EX|SE→masks_exudate` (SE-included policy logged + `MERGE_README.txt`).

## 2. Measurements (honest split)

- **Measured here**: `python3 -m pytest tests/ -v` → **17/17 pass**
  (stdlib+Pillow, no MATLAB/numpy/data). Covers the 10×10 hand reference
  (`Dice .5625, IoU .3913…`), empty policies, macro-vs-pooled divergence,
  Tversky direction/guards, stem pairing/orphan/duplicate, flatten,
  range/inversion, seeded leak-free split, config parity grep.
  Two harness failures during development caught and fixed a real
  1-indexed-vs-0-indexed test bug and an over-permissive `0-255` rule.
- **Synthetic only**: all Python numbers are logic checks on synthetic
  masks, not clinical performance. No Dice/IoU on real fundus images.
- **Unverified (no MATLAB here)**: `trainnet` runs, checkpoint resume,
  `pixelLabelDatastore` with `labelIDs [0 255]`, `runSelfTests.m` 4 new
  checks, `evaluateSegmentationDataset`/`benchmark` on real nets, any
  sensitivity/specificity/QWK. Do not quote clinical numbers until a
  MATLAB run with real data produces them via
  `evaluateSegmentationDataset` + CSV.

## 3. Remaining limitations

- No trained `unet_*.mat`, no `data/` — training never executed here.
- `PatientIdFcn` for seg sources is identity (image-level); upgrade if a
  patient/eye key becomes available. `DDR_seg ⊂ DDR` overlap means grader
  fusion on shared IDs is memorized — dedupe or report before Stage-3
  claims cross-population strength.
- `DIARETDB1` confidence markings still need manual inspection; validator
  flags but does not convert them.
- Augmentation is horizontal-flip only (vertical/rotation excluded to
  protect quadrant orientation); class-imbalance beyond Tversky-β
  (sampling/loss schedule) is a future benchmark.
- `512 vs 768` not benchmarked (VRAM unknown); pipeline supports the
  override but defaults to 512.
- `bridge_server.py` CORS/auth, SimEvents paths, VB/IRMA detectors remain
  out of Stage-2 scope.

## 4. Model / data contracts (Stage-3 must respect)

- Files: `unet_Vessels.mat`, `unet_MicroaneurysmsHemorrhages.mat`,
  `unet_Exudates.mat` each containing `net` + `.meta.json` (sizes, seed,
  counts, split level). Class 2 = Foreground everywhere.
- Inference I/O: `runSegmentationNet(net, enhancedGray, roiMask,
  netInputSize=cfg.inputSize)` → `(maskOrig orig-res logical & roiMask,
  maskNet 512 logical)`. Both inputs must come from the **same**
  `assessAndEnhanceImage` call (size-mismatch errors).
- Preprocessing: never train on raw pixels; always
  `preprocessFundusForSegmentation` (enhanced + bilinear + `single/255`).
- Fusion: `lesionMask = maheMask | exudateMask`; to `[224 224]` with
  `nearest` for masks, `bilinear` for `enhancedRGB`; feed **predicted**
  masks only (GT in fusion = leakage, `train_DR_Grader` errors if seg
  nets missing).
- Masks on disk: binary PNG `0/255`; `0/1` auto-converts with warning;
  never mix without conversion (validator errors).
- Reporting: always give macro + pooled + `nEmpty/nEmptyCorrect`; never
  report pooled alone on lesion-rare sets.

## 5. Exactly what Stage-3 should build on

1. Run `datasetRegistry` → `mergeDDRSegMasks` (if DDR) →
   `train_UNet_Segmentation` in MATLAB; confirm 3 `.mat` + `.meta.json`.
2. Run `runSelfTests` (now 12 checks) then
   `evaluateSegmentationDataset(net, valImages, valMasks, 'SaveDir', …)`
   per target + per-dataset + cross-dataset (train STARE/CHASE/HRF, test
   DRIVE; train DDR_seg, test DIARETDB1/IDRiD) before trusting any grade.
3. Inspect `visualizeSegmentationFailures` worst-K (border vs whole-lesion
   misses drive different fixes) and `benchmarkSegmentationInference`
   for the deployment budget.
4. Only then train `train_DR_Grader` (needs predicted masks) →
   `calibrateTemperature` on val (never Messidor-2) → `compareModels`.
5. Keep `segmentationConfig` as the only place resolution/labels change;
   keep this handoff's empty-policy + macro/pooled reporting so judges
   hear measured, synthetic and unverified claims separately.
