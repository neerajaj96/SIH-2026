# Stage-1 → Stage-2 Stable Contract: Image Quality + Enhancement

Consumer: Stage 2 (segmentation training/inference via `preprocessFundusForSegmentation.m` → `runSegmentationNet.m`, and DR grader fusion via `train_DR_Grader.m` / `runScreeningPipeline.m`).

## Canonical entry points

- New code: `rep = assessFundusQuality(img)` or `assessFundusQuality(img,'Config',qualityConfig(),'FocusThresh',[],'EntropyThresh',[])`.
- Legacy (frozen): `[isGradeable,enhancedRGB,enhancedGray,focusScore,entropyScore,roiMask] = assessAndEnhanceImage(img,focusThresh,entropyThresh)`. Same shapes/types as before; defaults from `qualityConfig`.
- Config: `cfg = qualityConfig()` (v1.1.0-stage1-config). Never hardcode `8/3.5` — read `cfg.focusThresh/cfg.entropyThresh`. Calibrated override: `[ft,et,info] = qualityLoadCalibration(cfg,dir)` reads `qualityThresholds.mat` when present.
- Calibrate: `[ft,et] = calibrateQualityThresholds(goodDir,badDir,'OutputMat',path)` writes `.mat + .meta.json`.

## Report struct (`assessFundusQuality`)

- `decision`: `'PASS'|'BORDERLINE'|'FAIL'`; `isGradeable = decision ~= "FAIL"`.
- `reasons{}`: machine-readable strings per failed/marginal check; `recaptureGuidance{}`: clinical re-shoot instructions.
- `focusScore, entropyScore, focusThresh, entropyThresh`.
- `scores`: `illuminationMedian/P5/P95, contrastP95P5, underexposedFrac, overexposedFrac, specularFrac, coverageFrac, circularity, roiDiameterPx, focusScore, entropyScore`.
- `enhancedRGB (HxWx3 uint8, background=0)`, `enhancedGray (HxW uint8, background=0)`, `roiMask (HxW logical)`, `roiInfo(coverageFrac,circularity,roiDiameterPx,numPixels,framePixels)`.
- `configVersion, isCalibrated, calibrationSource, elapsedCpuSec`. Deterministic (no rng).

## Decision policy

- `FAIL` if `focus<ft` or `entropy<et` or `coverage<0.05` or `underexp>0.25` or `overexp>0.15`. Recapture required.
- `BORDERLINE` if above thresholds but within `15%` (focus) / `10%` (entropy) margins, or mild flags (specular>0.02, contrast<0.25, circularity<0.55). Gradeable-with-warning: proceed but log `reasons`; curators review training inclusion.
- `PASS` otherwise.

## Stage-2 usage rules

1. Train/inference MUST use `enhancedGray` (not raw) via `preprocessFundusForSegmentation` (bilinear image, nearest mask to `[512 512]`, `/255 single`). Fusion uses `enhancedRGB` resized `[224 224]` + masks.
2. `roiMask` from the SAME quality call as the enhanced image (enforced by `runSegmentationNet:roiMaskSizeMismatch`).
3. Quality curation at file-list stage (`calibrateQualityThresholds` + `BORDERLINE` review), never silent per-call filtering (`-Inf` calls enhance-only by design).
4. Enhancement guards: flat ROI contrast `<0.12` halves CLAHE to `0.005`; ROI diameter `<256px` skips `imnlmfilt`. Normal images identical to legacy path.

## Measured here vs assumptions

- Measured (Python stdlib mirror, 20 checks, 0.10s): ROI coverage/circularity, sharp>blur focus ordering, textured>flat entropy ordering, PASS/BORDERLINE/FAIL taxonomy, midpoint/sens/spec/AUC=1 on separated vectors, determinism, all-black/white/tiny/speck robustness.
- Assumed (no MATLAB/Octave in workspace): MATLAB `testQualitySubsystem.m` (8 groups) + 2 new `runSelfTests` numeric checks NOT yet executed — run `runSelfTests; testQualitySubsystem` in MATLAB before trusting numbers. No real-camera/dataset validation; thresholds remain placeholders until `calibrateQualityThresholds` runs on labeled gradeable/ungradeable sets per camera.
