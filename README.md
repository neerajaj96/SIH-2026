# SIH-2026 — Explainable AI for Diabetic Retinopathy Screening (SIH Problem Statement 26038)

Rural-India DR screening pipeline for Smart India Hackathon 2026 (MathWorks sponsor).
Source files imported from shared Drive folder (40 files, deduplicated).

## What this is

MATLAB-first pipeline + web demo + deployment scaffolding:

- **Quality gate:** `assessAndEnhanceImage.m`, `calibrateQualityThresholds.m`
- **Segmentation:** `train_UNet_Segmentation.m` (3× binary U-Nets: vessels, MA/haemorrhage, exudates), `runSegmentationNet.m`, `preprocessFundusForSegmentation.m`, `evaluateSegmentation.m`, `validateMaskConventions.m`
- **Grading:** `train_DR_Grader.m` (DenseNet-121, 5-channel RGB+vessel+lesion, softmax + hybrid CE + ordinal penalty), `train_Baseline_ResNet50.m`, `compareModels.m`, `computeQWK.m`, `calibrateTemperature.m`
- **Clinical rule second opinion:** `assignClinicalGrade.m` (ICDR 4-2-1), `partitionQuadrants.m`, `localizeOpticDiscFovea.m`, `detectNeovascularization.m` (screening proxy only)
- **Inference:** `production_inference.m` (SIMULATED until real `.mat` weights exist), `runScreeningPipeline.m`, `runBatchScreening.m`, `screenOneImage-1.m`, `saveModelWithMetadata.m`, `getOrLoadCachedModels.m`, `loadModelsIfPresent.m`
- **Data:** `datasetRegistry.m` (single source of truth), `buildGradingDatasets.m`, `buildMessidorTestSet.m`, `buildPatientLevelSplit.m`
- **Telemed / capacity:** `buildTelemedModel.m`, `SimEvents_Telemed_Model.m`, `optimizeResourceAllocation.m`, `erlangCWaitHours.m`
- **Demo + deploy:** `netrasetu.html` (mock-data console), `bridge_server.py` (FastAPI → MATLAB Engine bridge), `production_schema.sql` (Postgres migration target)
- **Docs:** `SIH26038_project_handoff.md`, `LIMITATIONS.md`, `SIH2026_Problem_Statement_26038.pdf`, `SIH26038_Idea_Presentation-1.pptx`

See `SIH26038_project_handoff.md` for architecture decisions and `LIMITATIONS.md` for honest gaps.

## Status — validated vs not (from handoff + self-tests)

Solid (tested via `runSelfTests.m`): quality gate, enhancement, loss functions, QWK, 4-2-1 logic (7/7), OD/fovea sanity check, Erlang-C math, patient-level split guard.
Not yet real: no trained `.mat` weights, no end-to-end MATLAB run with toolboxes, venous beading / IRMA detectors missing, neovascularization is a proxy, `netrasetu.html` on mock data (not wired to `bridge_server.py`), bridge has open CORS + no auth, SimEvents paths need interactive verification.

## Quickstart

1. MATLAB with Image Processing + Deep Learning Toolboxes (+ DenseNet-121 support package, SimEvents for telemed sim).
2. Fetch data per `datasetRegistry.m`, sort into `data/` (see `validateMaskConventions.m`).
3. `runSelfTests` → `train_UNet_Segmentation` → `train_DR_Grader` → `calibrateTemperature` → `production_inference` (SIMULATED flag flips once `.mat` files exist).
4. Demo: open `netrasetu.html`; prod bridge: `pip install fastapi uvicorn python-multipart matlabengine` then `python bridge_server.py`.

Do not commit `data/` or `*.mat` (gitignored).
