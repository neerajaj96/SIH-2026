# SIH-2026 — Explainable AI for Diabetic Retinopathy Screening (SIH Problem Statement 26038)

Rural-India DR screening pipeline for Smart India Hackathon 2026 (MathWorks sponsor).
Source files imported from shared Drive folder (40 files, deduplicated).

## What this is

MATLAB-first pipeline + web demo + deployment scaffolding:

- **Quality gate:** `assessAndEnhanceImage.m`, `calibrateQualityThresholds.m`
- **Segmentation:** `train_UNet_Segmentation.m` (3× binary U-Nets: vessels, MA/haemorrhage, exudates), `runSegmentationNet.m`, `preprocessFundusForSegmentation.m`, `evaluateSegmentation.m`, `validateMaskConventions.m`
- **Grading:** `train_DR_Grader.m` (DenseNet-121, 5-channel RGB+vessel+lesion, softmax + hybrid CE + ordinal penalty), `train_Baseline_ResNet50.m`, `compareModels.m`, `computeQWK.m`, `calibrateTemperature.m`
- **Clinical rule second opinion:** `assignClinicalGrade.m` (status-aware ICDR 4-2-1: SUFFICIENT/INSUFFICIENT_EVIDENCE/PROXY, VB/IRMA UNAVAILABLE stubs, merged-MA/HE "4" with speckle guard), `partitionQuadrants.m` (validity-gated), `localizeOpticDiscFovea.m` (+validity/provenance), `detectNeovascularization.m` (screening proxy with usability gates)
- **Inference:** `production_inference.m` (SIMULATED until real `.mat` weights exist), `runScreeningPipeline.m`, `runBatchScreening.m`, `screenOneImage.m`, `saveModelWithMetadata.m`, `getOrLoadCachedModels.m`, `loadModelsIfPresent.m`
- **Data:** `datasetRegistry.m` (single source of truth), `buildGradingDatasets.m`, `buildMessidorTestSet.m`, `buildPatientLevelSplit.m`
- **Telemed / capacity:** `telemedConfig.m` (single versioned config), `buildTelemedModel.m` (SimEvents topology + transmission server + wait logging), `SimEvents_Telemed_Model.m` (demo), `optimizeResourceAllocation.m` (SLA sweep on tested Erlang-C), `erlangCWaitHours.m`, `evaluateTelemedConsistency.m` (analytic-vs-sim, MATLAB-GATED). See `STAGE8_TELEMED_SIMULATION_HANDOFF.md` for the assumption ledger.
- **Demo + deploy:** `netrasetu.html` (LIVE bridge console with explicit DEMO/OFFLINE modes), `bridge_server.py` (FastAPI → MATLAB Engine bridge, serialized screening, versioned API), `production_schema.sql` (Postgres migration target)
- **Docs:** `SIH26038_project_handoff.md`, `LIMITATIONS.md`, `SIH2026_Problem_Statement_26038.pdf`, `SIH26038_Idea_Presentation-1.pptx`

See `SIH26038_project_handoff.md` for architecture decisions and `LIMITATIONS.md` for honest gaps.

## Status — validated vs not (from handoff + self-tests)

Solid (tested via `runSelfTests.m`): quality config + calibrator math, loss functions, QWK, 4-2-1 logic (7/7), OD/fovea sanity check, Erlang-C math, patient-level split guard. Full quality image tests: `testQualitySubsystem.m` (MATLAB, run in MATLAB) + `tests/python/test_quality_mirror.py` (20/20 PASS here) — see `QUALITY_CONTRACT.md` for measured-vs-assumed.
Not yet real: no trained `.mat` weights, no end-to-end MATLAB run with toolboxes, venous beading / IRMA detectors missing, neovascularization is a proxy, `netrasetu.html` defaults LIVE against `bridge_server.py` (explicit DEMO fallback; bridge untested live here), bridge requires `SIH_API_KEY` (fail-closed, no anonymous mode), CORS allowlist + rate limits configured via env (SECURITY-GATED deployment items), SimEvents paths need interactive verification. See `STAGE7_NETRASETU_HANDOFF.md`, `STAGE9_SECURITY_HANDOFF.md`.

## Quickstart

1. MATLAB with Image Processing + Deep Learning Toolboxes (+ DenseNet-121 support package, SimEvents for telemed sim).
2. Fetch data per `datasetRegistry.m`, sort into `data/` (see `validateMaskConventions.m`).
3. `runSelfTests` → `train_UNet_Segmentation` → `train_DR_Grader` → `calibrateTemperature` → `production_inference` (SIMULATED flag flips once `.mat` files exist).
4. Bridge (fail-closed auth — key required, no anonymous mode):
   `pip install fastapi uvicorn python-multipart matlabengine`
   `export SIH_API_KEY='<operator key>' SIH_CORS_ORIGINS='http://localhost:8000' SIH_PROJECT_DIR="$PWD"`
   `python bridge_server.py`, then `GET /health` (liveness) and `GET /health/deep` (`ready` vs `degraded` vs `engine-unavailable` vs `busy`).
   Operator runbook: `docs/RUNBOOK.md` (auth, CORS, rate limits, smoke `scripts/smoke_e2e.py --strict`).
5. Demo: open `netrasetu.html` (LIVE default against the bridge; explicit DEMO toggle for illustrative scenarios; OFFLINE disables screening, never mocks).

Do not commit `data/` or `*.mat` (gitignored). Never commit `.env` files, API keys, weights beyond `.mat` gitignore, patient data, or generated `clinical_rule_eval_*.csv` / `matlab_env_*.mat` / `report_*` evidence (see `.gitignore`).

Status vocabulary (single glossary: `docs/GLOSSARY.md`): VERIFIED = checked here; MEASURED = approved real run only; SIMULATED = placeholder plumbing; UNEXECUTED = written, never run here; DATA-/MATLAB-/DEPLOYMENT-/SECURITY-GATED = blocked on that dependency; NOT_MEASURED = default for unevaluated numbers; ASSUMED/SCENARIO = planning values, never field measurements.
