# STAGE-5 EXPLAINABILITY HANDOFF — Grad-CAM + Calibration + Disagreement

Scope: `explainGradCAM.m`, `analyzeDisagreement.m`,
`explainabilityConfig.m`, calibration provenance/mismatch handling,
batch agreement fix, additive bridge provenance keys, report
reliability/status blocks. Out of scope (later stages): VB/IRMA
detectors, Grad-CAM architecture changes, calibration fitting redesign,
UI/frontend, DB, SimEvents, treatment/referral logic.

## Capabilities SUPPORTED

- Canonical Grad-CAM (`explainGradCAM`): validated FeatureLayer
  (`""`=auto, recorded; unknown names → UNAVAILABLE/INVALID, never
  substituted), `ReductionLayer="prob"`, original-coordinate resize +
  ROI masking before measurement, deterministic activation-region math
  (highFrac mass; border band + disc zone fractions; lesion overlap),
  reliability `VALID/DEGRADED/UNAVAILABLE/INVALID`, lesion-aware
  disc-dominance, fused-channels causality caveat on every result.
- Structured disagreement (orthogonal): `clinicalEvidenceStatus`,
  `gradeRelationship` (`AGREE/NUMERIC_DISAGREE/NOT_COMPARABLE`),
  `confidenceStatus` (`CALIBRATED/UNCALIBRATED/LOW_CONFIDENCE` — LOW is
  a DATA-GATED review heuristic), `explanationStatus`, `escalate`,
  `reasons`. Missing evidence is never disagreement.
- Calibration: fitting math unchanged (val-only, stable NLL, bounded
  search, 0-indexed consistency); artifacts carry provenance
  (model/dataset tags, class ordering, T, timestamp, version) +
  reliability table (`binLower/binUpper/meanConfidence/accuracy/
  sampleCount`); loaders distinguish `CALIBRATED_VALID /
  CALIBRATED_MISMATCH / UNCALIBRATED_FALLBACK / UNAVAILABLE`;
  fallback T=1.5 is never "calibrated"; hot-reload preserved.
- Batch: `gradesAgree` NaN unless both grades exist; disjoint
  `nInsufficient` triage count; `ruleStatus` column already present.
- Bridge: additive provenance keys via `safe_get` (legacy keys frozen);
  MATLAB-runtime gated, documented UNTESTED end-to-end.
- Reports state landmark/explanation/calibration/evidence status and
  never crash on NaN landmarks (markers skipped, unreliability stated).

## Capabilities UNAVAILABLE (explicit)

- Explanation quality vs annotated lesions: DATA-GATED.
- Calibration ECE/NLL/reliability on real data: DATA-GATED.
- Confirmed feature-layer name: MATLAB-GATED (auto-select + validation
  in place; pin the exact DenseNet name via `net.Layers` when available).
- All MATLAB runtime behavior: MATLAB-GATED.
- VB/IRMA detectors, MA/hemorrhage separation: still UNAVAILABLE (Stage-4).

## Contracts

- Single config: `explainabilityConfig.m` v1 (`gradingConfig` frozen).
- Single Grad-CAM path: pipeline + report consume `explainGradCAM`
  results only. Single disagreement model: `analyzeDisagreement`.
- Statuses: explanation `VALID/DEGRADED/UNAVAILABLE/INVALID`;
  calibration `CALIBRATED_VALID/MISMATCH/UNCALIBRATED_FALLBACK/
  UNAVAILABLE`; disagreement dimensions per struct above.

## Tests (this workspace, Python 3.12, no MATLAB/data/weights)

- `tests/test_stage5_explain.py`: 45/45 PASS (executable above).
- MATLAB `testExplainability.m`: UNEXECUTED here.
- Prior suites must stay green (re-run at finalization).

## Stage-6 prerequisites

MATLAB runs of new tests; `net.Layers` feature-layer confirmation;
labeled lesion/NV/landmark data for explanation validation; val-set
calibration run with reliability review; bridge live-call test; DDR
discipline unchanged.
