# STAGE-4 CLINICAL HANDOFF — Evidence Engine (Conservative)

Scope: optic-disc/fovea localization + validity, quadrant validity,
status-aware 4-2-1 rule engine, NV screening proxy, MA/HE speckle guard,
pipeline integration, rule-grade eval runner. Out of scope (later
stages): VB/IRMA detectors, Grad-CAM, calibration/segmentation/grader
redesign, bridge/UI/DB/Simulink, treatment/referral logic.

## Capabilities SUPPORTED (this release)

- Evidence states everywhere: `VERIFIED / PROXY / NOT_DETECTED /
  UNAVAILABLE / INVALID`; rule status `SUFFICIENT / INSUFFICIENT_EVIDENCE /
  PROXY / INVALID`. Missing detectors yield UNAVAILABLE, never zero.
- 4-2-1 truth-table logic VERIFIED on synthetic logic tests (MATLAB
  `runSelfTests` ladder + insufficient case; UNEXECUTED here).
- Hemorrhage counting on the MERGED MA/HE channel with canonical
  speckle guard (`filterLesionComponents`, ≥6px, conn 8, both paths);
  0/10/21 single-pixel speckles per quadrant cannot fire Severe.
- NV screening proxy with usability gates (sparse mask / NaN geometry /
  <3 segments → INVALID), exposed features + reliability, config
  thresholds, PROXY wording in every evidence string.
- Landmark validity (`CONFIDENT/FALLBACK/UNRELIABLE`), explicit OD-radius
  and fovea fallbacks, NaN-safe; quadrant `VALID/FALLBACK/INVALID`
  (zeros mask + blocked reasoning when invalid; fallback needs explicit
  config permission, default denied).
- Pipeline exposes DL grade + rule grade/status/trigger + landmark
  validity + NV status + quality decision side by side; reports and
  batch CSV render INSUFFICIENT_EVIDENCE explicitly (NaN-safe).
- `auditQualityBatch` (file-list curation) + `evaluateClinicalRule`
  (held-out rule-vs-label runner; reports exclusions, no fabricated
  metrics without data).

## Capabilities UNAVAILABLE (explicit, not silent)

- VB detection: UNAVAILABLE (`assessVenousBeading` = extension stub).
- IRMA detection: UNAVAILABLE (`assessIRMA` = extension stub).
- Vitreous hemorrhage detection: UNAVAILABLE/manual input only.
- Definitive Level 0/1/2 rule grades: impossible until the above are
  assessable (engine returns INSUFFICIENT_EVIDENCE + NaN).
- NV diagnosis: PROXY screening flag only.
- MA-vs-hemorrhage separation: merged channel only.
- Landmark/NV/VB/IRMA accuracy numbers: DATA-GATED.
- All MATLAB behavior: MATLAB-GATED (no runtime here).

## Contracts

- `assignClinicalGrade` keeps `[grade, evidence]` + additive `ruleReport`
  (`status/grade/evidence/trigger/provenance{source=MA_HE_COMBINED,
  maskSource=trained|placeholder|unspecified,...}/evidenceStatus`);
  `localizeOpticDiscFovea` keeps 3 outputs + additive `lmStatus`;
  `partitionQuadrants` keeps mask + additive `qValid`;
  `detectNeovascularization` keeps 3 outputs + additive `nvReport`.
- Single config: `clinicalConfig.m` v1 (`connectivity 8`, thresholds,
  bands, fallback policy, status vocabulary).
- Training audit and rule eval write timestamped CSV manifests.

## Tests (this workspace, Python 3.12, no MATLAB/data/weights)

- `tests/test_stage4_clinical.py`: 64/64 PASS (executable above).
- MATLAB `testClinicalReasoning.m` incl. end-to-end pipeline propagation
  test (UNEXECUTED here); `evaluateClinicalRule` reports Wilson CIs via
  `wilsonScoreInterval` (no bare point estimates).
- Prior suites must stay green (re-run at finalization).

## Stage-5 prerequisites

MATLAB runs of all new tests; labeled VB/IRMA/NV/landmark data for
threshold validation; DDR-overlap discipline; bridge wiring of new
quality/clinical keys with a live test.
