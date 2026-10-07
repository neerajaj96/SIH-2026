# STAGE-12 EVALUATION HANDOFF — Real-Data Validation Framework

Scope: `evaluationManifest.m`, `patientBootstrapCI.m`,
`auditEvalLeakage.m`, `runHeldoutEvaluation.m`, `testEvalContracts.m`
(UNEXECUTED), `tests/test_stage12_eval.py`, protocol/handoff/template
docs. No P1–P11 algorithm/config changes; metric engines reused.

## What exists (all DATA-GATED for real numbers)

- Manifest contract (fields, ICDR/split/referable/provenance
  validation, patient-claim gating).
- Leakage auditor (roles, image/patient disjointness, EXACT vs
  POSSIBLE vs UNSUPPORTED-near, calibration/threshold separation,
  preprocessing identity, GT attestation, DDR overlap).
- Patient bootstrap (degeneracy-counted, min-valid policy,
  PATIENT vs IMAGE_LEVEL_BOOTSTRAP labeling, seed reproducibility).
- Orchestrator (role enforcement, FINDINGS-withhold, NOT_MEASURED
  defaults, MEASURED only from real TEST predictions).
- Protocol + report template (this stage).

## Claim states (authoritative vocabulary)

VERIFIED (checked here) / MEASURED (approved real run only) /
NOT MEASURED / DATA-GATED / RUNTIME-GATED. Targets are REQUIREMENTS.

## Tests (this workspace: Python, no MATLAB/data)

- `tests/test_stage12_eval.py`: counts at finalization (contracts only;
  reference QWK 0.88940092 + Wilson values are TEST vectors).
- MATLAB `testEvalContracts.m`: UNEXECUTED.
- Prior suites re-run at finalization.

## Stage-13 prerequisites

Real labeled dataset + `.mat` weights + MATLAB run of the protocol;
per-camera calibration data; adjudication provenance; frozen TEST
split before first metric look.
