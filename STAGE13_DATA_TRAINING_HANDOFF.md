# STAGE-13 DATA/TRAINING HANDOFF — Acquisition, Provenance & Freezing

Scope: `datasetRegistry.m` (structured fields), `acceptDataset.m`,
`freezeCandidate.m`, `tests/test_stage13_data.py`,
`testDataContracts.m` (UNEXECUTED), runbook acquisition section, this
file. No P1–P12 algorithm/config changes; no training executed here.

## Lifecycle (enforced order)

TEST_SPLIT_FROZEN (manifest + test hash recorded FIRST)
→ TRAIN/VAL development + model selection (never touching TEST)
→ PRE_TRAIN_CANDIDATE (manifest/split/config/code identity, no timestamps)
→ training → FINALIZED_CANDIDATE (+ checkpoint, VAL calibration with
matching provenance) → Stage 14 evaluates the untouched TEST cohort.

## Candidate identity

Content-hash over immutable inputs only (timestamps excluded —
identical inputs at different times yield identical IDs). git SHA
best-effort (`unavailable` when no `.git`, never fabricated) plus an
independent content hash. Calibration mismatch (checkpoint/split/
classes) INVALIDATES; never silently attach fresh artifacts. Test
changes force a new evaluation version, never a rewrite.

## Provenance record (unknowns explicit, never fabricated)

candidateId/state, gitCommitSha, contentHash, manifest/train/val/test
hashes, preprocessing + model config versions, checkpoint ID/hash,
calibration artifact ID/hash + dataset/split, MATLAB/toolboxes/HW,
seed, createdAt/finalizedAt (metadata only).

## Assumption ledger

- VERIFIED: registry structure, acceptance verdicts, freeze state
  machine, content-hash stability, calibration-mismatch rejection,
  preprocessing parity (train calls canonical functions).
- ASSUMED: nothing new (defaults documented in telemed/quality configs).
- SCENARIO: ARDA severity mix (never measured prevalence).
- DATA-GATED: every download, label, split count, weight, T value,
  timing, and clinical number. MATLAB-GATED: all `.m` execution.

## Stage-14 prerequisites

Real dataset(s) accepted via `acceptDataset`; TEST frozen via
`freezeCandidate('PRE_TRAIN',…)` BEFORE any selection iteration;
training per existing seeded scripts; VAL-only calibration;
`FINALIZE` with matching provenance; then untouched-TEST evaluation.
