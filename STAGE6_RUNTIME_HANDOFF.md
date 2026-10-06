# STAGE-6 RUNTIME HANDOFF — Inference Reliability & Production Hardening

Scope: end-to-end runtime correctness, failure tolerance, batch/report
robustness, collision-proof artifacts, CWD candor, observability. No
P1–P5 algorithm/config changes (only the batch isolation + CWD notice +
artifact naming + report-status surface).

## What changed (this stage)

- `runBatchScreening.m`: per-file `try/catch` → schema-preserving ERROR
  rows (`status=error`, `qualityDecision=ERROR`, `ruleStatus=INVALID`);
  batch always continues; counts exclude ERROR rows appropriately
  (`==1`/`==0` sums skip NaN); simulated warning names `pwd`.
- `uniqueArtifactPath.m` (new): `stem_timestamp` + deterministic `_v2…`
  collision loop (no millisecond reliance); wired into batch CSV, audit
  CSV, eval manifest, report PNGs, PDF/txt reports.
- Reports already carried status blocks + NaN-safe markers (Stage-5);
  no computation changes here.

## Contracts (unchanged, pinned)

- Single core `runScreeningPipeline`; single Grad-CAM/disagreement/
  fusion/temperature paths (pinned by `test_stage6_runtime.py`).
- `r` exposes every stage state (quality/seg/landmark/clinical/DL/
  calibration/explanation/disagreement); batch CSV + manifests mirror it.
- NaN means unevaluable everywhere; never coerced to 0/disagreement.

## Benchmark methodology (MATLAB-GATED — run when MATLAB exists)

- Cold start: fresh session, `tic; getOrLoadCachedModels(); toc` (loads
  4 nets + temperature validation).
- Warm inference: `runScreeningPipeline` ×N on one image (`cputime`
  per stage via `r.elapsedCpuSec`-style probes if added later).
- Batch: 100-image folder incl. 1 corrupt + 1 ungradeable; record
  wall time, per-row status distribution, CSV correctness.
- Memory: `memory` (CPU) / `gpuDevice` (GPU) before/after; figure-handle
  check (`findall(0,'type','figure')` empty after runs).
- Environment to record: MATLAB release, toolboxes, CPU/GPU, image
  sizes, weight presence. No numbers exist yet — do not invent any.

## Tests (this workspace, Python 3.12, no MATLAB/data/weights)

- `tests/test_stage6_runtime.py`: PASS count at finalization.
- MATLAB `testStage6Runtime.m`: 16 areas, UNEXECUTED here.
- Prior suites must stay green.

## Known residual risks

- `Report(path.pdf,'pdf')` with extension-bearing unique path: accepted
  per `mlreportgen` filename rules but UNVERIFIED without MATLAB.
- Timestamp readability vs clock skew: collision loop is authoritative,
  timestamp cosmetic.
- Engine/CWD launches must `cd` to project root (documented in batch
  warning; bridge behavior unchanged).
- All MATLAB runtime + timing/memory + bridge live-call: MATLAB-GATED.
- All clinical numbers: DATA-GATED as before.
