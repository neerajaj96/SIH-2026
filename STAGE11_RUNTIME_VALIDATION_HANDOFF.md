# STAGE-11 RUNTIME VALIDATION HANDOFF — Real-MATLAB Verification

Scope: `scripts/matlab_env_report.m`, `scripts/benchmarkStage11.m`,
`testStage11Runtime.m` (UNEXECUTED), `tests/test_stage11_runtime.py`,
this file + runbook MATLAB table. No P1–P11 algorithm/config changes.

## Verification matrix (status at this workspace: UNEXECUTED throughout)

| # | Check | Requires | Status here |
|---|---|---|---|
| A | MATLAB env detected (`ver`, toolboxes, `pyenv`, assets) | MATLAB | UNEXECUTED |
| B | Simulated pipeline (gate/insufficient/malformed) | MATLAB | UNEXECUTED |
| C | Trained pipeline (assets → DL/fusion/temp/Grad-CAM/clinical) | MATLAB + `.mat` | UNEXECUTED |
| D | Engine conversion MATLAB→JSON | Engine | UNEXECUTED |
| E | Live HTTP `/screen` | serving bridge | UNEXECUTED |
| F | Repeat/reuse isolation | MATLAB | UNEXECUTED |
| G | Failure/recovery mapping | MATLAB | UNEXECUTED |
| H | Cold/warm timing + memory | MATLAB | UNEXECUTED |

Simulated checks prove plumbing/schema only — never trained inference,
never clinical accuracy. Trained checks additionally require
`unet_*.mat` + `trained_dr_grader.mat`; without them they record
UNEXECUTED (the runner cannot PASS them).

## Failure matrix (inference → HTTP/UI mapping)

| Failure | Internal state | HTTP | Frontend |
|---|---|---|---|
| quality FAIL | `ungradeable` | 200 + decision | OFFLINE? no — FAIL panel, recapture guidance |
| missing models | SIMULATED | 200, `dl:null`, `modelPresent:false` | unavailable DL panel |
| corrupt model | error | 500 safe detail | failure panel, nothing graded |
| malformed image | 400 pre-decode | 400 | client error text |
| oversized | 413 pre-retain | 413 | client error text |
| bad key / no key | 401 / 503 | 401 / 503 | key prompt / OFFLINE |
| engine busy | 429 | 429 | retry prompt |
| engine dead | 503 after 1 retry | 503 | OFFLINE + reference ID |
| temp/artifact unwritable | 500 safe detail | 500 | failure panel |
| NaN rule grade | INSUFFICIENT_EVIDENCE | 200, `rule:null` | "not Grade 0" panel |

## Environment to record (fill from `matlab_env_report` output)

MATLAB version / toolboxes / Python+Engine versions / platform /
CPU-GPU / `.mat` names + variables / calibration artifact state /
fixture IDs / cold-warm timings / memory before-after / failures.
Nothing below may be filled without execution.

## Tests (this workspace: Python, no MATLAB/assets/Engine/deployment)

- `tests/test_stage11_runtime.py`: counts at finalization (harness
  contracts only).
- MATLAB `testStage11Runtime.m` + `scripts/benchmarkStage11.m` +
  `scripts/matlab_env_report.m`: UNEXECUTED.
- Prior suites re-run at finalization.

## Gates

- MATLAB-GATED: every row above. DATA-GATED: all clinical numbers.
- DEPLOYMENT-GATED: TLS/proxy/hosting. SECURITY-GATED: pentest/audit.
