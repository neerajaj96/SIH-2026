# STAGE-8 TELEMED HANDOFF — Workflow, SimEvents & Resource Optimization

Scope: `telemedConfig.m`, `telemedTransmissionSeconds.m`,
`buildTelemedModel.m` (config + transmission server + logging),
`optimizeResourceAllocation.m` (single Erlang-C + surfacing),
`evaluateTelemedConsistency.m`, `tests/test_stage8_telemed.py`,
`testTelemedSim.m` (UNEXECUTED). No P1–P7 algorithm/config changes.

## System model (authoritative flow)

arrival (exponential gaps) → Bandwidth_Queue (cap 100) →
Transmission_Server (derived payload/bandwidth/overhead delay) →
AI_Pipeline_Server (fixed 2.5s ASSUMED) → Clinical_Triage
(severity-mix switch) → Cleared_NonReferable terminator |
Ophthalmologist_Review (normal, mean 30s ASSUMED) → Reviewed_Referable.
Wait monitors (Scopes for demo) + To-Workspace logs (machine-readable)
on both queues. Severity mix is an ARDA-sourced SCENARIO, never field data.

## Contracts

- One Erlang-C: `erlangCWaitHours.m` (local duplicate deleted; optimizer
  calls it; `rho>=1 → Inf` everywhere).
- One config: `telemedConfig.m` v1 (demand, ARDA mix + provenance,
  transmission, AI/review service, queues, sim control, tolerance).
- One transmission math: `telemedTransmissionSeconds.m`
  (`MB*8*(1+overhead)/Mbps`, explicit units, validated inputs).
- Optimizer objective (explicit): minimize reviewers s.t. Erlang-C wait
  ≤ SLA AND utilization < 1; per-candidate utilization + stability
  reported; infeasible stays NaN with a triage-selectivity note.
- Consistency: analytic vs seeded-replication sim mean ± t-CI,
  warm-up excluded, tolerance verdict; unstable analytic ⇒ comparison
  meaningless (stated, not fudged).
- Output contract per plan §M (scenario/parameters/rates/servers/
  utilization/metrics/waits/throughput/stability/bottleneck/optimizer/
  metadata/comparison/assumptionStatus).

## Tests (this workspace: Python, no MATLAB/SimEvents/data)

- `tests/test_stage8_telemed.py`: 55/55 PASS (executable above).
- MATLAB `testTelemedSim.m`: UNEXECUTED here.
- Prior suites re-run at finalization.

## Assumption ledger

- VERIFIED: Erlang-C math (M/M/1 closed form, here + runSelfTests);
  duplicate removed; mix sums to 1; conversions explicit.
- ASSUMED: image 2MB, bandwidth 4Mbps, overhead 10%, AI 2.5s/1 worker,
  review 30s/120s scenarios, queue cap 100, warm-up 300min, 10 reps,
  tolerance 35%.
- SCENARIO: demand (100k/2000h/30min SLA), ARDA severity mix, camp pace,
  surge horizon, 8 canned scenarios (documented in handoff, not measured).
- DATA-GATED: arrival rates, service times, bandwidth, payloads,
  prevalence, reviewer capacity, every operational number.
- MATLAB-GATED: SimEvents build/run, consistency numbers, timing/memory,
  seed reproducibility on engine, block-path CAUTION verification.

## Stage-9 prerequisites

MATLAB+SimEvents run of `testTelemedSim` + `evaluateTelemedConsistency`
(record methodology/env); field-measure arrival/service/bandwidth
payloads; replace ASSUMED values with MEASURED + provenance; re-tune
tolerance on real variability; live-demo block-path check.
