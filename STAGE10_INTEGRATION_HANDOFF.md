# STAGE-10 INTEGRATION HANDOFF — E2E, Runtime & Deployment Verification

Scope: `scripts/smoke_e2e.py`, `tests/test_stage10_e2e.py`,
`testE2ESmoke.m` (UNEXECUTED), `.github/workflows/python-contracts.yml`,
`docs/RUNBOOK.md`, this file. No P1–P9 algorithm/config changes.

## Verification levels (evidence required for PASS)

1. UNIT/CONTRACT — stdlib suites green in this env (see below).
2. LOCAL INTEGRATION — cross-suite consensus by file-text + stub
   execution (struct keys MATLAB↔bridge↔frontend identical).
3. MATLAB RUNTIME — requires MATLAB run (UNEXECUTED).
4. LIVE HTTP — requires bridge+Engine+assets (smoke runner; SKIPs here).
5. DEPLOYMENT — requires exercised topology (none exercised).

## Evidence matrix (claim → evidence → status)

| Claim | Required evidence | Status |
|---|---|---|
| Screening math correct | P1–P9 suites | VERIFIED (below) |
| Contracts agree across stages | stub + static consensus | VERIFIED |
| MATLAB screening works | Engine execution | MATLAB-GATED |
| LIVE frontend works | browser+HTTP run | UNEXECUTED here |
| Deployment works | exercised topology | DEPLOYMENT-GATED |
| TLS/reverse-proxy correct | HTTPS endpoint check | DEPLOYMENT-GATED |
| Memory stable | repeated-run measurement | MATLAB-GATED |
| Clinical numbers | labeled data | DATA-GATED |

## Deployment topology (supported / impossible)

- SUPPORTED: static file (any host/`file://` with CORS caveat) +
  loopback MATLAB bridge; same-origin reverse-proxy split (recommended).
- IMPOSSIBLE: serverless/static-only MATLAB Engine execution.
- Secrets via env on the host; browsers hold only session keys (or none
  behind an injecting proxy); TLS at proxy; `/health` vs `/health/deep`
  semantics in runbook; temp/artifacts local; no resource limits set
  here (operator concern, documented).

## Adversarial scenarios (designed; live ones gated)

Schema drift, CORS-correct-but-blocked (`file://` null origin),
auth accepted-but-unconfigured confusion (fail-closed covers),
CWD-dependent engine start, silent SIMULATED serving (now loudly
warned with `pwd`), stale state across requests (single-flight +
no cross-request MATLAB state besides model cache), non-serializable
values (converter defaults), frontend null-crash (null-safe renderer),
restart handle/file leaks (finally-cleanup + atexit), tmp leaks,
healthy-health-but-useless-screening (deep `degraded` state).

## Tests (this workspace: Python, no MATLAB/browser/network/deployment)

- `tests/test_stage10_e2e.py`: counts at finalization (incl. live dead-
  port run proving honest SKIP behavior).
- `scripts/smoke_e2e.py --bridge http://127.0.0.1:9`: exits 0, SKIPs.
- MATLAB `testE2ESmoke.m`: UNEXECUTED.
- Prior suites re-run at finalization.

## Known limitations

- `__pycache__/` ignored + untracked (verified, nothing committed).
- README quickstart predates auth until runbook backlink lands.
- No CI existed; new workflow runs stdlib suites as gating, with
  explicitly `if: false` UNEXECUTED placeholders (never fake-green).
