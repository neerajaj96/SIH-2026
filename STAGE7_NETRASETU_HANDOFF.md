# STAGE-7 NETRASETU HANDOFF — Live Integration / Web + MATLAB Bridge

Scope: `bridge_schema.py`, `bridge_server.py` (hardening only),
`screenOneImage.m` additive fields, `netrasetu.html` LIVE/OFFLINE/DEMO +
renderer + bounded queue fallback, `tests/test_stage7_bridge.py`,
`testBridgeContract.m` (UNEXECUTED). No P1–P6 algorithm/config changes;
no frontend redesign; no auth/CORS architecture; no DB.

## API contract (`apiVersion 1.0-stage7`)

Legacy flat keys frozen (`status` … `gradCamOnDisc`). Additive nullable
groups: `quality{decision,reasons,guidance,calibrated}`,
`clinical{ruleStatus}`, `explanation{status}`,
`disagreement{clinicalEvidenceStatus,gradeRelationship,confidenceStatus,
explanationStatus,escalate,reasons,configVersion}`,
`temperature{value,state}`, `landmarkStatus{od,fovea,quadrant}`,
`modelPresent`. NaN→null always; missing→null (never KeyError); rule
NaN stays null (never Grade 0). Canonical semantics live in MATLAB;
flat legacy keys are compatibility aliases, computed once.

## Concurrency / health / observability

One dedicated worker thread + `threading.Lock` (serialized screening;
second concurrent request → HTTP 429 immediately; event loop never
blocked). Engine crash → invalidate + one fresh-engine retry → HTTP 503
(no synthetic results, dead engine never reused, lock released in
`finally`). Uploads streamed with 15 MB cap → HTTP 413 (never retained).
Logs: start/end/duration/status only (no image/PII/filenames).
`/health` = process + version; `/health/deep` = engine + project files
+ model-asset inventory (names only).

## Frontend modes

LIVE default (`http://localhost:8000`, configurable + persisted,
never a production hostname): probes `/health`, stays LIVE or shows
OFFLINE (no screening, no mock). DEMO is explicit-toggle only with
badge; SCENARIOS unreachable from LIVE (pinned). Renderer is null-safe,
escapes all backend text, preserves every uncertainty state, never
grades in JS. Queue: Claude-`db` when present, else bounded (150, FIFO,
quota-eviction with visible `quota-unavailable` state) browser-local
storage of backend-shaped records (thumbnail only, reviewer fields
separate). Upload guards: type + 15 MB client-side.

## Deployment

- static frontend: READY-TO-CONFIGURE (any static host; not exercised).
- MATLAB bridge: MATLAB-HOST REQUIRED (MATLAB + Engine + project +
  `.mat` assets; not exercised here).
- live deployment: NOT VERIFIED. Never claim Vercel/Netlify can run
  the MATLAB Engine.

## Tests (this workspace: Python, no MATLAB/Engine/browser)

- `tests/test_stage7_bridge.py`: 38/38 PASS (mapping via stubbed
  fastapi/matlab, schema, NaN/null, modes, escaping, queue bounds).
- MATLAB `testBridgeContract.m`: UNEXECUTED (incl. 10-step live-engine
  checklist). Prior suites re-run at finalization.

## Gated items

- MATLAB-GATED: engine lifecycle/concurrency under load, struct
  conversion on real engine, `/health/deep`, live `/screen` ×10,
  failure injection, feature-layer confirmation.
- DATA-GATED: all clinical numbers (unchanged).
- SECURITY-GATED (Stage-7 scope; SUPERSEDED by Stage-9 implementation):
  fail-closed shared-key auth (`hmac.compare_digest`, 401/503, no bypass),
  explicit CORS allowlist (`SIH_CORS_ORIGINS`, `*` refused at startup),
  sliding-window rate limiting, streaming upload validation. See
  `STAGE9_SECURITY_HANDOFF.md` and `bridge_server.py`.
