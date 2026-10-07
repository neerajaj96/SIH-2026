# NetraSetu Runbook — Bridge + Console Operations

## 1. Local development (loopback)

1. Install: `pip install fastapi uvicorn python-multipart` + MATLAB
   Engine for Python (from your MATLAB install).
2. Configure (fail-closed — no anonymous mode exists):
   `export SIH_API_KEY=<long-random-dev-key>`
   optional: `SIH_CORS_ORIGINS=http://localhost:8000`
   `SIH_RATE_LIMIT_PER_MIN=30`, `SIH_PROJECT_DIR=/path/to/repo`.
3. Start: `python bridge_server.py` (listens `0.0.0.0:8000`).
4. Check: `GET /health` (process), then `GET /health/deep`
   (engine + `screenOneImage` on path + `.mat` inventory).
5. Console: open `netrasetu.html` (LIVE default; enter the same API
   key in the key field — session-only, never stored in localStorage).
6. Smoke: `python3 scripts/smoke_e2e.py --bridge http://localhost:8000
   --key <dev-key>` (transport/schema now; clinical path SKIPs without
   MATLAB + assets + fixture).

## 2. Recommended production shape

Browser → same-origin reverse proxy (TLS here) → loopback bridge.
Proxy injects `X-API-Key` server-side where applicable so browsers
never hold the long-lived secret; set `SIH_CORS_ORIGINS` to the proxy
origin only; `X-Forwarded-For` honored ONLY with `SIH_TRUSTED_PROXY`
set. Static console may be served anywhere; the MATLAB bridge needs a
MATLAB host — serverless/static hosting can NEVER run the Engine.

## 3. Operations

- Logs: `evt=... rid=... route=... status=... ms=...` (no image/PII).
  Correlate with client `requestId` / `X-Request-ID` header.
- Auth failures: 401 (wrong/missing key), 503 (key unconfigured).
- Busy: 429 = engine occupied, retry shortly (single-flight design).
- Oversized: 413 over 15 MB. Malformed: 400 (magic/dimension gates).
- Engine crash: invalidated + one retry, then 503 (never synthetic).
- Restart: re-running `bridge_server.py` is stateless (persistent
  MATLAB cache lives per-process only).
- Queue: browser-local fallback capped at 150 FIFO; Claude-`db` path
  unchanged where available.

## 4. Troubleshooting matrix

| Symptom | Likely cause | Check |
|---|---|---|
| OFFLINE badge, health fails | bridge not running / wrong URL | process, `bridgeUrl` field, firewall |
| 401 on /screen | key mismatch/absent | `SIH_API_KEY` vs key field |
| 503 unconfigured | no `SIH_API_KEY` in env | export key, restart bridge |
| 503 engine-unavailable | MATLAB missing/crashed | `/health/deep`, MATLAB install |
| `degraded` deep-health | missing `.mat` (SIMULATED) | asset inventory in response |
| 429 busy | concurrent screening | retry; single-flight by design |
| 413 | >15 MB upload | downscale client-side |
| Empty queue, cases submitted | Claude-`db` absent + quota full | `quota-unavailable` banner |
| CORS error in console | origin not allowlisted / file:// | proxy or `SIH_CORS_ORIGINS` |

## 5. Hard limits of this document

TLS, secret rotation, log retention, WAF, pentest, clinical accuracy,
MATLAB runtime numbers: DEPLOYMENT-/SECURITY-/DATA-GATED (see
STAGE10/STAGE11 handoffs). Nothing here certifies production readiness.

## 6. MATLAB prerequisites (Stage-11 runtime validation)

| Requirement | Detail | Status here |
|---|---|---|
| MATLAB release | R2022b or newer (dlnetwork, gradCAM) | UNEXECUTED |
| Image Processing Toolbox | ROI/morphology/CLAHE/resize | UNEXECUTED |
| Deep Learning Toolbox | dlnetwork, predict, gradCAM | UNEXECUTED |
| DenseNet-121 support package | `imagePretrainedNetwork("densenet121")` | UNEXECUTED |
| Statistics & ML Toolbox | `tinv` (consistency CI), tests | UNEXECUTED |
| Report Generator | optional (PDF path; txt fallback otherwise) | UNEXECUTED |
| Simulink + SimEvents | Stage-8 model only | UNEXECUTED |
| MATLAB Engine for Python | version-matched to MATLAB + Python | UNEXECUTED |
| Model assets | `unet_*.mat`, `trained_dr_grader.mat` in bridge CWD | absent |
| Calibration assets | `calibrated_temperature.mat`, `qualityThresholds.mat` | absent |
| Fixtures | synthetic fundus PNG + corrupt/empty files | absent |

Run order on a capable host: `matlab_env_report` → `testStage11Runtime`
→ `benchmarkStage11` → `scripts/smoke_e2e.py --bridge <url> --key <key>
--strict`. Record every figure in the Stage-11 handoff evidence table.
