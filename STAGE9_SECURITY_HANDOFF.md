# STAGE-9 SECURITY HANDOFF — Threat Model, Boundary & Operations

Scope: `bridge_schema.py` (security constants), `bridge_server.py`
(auth/CORS/limits/IDs/logging/validation), `netrasetu.html` (key
handling, CSP, safe rendering, bounded queue), `tests/test_stage9_security.py`,
this file. No P1–P8 algorithm/config changes; no pentest performed.

## Threat model

- Adversary: network-adjacent client (malicious uploads, XSS probes,
  key guessing/omission, CORS misuse, queue/API abuse, quota attacks)
  and operator error (missing key, open CORS, wrong CWD).
- Trusted (verified, not blind): MATLAB pipeline outputs (converter
  stays defensive: safe_get, NaN→null, type coercion).
- Trust boundaries: browser↔bridge (HTTP + API key), bridge↔engine
  (local process, serialized), browser storage (untrusted device).
- Explicitly NOT covered: TLS termination, user identity/roles,
  log retention/DPL, WAF/rate-limit distribution, pentest.

## Security boundary & fail-closed rules

- `/screen` requires `X-API-Key` == `SIH_API_KEY`; absent key on either
  side → 401; unconfigured server → 503. No anonymous mode exists.
- CORS allowlist from `SIH_CORS_ORIGINS` (default localhost:8000 dev);
  empty allowlist refuses startup; `*` forbidden in code.
- Rate limit 30/min/IP (sliding window, in-memory); direct peer address
  unless `SIH_TRUSTED_PROXY` explicitly names a proxy (first
  X-Forwarded-For entry only).
- Uploads: 15 MB streaming cap (413, never retained), content-type
  prefix, magic-signature gate (JPEG/PNG/GIF/BMP), JPEG-SOF/PNG-IHDR
  dimension parse with `MAX_IMAGE_DIMENSION` 12000px, traversal-proof
  extension allowlist. Well-formed-but-undecodable files stay
  decoder-gated downstream (input failure, never a silent grade).
- Errors: `{detail (allowlisted ≤300 chars), requestId}` envelope; no
  tracebacks/paths/engine internals. Request IDs allowlisted
  `[A-Za-z0-9._-]{1,64}`, minted otherwise; echoed on errors + header.
- Logs: allowlisted fields only (rid/route/status/ms/category/size/
  detail-category); never image bytes, filenames, patient data,
  evidence text, or secrets.

## Deployment configuration (operator checklist)

1. Set `SIH_API_KEY` (long random; never commit) on the bridge host.
2. Set `SIH_CORS_ORIGINS` to the console origin(s); prefer same-origin
   reverse proxy and narrow `connect-src` accordingly (static meta CSP
   covers the localhost default only — replace with headers in prod).
3. Serve the console over TLS (reverse proxy); bridge binds loopback.
4. Verify `/health` then `/health/deep` (models present, project files).
5. Confirm 401 without key, 413 over 15 MB, 429 on burst/second call.
6. Rotate the key on any exposure; revocation = restart with new key.

## Frontend guarantees

- Key in `sessionStorage` only; CSP meta (honest: `unsafe-inline`
  disclosed, reverse-proxy tradeoff documented); every backend string
  escaped; null-rule/insufficient/proxy states preserved; queue bounded
  (150 FIFO, quota-visible, thumbnails only, reviewer keys separate).

## Tests (this workspace: Python, no network/MATLAB/browser/TLS)

- `tests/test_stage9_security.py`: 42/42 PASS (executable above).
- MATLAB `testBridgeContract.m`: UNEXECUTED (engine paths).
- Prior suites re-run at finalization.

## Gates

- MATLAB-GATED: engine lifecycle under load, real struct conversion,
  live `/screen`, failure injection, feature-layer pinning.
- DATA-GATED: all clinical numbers (unchanged).
- SECURITY-GATED: TLS, pentest, audit-log retention, WAF/rate-limit
  distribution, secret management, upload AV scanning.
- DEPLOYMENT-GATED: reverse-proxy behavior, CORS handshake live,
  static hosting + MATLAB-host split (READY-TO-CONFIGURE).
