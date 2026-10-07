# Glossary — single authoritative status vocabulary

Every handoff, contract, report, and UI string uses these terms with exactly
these meanings. If a document needs a new term, add it here first.

| Term | Meaning | Example |
|---|---|---|
| `VERIFIED` | Checked here (logic/math/synthetic contract tests) | QWK mirror `0.88940092` as a TEST VECTOR |
| `MEASURED` | Approved real-data run only | held-out TEST metrics from `runHeldoutEvaluation` on real weights |
| `NOT_MEASURED` / `null` | Default for every unevaluated number | all `runHeldoutEvaluation` endpoints before a real run |
| `TEST/LOGIC` | Synthetic fixture, never project evidence | Python mirror values, MATLAB contract fixtures |
| `SIMULATED` | Placeholder plumbing without trained weights | `production_inference` SIMULATED banner |
| `UNEXECUTED` | Written but never run in this workspace | all MATLAB `test*.m` suites here (no MATLAB) |
| `MATLAB-GATED` | Blocked on a MATLAB execution | engine/consistency/train paths |
| `DATA-GATED` | Blocked on real data/field measurement | downloads, labels, prevalence, timings |
| `DEPLOYMENT-GATED` | Blocked on a staging/prod topology | TLS, hosting, live bridge verification |
| `SECURITY-GATED` | Blocked on a security review/hardening step | pentest (never claimed), rotation audit |
| `ASSUMED` | Engineering placeholder, explicitly provisional | `imageMB=2.0`, `AI=2.5s` |
| `SCENARIO` | Illustrative program-planning value | ARDA severity mix, 100k/yr demand |
| `REQUIREMENT` | Target to beat, never a result | `>90%` sensitivity, `>85%` specificity |
| `TEST VECTOR` | Reference number that pins math, not performance | QWK/Wilson/Dice values in suites |
| `INSUFFICIENT_EVIDENCE` | Rule engine cannot assert a grade (not Grade 0) | VB/IRMA/vitreous UNAVAILABLE |
| `NOT_COMPARABLE` | DL/rule comparison impossible (a grade is NaN) | never manufactured disagreement |
| `ATTESTED_NOT_COMPUTED` | Caller-supplied hash, not recomputed from bytes | freeze inputs without `hashArtifacts` |
| `UNVERIFIABLE` | Audit cannot prove clean (explicit limitation) | near-duplicate check unsupported |

Rules:

1. Targets (`REQUIREMENT`) are never reported as achievements.
2. `TEST VECTOR` / `TEST/LOGIC` values never enter an evaluation report.
3. `SCENARIO` / `ASSUMED` values never justify staffing or clinical claims
   without field measurement (re-run with measured inputs).
4. `MEASURED` requires a real TEST cohort, real weights, and a recorded
   leakage audit (`CLEAN` or `UNVERIFIABLE`-with-limitations; never `FINDINGS`).
5. `UNEXECUTED` suites are never cited as execution evidence (mirrors only).
