# NetraSetu judge demo — 2–5 minute deterministic path (DEMO-labelled)

All steps use DEMO fixtures or LIVE backend state actually present. Nothing
here fabricates a patient outcome. If the bridge/MATLAB/weights are absent,
say so and show the degraded state — that honesty is part of the demo.

## 0. Readiness (30s)

1. Open `netrasetu.html`. Point at the mode badge: LIVE (bridge reachable),
   OFFLINE (unreachable — screening disabled, no mock), DEMO (explicit toggle).
2. Open Connection diagnostics: bridge URL, key-present (never the key),
   API version, model presence, temperature state. If `modelPresent=false`,
   state: "simulated/offline pathway — no trained weights here."

## 1. Upload + quality (45s)

1. Enter `PHC-0042`, upload any fundus photo (or none — DEMO needs no PHI).
2. Toggle **Use demo data**, Run screening. Narrate the 8-stage timeline:
   Quality → Enhancement → Vessel/Lesion → Grading → Clinical rule →
   Explainability → Disagreement → Review.
3. Show the quality gate: PASS vs BORDERLINE (provisional, amber) vs FAIL
   (not gradable — recapture, downstream grades withheld).

## 2. Evidence + grade + calibration (45s)

1. DL grade + confidence + temperature state (CALIBRATED vs
   UNCALIBRATED_FALLBACK vs UNAVAILABLE).
2. Clinical rule second opinion: SUFFICIENT grade vs INSUFFICIENT_EVIDENCE
   (not Grade 0), NV PROXY (screening signal, never a diagnosis).

## 3. Explainability without overclaim (30s)

1. Model attention (Grad-CAM) vs segmented-lesion evidence vs rule evidence.
2. "Red regions are not proof of disease — masks + rule trace are the
   auditable basis." Show DEGRADED/UNAVAILABLE handling.

## 4. Disagreement → review → queue (45s)

1. NUMERIC_DISAGREE / escalate flag → routed to Review queue.
2. Open the case: AI output vs rule output vs reviewer decision (separate
   keys — review never overwrites AI). Confirm a grade, show Reviewed list.
3. Queue: browser-local, 150-cap, 30-day TTL, clear-all. No PHI leaves.

## 5. Telemed + provenance + limitations (45s)

1. Capacity view: Erlang-C staffing from the 14.1% ARDA-sourced SCENARIO
   mix — "scenario, not measured prevalence; re-run with field values."
2. Provenance: `freezeCandidate` PRE_TRAIN → FINALIZE, content-hash IDs,
   calibration-mismatch invalidation, TEST never touched.
3. Limitations (`LIMITATIONS.md` + `docs/GLOSSARY.md`): VB/IRMA unavailable,
   NV proxy, MATLAB-GATED/DATA-GATED items, no clinical claims.

Total: ~4 minutes, ≤15 clicks, recoverable at every step (OFFLINE/FAIL/
UNAVAILABLE all have explicit cards, never blank screens).
