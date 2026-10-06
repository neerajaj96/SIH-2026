# Known Limitations — SIH26038

Every item here is disclosed somewhere in the code's own comments. This
page exists so you have one place to read from when a judge asks "what
doesn't work yet" — reading this straight, confidently, is a stronger
answer than being caught off guard by a question you technically
documented but couldn't find.

## Not automated at all
- **Venous beading and IRMA** (2 of the 3 "4-2-1" Severe-NPDR triggers)
  have no detector. `assignClinicalGrade.m` accepts manual counts for
  both and defaults to 0 if not supplied. The "4" (quadrant hemorrhage
  count) and neovascularization criteria are the only two of the four
  ICDR triggers this pipeline assesses end to end today.

## Screening aids, not validated detectors
- **Neovascularization** (`detectNeovascularization.m`) is explicitly a
  screening proxy — vessel density + tortuosity in the peridiscal region,
  with thresholds calibrated against synthetic test patterns, not a
  labeled clinical NVD/NVE cohort. Say "screening flag" in your pitch,
  never "detector."
- **Optic disc / fovea localization** was checked against one real fundus
  photo (visually — landmarks landed correctly, OD-to-fovea distance came
  out at 2.17 disc diameters, in the literature's expected 2–2.5 range).
  That is a sanity check, not validation: no mean/median/95th-percentile
  localization error has been computed against expert-annotated
  coordinates across multiple patients, cameras, or image qualities.

## Depends on what actually downloaded
- The validation story is genuinely different depending on whether DDR
  came through. If it did: train on DDR, test on Messidor-2, held out
  entirely — real cross-population external validation. If it didn't:
  Messidor-2 alone, split internally by patient — legitimate,
  leakage-safe, but *within-dataset*, not external. Check which case
  `buildGradingDatasets.m` printed before claiming either one.
- DIARETDB1 ships per-grader confidence markings, not clean binary masks
  — inspect the actual downloaded files before assuming
  `pixelLabelDatastore` can read them directly.
- STARE, CHASE_DB1, HRF, and DDR each need a one-time manual sort into
  the `images/` + `masks/` folder structure `datasetRegistry.m` expects —
  their raw download layouts weren't seen firsthand, so no parser was
  written blind for formats that were never inspected. Once sorted,
  `validateMaskConventions.m` checks the sort was done consistently
  (same foreground/background convention across sources) before training
  starts on the combined set.

## Real numbers don't exist yet until you run these
- No model has been trained end-to-end in this project yet — that's true
  right up until you actually run `train_UNet_Segmentation.m` and
  `train_DR_Grader.m` with a real MATLAB license and real data. Until
  then, sensitivity/specificity/QWK numbers are targets, not results.
- Temperature calibration (`calibrateTemperature.m`) is a real,
  implemented, tested function — but it has to actually be *run* on a
  held-out validation split before `production_inference.m` stops
  reporting confidence as an "UNCALIBRATED PLACEHOLDER."
- The baseline-vs-pipeline ablation (`compareModels.m`, with Wilson-score
  confidence intervals) is fully wired and tested — but needs both
  `train_DR_Grader.m` and `train_Baseline_ResNet50.m` actually run first.
- Trained models now get a `.meta.json` sidecar (`saveModelWithMetadata.m`
  — training date, which datasets were present, a content hash) once you
  do train them, so "which version is this" has an answer. It has no
  answer until a real training run actually happens.

## Engineering caveats worth knowing before a live demo
- SimEvents block library paths and some event-action parameter names are
  more version-sensitive than base MATLAB — verify interactively before
  a live Simulink demo, don't assume the first run will just work.
- `runBatchScreening.m` processes a whole folder and won't halt on one
  bad image, but "won't halt" isn't "validated at scale" — it hasn't
  been run against anything close to 100,000 real images.
- The NetraSetu web console (`netrasetu.html`) runs entirely on
  deterministic demo data — it is not wired to `bridge_server.py` or any
  real MATLAB inference yet. Its shared review queue also depends on the
  `db` capability's organization-level sharing rules, which weren't
  confirmed to work for an external (non-org) viewer such as a judge —
  test that specifically before relying on it live.
- `bridge_server.py` is written against the documented MATLAB Engine for
  Python API and passed everything testable without a real MATLAB
  installation (routing, health checks, the struct-to-JSON conversion
  against a simulated struct) — the actual engine call has not been run
  for real. It also has no authentication and a wide-open CORS policy;
  both need tightening before this leaves a laptop.
- The `db` capability backing the demo queue caps out around 5,000
  documents — nowhere near "100,000+ patients annually." See
  `production_schema.sql` for the real migration target.

## What's actually solid
Worth saying out loud, not just the gaps: the quality gate, enhancement
pipeline, all three trained-network *architectures*, the ordinal-aware
loss, the temperature-scaling math, the ICDR 4-2-1 rule engine's logic,
QWK, Wilson intervals, Erlang-C staffing math, and the patient-level
split leakage guard are all independently tested — `runSelfTests.m`
reruns every one of these checks in MATLAB. The gap is real data and
real training time, not unverified logic.
