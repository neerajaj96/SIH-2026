# SIH26038 — Diabetic Retinopathy Screening Pipeline — Project Handoff

**Context for a new Claude conversation:** This is a MATLAB/Simulink project for
Smart India Hackathon 2026, Problem Statement 26038 ("Explainable AI for
Diabetic Retinopathy Screening in Rural India"), sponsored by MathWorks. Goal
is to place as high as possible in SIH, not just have a working prototype.
Re-upload the accompanying `.m` files along with this summary.

## Files in this project
- `assessAndEnhanceImage.m` — image quality gate (focus/entropy, ROI-restricted)
  + CLAHE/flat-field enhancement. Returns both RGB and grayscale enhanced output.
- `calibrateQualityThresholds.m` — derives gate thresholds from labeled images
  instead of guessing a constant.
- `train_UNet_Segmentation.m` — trains 3 binary U-Nets (vessels, MA/hemorrhage,
  exudates) via `unet`+`trainnet`, real Tversky loss.
- `train_DR_Grader.m` — DenseNet-121 (`imagePretrainedNetwork`), 5-channel input
  (RGB+vessel+lesion masks), genuine softmax head trained with a hybrid
  cross-entropy + ordinal-penalty loss (resolves a regression-vs-calibration
  architecture conflict — see "Key decisions" below).
- `localizeOpticDiscFovea.m` — optic disc + fovea landmarks (brightness +
  vessel-convergence / darkest-point methods). Verified against a real sample image.
- `partitionQuadrants.m` — 4 clinical grading quadrants centered on the fovea.
- `detectNeovascularization.m` — vessel density/tortuosity screening proxy
  (explicitly NOT a validated detector — say so to judges).
- `assignClinicalGrade.m` — the actual ICDR "4-2-1 rule" engine, giving
  lesion-level evidence tied to clinical criteria, as a second opinion
  alongside the deep-learning classifier.
- `optimizeResourceAllocation.m` — Erlang-C queueing analysis answering
  "how many ophthalmologists" for a target patient volume.
- `buildTelemedModel.m` / `SimEvents_Telemed_Model.m` — the SimEvents queue
  simulation (patient arrivals → bandwidth queue → AI → triage → review).
- `train_Baseline_ResNet50.m`, `compareModels.m`, `computeQWK.m` — the
  ablation/baseline comparison ("pipeline beats single-technique approach").
- `production_inference.m` — ties everything together; runs in SIMULATED
  mode (clearly labeled) until real trained `.mat` files exist.

## Key decisions made (don't redo these without reason)
1. **Regression vs. softmax conflict**: original design used ordinal
   regression (no softmax) for grading, but also wanted temperature-scaled
   calibration + Grad-CAM, which need a softmax. Fixed by using a real
   5-class softmax trained with cross-entropy + a soft ordinal penalty on
   the expected class value — not scalar regression.
2. **4-2-1 rule as a second opinion**: `assignClinicalGrade.m` runs
   independently of the deep-learning classifier; if they disagree,
   `production_inference.m` flags it rather than hiding it.
3. **Neovascularization detection** is honestly scoped as a screening proxy,
   not a claimed detector — this is a genuinely hard, open problem.
4. **Venous beading and IRMA** (2 of the 3 "4-2-1" triggers) have no
   detector — `assignClinicalGrade.m` accepts manual counts for them but
   doesn't compute them. Be upfront about this if asked.
5. Resource allocation uses Erlang-C math, not a SimEvents multi-server
   sweep (modeling N parallel reviewers in SimEvents would need untested
   topology). Finding: ~1 reviewer suffices at 100k patients/yr with good
   triage; a weaker triage + bigger scale (2M/yr, 50% referable) needs
   5–17 reviewers depending on review speed.

## Validated vs. not
Loss functions, QWK (matches scikit-learn), the 4-2-1 rule (7/7 test
cases), optic disc/fovea localization (checked against a real fundus
photo), and Erlang-C (matches closed-form M/M/1) were all numerically
tested before shipping — caught and fixed two real bugs this way. NOT
validated: anything needing real MATLAB + Deep Learning Toolbox + Simulink/
SimEvents + actual trained weights (none of that exists in this sandbox).
SimEvents block library paths carry a "verify interactively" caution.

## Timeline (confirm current status)
As of this project's last discussion, SIH 2026 internal/institute-level
hackathons were running through September 2026, with national screening
results expected October and the Grand Finale in December — confirm with
your SPOC, as dates may have moved.

## Not yet done
- No real trained model weights exist yet — this is the biggest gap.
- Venous beading / IRMA detectors.
- Real MATLAB verification of the several "verify interactively" flagged
  API calls (search each file for that phrase).
