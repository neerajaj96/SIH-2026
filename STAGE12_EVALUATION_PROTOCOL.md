# STAGE-12 EVALUATION PROTOCOL (authoritative, DATA-GATED until executed)

No number in this file is a result. Targets are REQUIREMENTS.

## 1. Dataset acceptance
Approved source + version pinned; image IDs unique; ICDR 0–4 with
adjudication provenance (grader count, adjudication rule); seg-GT per
task with class identity; metadata for any claimed subgroup; license
permitting evaluation use. Unlabeled downloads are INSUFFICIENT, never
"approximately labeled".

## 2. Label provenance
Record grader(s), adjudication, label date/version. Separately
adjudicated referable endpoints preserved alongside ICDR-derived
`>=2` (both reported, never silently replaced).

## 3. Split roles (strict)
TRAIN→weights; VAL→temperature/thresholds/hyperparameters;
TEST→final metrics only. Calibration/thresholds fitted on TEST =
contamination (audit FAILs). Splits frozen before first metric look;
altering TEST after inspecting performance invalidates the run.

## 4. Patient grouping
Group by patient ID whenever IDs exist (eye-level pairing at minimum
where documented). No patient/image across splits. Without IDs:
image-level split + explicit NOT-patient-level limitation.

## 5. Calibration / threshold separation
`calibrationArtifactId` + `thresholdSource` recorded per run; provenance
must match model + split; mismatch → reject artifact, no silent use.

## 6. Endpoints
Primary: QWK + confusion (grading); Dice/IoU per seg task.
Secondary: per-class/macro/weighted, referable sens/spec/PPV/NPV/
prevalence + Wilson, ROC-AUC/macro-OvR + ECE/NLL/reliability when
probs exist, error-distance, McNemar ablation on identical cohorts.

## 7. Uncertainty
Wilson (binary, with n + level) is descriptive/reference under
clustering; the inferential CI under clustering is patient bootstrap
(seed, B, valid/invalid counts + reasons, percentile method, level,
unit PATIENT vs IMAGE_LEVEL_BOOTSTRAP, min-valid policy). Every CI
states method + level + unit + n patients + n images. Minimum-n: no
subgroup claim below 30 patients (labels DATA-GATED anyway).

## 8. Missing data / exclusions / imbalance
Excluded images logged with reasons; missing labels excluded (never
imputed as negatives); invalid labels reject the run; imbalance
reported (counts), never silently reweighted at eval time.

## 9. Duplicates
EXACT (hash/id) → remove + report; POSSIBLE (stem/metadata) → review;
NEAR → UNSUPPORTED here (no similarity method; never claimed).

## 10. Claims policy
VERIFIED = checked here. MEASURED = approved real run only.
NOT MEASURED / DATA-GATED otherwise. Targets (`>90%/>85%`) are
REQUIREMENTS. Synthetic numbers are LOGIC tests, never results.
