# Evaluation Report Template — SIH-2026 DR Screening

> Every numerical field defaults to `status: NOT_MEASURED, value: null`.
> A field becomes `MEASURED` only after the approved protocol run on
> real held-out data. Synthetic/test numbers MUST NEVER appear here.

- dataset: {name, version, source} — status: NOT_MEASURED
- split: {nPatients, nImages, TRAIN/VAL/TEST counts, exclusions+reasons}
- model: {checkpointId, preprocessingVersion, calibrationArtifactId}
- grading: {qwk, confusion5x5, perClass, macro, weighted,
  referable{sens, spec, PPV, NPV, prevalence, WilsonCI}} — NOT_MEASURED
- segmentation (per task vessels/mahe/exudate):
  {dice, iou, sens, spec, precision, pooled, emptyPolicy} — NOT_MEASURED
- calibration: {T, datasetTag, NLL/ECE before/after, reliability} — NOT_MEASURED
- ablation: {baselineVsPipeline, McNemar, AUC} — NOT_MEASURED
- ruleVsDL: {agreement, disagreement, insufficientRate} — NOT_MEASURED
- bootstrap: {method, seed, B, valid/invalid, unit, level} — NOT_MEASURED
- leakageAudit: {overall, checks[]} — runnable without data
- limitations: [...]
- claimState: NOT_MEASURED

Targets (`>90% sensitivity`, `>85% specificity`) are REQUIREMENTS
listed here for reference only — never as achieved results.
