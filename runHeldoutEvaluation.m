function report = runHeldoutEvaluation(manifest, predictions, varargin)
% runHeldoutEvaluation: Manifest-gated held-out evaluation orchestrator.
% Enforces split roles + leakage audit BEFORE any metric is computed, and
% returns a report whose every numerical result defaults to
% status=NOT_MEASURED, value=null. Fields become MEASURED only from real
% TEST-cohort predictions supplied by the caller (a real trained run -
% impossible in this workspace, so every number here stays NOT_MEASURED
% until an approved evaluation executes).
%
% ROLE CONTRACT (no test tuning, ever):
%   TRAIN -> weights | VAL -> temperature/thresholds | TEST -> metrics.
%   predictions must cover TEST rows only; VAL-fitted calibration/
%   thresholds are recorded, TEST-fitted anything is rejected.
%
% INPUTS:
%   manifest    - evaluationManifest output (TEST rows evaluated)
%   predictions - struct with .imageId cellstr, .icdr (Nx1 0-4),
%                 optionally .probs (Nx5), .ruleGrade (Nx1, NaN ok),
%                 .checkpointId, .calibrationArtifactId, .splitRoles
%                 struct('calibrationSplit','thresholdSplit'),
%                 .inferenceUsedGT (logical, REQUIRED attestation)
%   'BootstrapB', 'BootstrapSeed', 'ReferableThreshold' (default 2)
%
% OUTPUT: report struct(datasetProvenance, splitProvenance, model,
%   preprocessingVersion, calibration, nPatients, nImages, nExclusions,
%   classDistribution, endpoints{status,value,source per metric},
%   confusion, bootstrap, ablation ('NOT_MEASURED here'), leakageAudit,
%   errorAnalysis, limitations, claimState).
%
% Requires: computeQWK.m, evaluateGrading.m, patientBootstrapCI.m,
%   auditEvalLeakage.m (all reused, never duplicated).

p = inputParser;
addParameter(p, 'BootstrapB', 1000, @isnumeric);
addParameter(p, 'BootstrapSeed', 42, @isnumeric);
addParameter(p, 'ReferableThreshold', 2, @isnumeric);
parse(p, varargin{:});

report = localBlankReport();
report.datasetProvenance = manifest;
if ~isfield(predictions, 'inferenceUsedGT')
    error('runHeldoutEvaluation:attestation - predictions.inferenceUsedGT (predicted-masks-only attestation) is required.');
end
audit = auditEvalLeakage(manifest, 'InferenceUsedGT', predictions.inferenceUsedGT, ...
    'CalibrationSplit', predictions.splitRoles.calibrationSplit, ...
    'ThresholdSplit', predictions.splitRoles.thresholdSplit);
report.leakageAudit = audit;
if strcmp(audit.overall, 'FINDINGS')
    report.limitations{end+1} = 'leakage audit FINDINGS - metrics withheld until resolved.';
    return;
end

rows = manifest.rows;
isTest = strcmp({rows.split}, 'TEST') & ~[rows.exclusion];
tRows = rows(isTest);
[~, loc] = ismember({tRows.imageId}, predictions.imageId);
if any(loc == 0)
    error('runHeldoutEvaluation:coverage - predictions miss %d TEST imageIds.', sum(loc == 0));
end
yTrue = [tRows.labelICDR]';
yPred = predictions.icdr(loc)';
pids = {tRows.patientId};

% Core point metrics (reused engine - never reimplemented here).
eg = evaluateGrading(yTrue, yPred, [], 'ReferableThreshold', p.Results.ReferableThreshold);
report.endpoints.qwk = localMeasured(eg.qwk, 'evaluateGrading/computeQWK on TEST');
report.endpoints.confusion = eg.confusion;
report.endpoints.confusionStatus = 'MEASURED';
report.endpoints.referable = eg.referable;
report.endpoints.perClass = eg.perClass;
report.nImages = numel(yTrue);
if manifest.patientIdsAvailable
    report.nPatients = manifest.nPatients;
    ci = patientBootstrapCI([yTrue, yPred], pids, @(s) computeQWK(s(:,1), s(:,2), 5), ...
        'B', p.Results.BootstrapB, 'Seed', p.Results.BootstrapSeed);
    report.bootstrap.qwk = ci;
else
    report.limitations{end+1} = 'no patient IDs - patient bootstrap impossible; image-level inference only.';
end
report.claimState = 'MEASURED';
end

function r = localBlankReport()
% Every numerical result defaults to NOT_MEASURED/null - MEASURED only
% from the real TEST run above.
m = struct('status', 'NOT_MEASURED', 'value', [], 'source', 'none');
r = struct('datasetProvenance', [], 'splitProvenance', 'TRAIN->weights/VAL->fitting/TEST->metrics', ...
    'model', [], 'preprocessingVersion', '', 'calibration', [], ...
    'nPatients', NaN, 'nImages', 0, 'nExclusions', 0, 'classDistribution', [], ...
    'endpoints', struct('qwk', m, 'confusion', [], 'confusionStatus', 'NOT_MEASURED', ...
        'referable', [], 'perClass', []), ...
    'confusion', [], 'bootstrap', struct('qwk', []), ...
    'ablation', 'NOT_MEASURED (needs baseline+pipeline predictions on same TEST cohort)', ...
    'leakageAudit', [], 'errorAnalysis', [], ...
    'limitations', {{}}, 'claimState', 'NOT_MEASURED');
end
