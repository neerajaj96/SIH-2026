function ci = patientBootstrapCI(values, groups, metricFcn, varargin)
% patientBootstrapCI: Patient-level bootstrap confidence interval for an
% aggregate metric (QWK, referable sens/spec, AUC where valid).
%
% Resampling unit is PATIENTS whenever grouping exists (clustered
% uncertainty); image-level resampling is available ONLY as an explicit,
% weaker fallback labeled IMAGE_LEVEL_BOOTSTRAP (never presented as
% patient-level uncertainty). Wilson binary CIs (evaluateGrading.m) stay
% valid as descriptive/reference intervals for single-image-per-patient
% data only.
%
% DEGENERACY CONTRACT (never silent zeros):
%   valid replicate   -> metric recorded
%   invalid replicate -> NaN + explicit reason counted
% Undefined metric domains (single-class sample, no referable positives/
% negatives, undefined QWK/AUC, any metric error) are invalid replicates.
% If valid replicates < minValid (default: 200 or B if B<200... see
% below), CI status = UNAVAILABLE with reason (never a fabricated band).
%
% INPUTS:
%   values  - N x d numeric matrix (per-image metric inputs: labels,
%             predictions, scores - metricFcn decides the layout)
%   groups  - N x 1 patient identifiers (cellstr/numeric/string); [] for
%             image-level fallback (labeled as such in output)
%   metricFcn - @(sampleRows)scalar; must error or return NaN/Inf on
%             undefined domains (both count as invalid replicates)
%   'B' (default 1000), 'Seed' (default 42), 'Alpha' (default 0.05),
%   'MinValid' (default min(B,200)), 'Method' ('percentile', default)
%
% OUTPUT: ci struct(seed, B, validReplicates, invalidReplicates,
%   invalidReasons{ deduped }, alpha, method, resamplingUnit
%   ('PATIENT'|'IMAGE_LEVEL_BOOTSTRAP'), point, lo, hi, status
%   ('AVAILABLE'|'UNAVAILABLE'), nPatients, nImages).
%
% Requires: base MATLAB + Statistics Toolbox only if callers use tinv
% (this file uses prctile only - base MATLAB).

p = inputParser;
addParameter(p, 'B', 1000, @(x) isnumeric(x) && isscalar(x) && x >= 100);
addParameter(p, 'Seed', 42, @isnumeric);
addParameter(p, 'Alpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
addParameter(p, 'MinValid', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
addParameter(p, 'Method', 'percentile', @(x) strcmp(x,'percentile'));
parse(p, varargin{:});

B = p.Results.B;
minValid = p.Results.MinValid;
if isempty(minValid), minValid = min(B, 200); end

N = size(values, 1);
if isempty(groups)
    unit = 'IMAGE_LEVEL_BOOTSTRAP';
    unitIds = (1:N)';
    nPatients = NaN;
else
    unit = 'PATIENT';
    if iscell(groups), groups = string(groups); end
    [~, ~, unitIds] = unique(groups(:));
    nPatients = max(unitIds);
end
nUnits = max(unitIds);

% Deterministic stream (RandStream, not global rng state).
rs = RandStream('mt19937ar', 'Seed', p.Results.Seed);
mets = NaN(B, 1);
reasons = {};
nInvalid = 0;
for b = 1:B
    draw = randi(rs, nUnits, [nUnits, 1]);
    sel = ismember(unitIds, draw);
    if sum(sel) == 0
        nInvalid = nInvalid + 1; reasons{end+1} = 'empty-resample'; %#ok<AGROW>
        continue;
    end
    try
        m = metricFcn(values(sel, :));
    catch ME
        nInvalid = nInvalid + 1; reasons{end+1} = ['metric-error: ' ME.identifier]; %#ok<AGROW>
        continue;
    end
    if ~isscalar(m) || ~isfinite(m)
        nInvalid = nInvalid + 1; reasons{end+1} = 'undefined-domain (NaN/Inf/non-scalar)'; %#ok<AGROW>
        continue;
    end
    mets(b) = m;
end
valid = mets(isfinite(mets));
nValid = numel(valid);

ci = struct('seed', p.Results.Seed, 'B', B, 'validReplicates', nValid, ...
    'invalidReplicates', nInvalid, 'invalidReasons', {unique(reasons)}, ...
    'alpha', p.Results.Alpha, 'method', p.Results.Method, ...
    'resamplingUnit', unit, 'nPatients', nPatients, 'nImages', N, ...
    'point', NaN, 'lo', NaN, 'hi', NaN, 'status', 'UNAVAILABLE', ...
    'unavailableReason', '');
if nValid < minValid
    ci.unavailableReason = sprintf(['only %d/%d valid replicates (minimum %d) - degenerate resampling ' ...
        '(single-class draws, no positives/negatives, or undefined metric); no CI fabricated.'], nValid, B, minValid);
    return;
end
try
    point = metricFcn(values);
catch
    point = NaN;
end
ci.point = point;
ci.lo = prctile(valid, 100 * p.Results.Alpha / 2);
ci.hi = prctile(valid, 100 * (1 - p.Results.Alpha / 2));
ci.status = 'AVAILABLE';
end
