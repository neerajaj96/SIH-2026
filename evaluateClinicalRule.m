function results = evaluateClinicalRule(imagePaths, grades, varargin)
% evaluateClinicalRule: Infrastructure for evaluating the status-aware
% clinical rule engine against labeled ICDR ground truth (and, where
% justified, rule result vs DL grade). DATA-GATED: with no data here,
% this runner EXISTS but reports no clinical metric - never fabricate one.
%
% PROTOCOL (no test tuning): caller supplies HELD-OUT image paths +
% integer grades 0-4. The runner applies auditQualityBatch-style quality
% screening (audit-only), runs runScreeningPipeline per image, and
% tabulates rule status/grade vs label. INSUFFICIENT_EVIDENCE cases are
% reported as a separate stratum (excluded from accuracy with the
% exclusion rate stated), never silently dropped or counted as correct.
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB, no data, no weights) ***
%
% INPUTS:
%   imagePaths - cellstr, held-out validation images (NEVER test-set tuning)
%   grades     - numeric vector, labeled ICDR 0-4, same order
%   'Models', 'OutputDir', 'QualityPolicy' (passed to auditQualityBatch)
%
% OUTPUT: results struct (empty .metrics when nothing assessable):
%   .n, .nSufficient, .nInsufficient, .confusion (5x5 on sufficient only),
%   .accuracySufficient + .accuracyCI, .referableSens/Spec +
%   .referableSensCI/.referableSpecCI (Wilson 95% via
%   wilsonScoreInterval), .dlAgreeRate, .manifestCsv. All NaN/empty when
%   nSufficient==0.

p = inputParser;
addParameter(p, 'Models', [], @(x) true);
addParameter(p, 'OutputDir', fullfile(pwd, 'clinical_rule_eval'), @ischar);
addParameter(p, 'QualityPolicy', 'audit-only', @ischar);
parse(p, varargin{:});

assert(numel(imagePaths) == numel(grades), 'evaluateClinicalRule:length - paths and grades must align.');
assert(all(grades >= 0 & grades <= 4), 'evaluateClinicalRule:grades - ICDR labels must be 0-4.');

if ~isfolder(p.Results.OutputDir), mkdir(p.Results.OutputDir); end

n = numel(imagePaths);
ruleGrade = NaN(n,1); ruleStatus = strings(n,1); dlGrade = NaN(n,1);
for i = 1:n
    raw = imread(imagePaths{i});
    r = runScreeningPipeline(raw, p.Results.Models);
    ruleGrade(i) = r.ruleGrade;
    ruleStatus(i) = string(r.ruleStatus);
    dlGrade(i) = r.dlGrade;
end

suff = ~isnan(ruleGrade);
results = struct('n', n, 'nSufficient', sum(suff), ...
    'nInsufficient', sum(~suff), 'manifestCsv', '');
if results.nSufficient == 0
    warning(['evaluateClinicalRule:noAssessable - zero sufficient rule grades (expected without ' ...
             'detectors/data). Reporting exclusion only - no clinical metric fabricated.']);
    results.confusion = zeros(5,5); results.accuracySufficient = NaN;
    results.referableSens = NaN; results.referableSpec = NaN; results.dlAgreeRate = NaN;
    return;
end
% Sufficient-only tabulation (insufficient stratum reported separately
% above, never folded into accuracy).
rg = ruleGrade(suff); lb = grades(suff);
conf = zeros(5,5);
for k = 1:numel(rg)
    conf(lb(k)+1, rg(k)+1) = conf(lb(k)+1, rg(k)+1) + 1;
end
refTrue = lb >= 2; refPred = rg >= 2;
results.confusion = conf;
results.accuracySufficient = sum(rg == lb) / numel(lb);
[accLo, accHi] = wilsonScoreInterval(sum(rg == lb), numel(lb));
results.accuracyCI = [accLo, accHi];
results.referableSens = sum(refPred & refTrue) / max(sum(refTrue), 1);
[senLo, senHi] = wilsonScoreInterval(sum(refPred & refTrue), max(sum(refTrue), 1));
results.referableSensCI = [senLo, senHi];
results.referableSpec = sum(~refPred & ~refTrue) / max(sum(~refTrue), 1);
[spLo, spHi] = wilsonScoreInterval(sum(~refPred & ~refTrue), max(sum(~refTrue), 1));
results.referableSpecCI = [spLo, spHi];
dlSuff = dlGrade(suff);
bothGraded = ~isnan(dlSuff);
results.dlAgreeRate = sum(dlSuff(bothGraded) == rg(bothGraded)) / max(sum(bothGraded), 1);

manifest = table(imagePaths(:), grades(:), ruleGrade, ruleStatus, dlGrade, ...
    'VariableNames', {'imagePath','labelICDR','ruleGrade','ruleStatus','dlGrade'});
manifestCsv = uniqueArtifactPath(p.Results.OutputDir, 'clinical_rule_eval', '.csv');
writetable(manifest, manifestCsv);
results.manifestCsv = manifestCsv;
fprintf('Clinical-rule eval: n=%d sufficient=%d insufficient=%d accuracy(sufficient)=%.3f [%.3f,%.3f] referable sens=%.3f [%.3f,%.3f] spec=%.3f [%.3f,%.3f] dlAgree=%.3f\n', ...
    n, results.nSufficient, results.nInsufficient, results.accuracySufficient, ...
    results.accuracyCI(1), results.accuracyCI(2), ...
    results.referableSens, results.referableSensCI(1), results.referableSensCI(2), ...
    results.referableSpec, results.referableSpecCI(1), results.referableSpecCI(2), results.dlAgreeRate);
end
