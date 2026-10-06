function results = evaluateGrading(yTrue, yPred, probs, varargin)
% evaluateGrading: Held-out grading evaluation for the 5-class ICDR grader
% (Stage-3 P5). Produces the full metric set a reviewer expects - QWK,
% macro/weighted F1, per-class recall/specificity/precision, confusion
% matrix, referable-DR sens/spec with Wilson CIs, error-distance breakdown
% (adjacent vs severe), and optional ROC-AUC - from HELD-OUT predictions
% only. No training, no tuning, no threshold fitting inside.
%
% HELD-OUT RULE: pass test-split (or val-split) predictions that played no
% role in training, early stopping, temperature fitting or lambda choice.
% Tuning anything against the set you evaluate here contaminates every
% number below (see buildGradingDatasets.m manifest.testHeldOut).
%
% INPUTS:
%   yTrue  - [N 1] integer ICDR grades 0-4 (ground truth, held out)
%   yPred  - [N 1] integer grades 0-4 (argmax-1 of grader softmax)
%   probs  - (optional) [N 5] softmax probability rows, same order. Omit or
%            [] to skip AUC (AUC needs scores, not hard labels).
%   Name/Value:
%     'TargetName' - label for printed tables (default 'grader')
%     'WilsonAlpha' - CI level for referable sens/spec (default 0.05)
%
% OUTPUT:
%   results - struct with .qwk, .confusion (5x5 rows=true cols=pred),
%     .perClass (struct array: recall/specificity/precision/f1/support),
%     .macroF1, .weightedF1, .referable (sens/spec + Wilson CIs + counts),
%     .errDist (meanAbsErr, distHist 0..4, adjacentRate, severeRate),
%     .aucReferable/.aucMacroOVR (only if probs given), .n
%
% Requires: computeQWK.m, wilsonScoreInterval.m (both untouched shared).

p = inputParser();
p.addParameter('TargetName', 'grader', @(s) ischar(s) || isstring(s));
p.addParameter('WilsonAlpha', 0.05, @isnumeric);
p.addParameter('ReferableThreshold', [], @isnumeric);
p.parse(varargin{:});
opt = p.Results;
targetName = char(opt.TargetName);
refThr = opt.ReferableThreshold;
if isempty(refThr)
    gc = gradingConfig(); % frozen ICDR referable cut (currently grade >= 2)
    refThr = gc.referableThreshold;
end

yTrue = yTrue(:); yPred = yPred(:);
n = numel(yTrue);
assert(n >= 1, 'evaluateGrading:empty - no predictions to evaluate.');
assert(numel(yPred) == n, 'evaluateGrading:sizeMismatch - yTrue (%d) vs yPred (%d).', n, numel(yPred));
assert(all(yTrue == floor(yTrue)) && all(yTrue >= 0) && all(yTrue <= 4), 'evaluateGrading:badTrue - yTrue must be integers 0..4.');
assert(all(yPred == floor(yPred)) && all(yPred >= 0) && all(yPred <= 4), 'evaluateGrading:badPred - yPred must be integers 0..4.');
haveProbs = nargin >= 3 && ~isempty(probs);
if haveProbs
    assert(isequal(size(probs), [n 5]), 'evaluateGrading:badProbs - probs must be [N 5], got %s.', mat2str(size(probs)));
end

% Confusion: rows true, cols pred (standard layout).
C = zeros(5, 5);
for i = 1:n
    C(yTrue(i)+1, yPred(i)+1) = C(yTrue(i)+1, yPred(i)+1) + 1;
end

% Per-class one-vs-rest.
perClass = repmat(struct('grade',0,'recall',NaN,'specificity',NaN,'precision',NaN,'f1',NaN,'support',0), 5, 1);
for c = 0:4
    tp = C(c+1, c+1);
    fn = sum(C(c+1, :)) - tp;
    fp = sum(C(:, c+1)) - tp;
    tn = n - tp - fn - fp;
    rec = localSafeDiv(tp, tp + fn, NaN);
    spec = localSafeDiv(tn, tn + fp, NaN);
    prec = localSafeDiv(tp, tp + fp, NaN);
    if isnan(rec) || isnan(prec) || (rec + prec) == 0
        f1 = NaN; % undefined: no support (rec NaN) or no predictions+no hits
        if (tp + fn) > 0 && (tp + fp) == 0
            f1 = 0; % has support but never predicted: recall 0 -> F1 0, not NaN
        end
    else
        f1 = 2 * prec * rec / (prec + rec);
    end
    perClass(c+1) = struct('grade', c, 'recall', rec, 'specificity', spec, ...
        'precision', prec, 'f1', f1, 'support', tp + fn);
end
f1vals = [perClass.f1];
macroF1 = mean(f1vals(~isnan(f1vals)));
supports = [perClass.support];
weightedF1 = sum(f1vals(~isnan(f1vals)) .* supports(~isnan(f1vals))) / max(sum(supports), 1);

% QWK (shared verified implementation).
qwk = computeQWK(yTrue, yPred, 5);

% Referable DR (grade >= refThr, the PS clinical target) + Wilson CIs.
trueRef = yTrue >= refThr; predRef = yPred >= refThr;
tp = sum(trueRef & predRef); fn = sum(trueRef & ~predRef);
tn = sum(~trueRef & ~predRef); fp = sum(~trueRef & predRef);
sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1);
[sensLo, sensHi] = wilsonScoreInterval(tp, tp + fn, opt.WilsonAlpha);
[specLo, specHi] = wilsonScoreInterval(tn, tn + fp, opt.WilsonAlpha);

% Error distance: how far are the misses (adjacent d=1 vs severe d>=2).
absErr = abs(double(yPred) - double(yTrue));
distHist = histcounts(absErr, -0.5:4.5);
nErr = sum(absErr > 0);
adjacentRate = localSafeDiv(sum(absErr == 1), nErr, NaN);
severeRate = localSafeDiv(sum(absErr >= 2), nErr, NaN);

results = struct('qwk', qwk, 'confusion', C, 'perClass', perClass, ...
    'macroF1', macroF1, 'weightedF1', weightedF1, ...
    'referable', struct('sensitivity', sens, 'specificity', spec, ...
        'sensitivityCI', [sensLo sensHi], 'specificityCI', [specLo specHi], ...
        'tp', tp, 'fn', fn, 'tn', tn, 'fp', fp), ...
    'errDist', struct('meanAbsErr', mean(absErr), 'distHist', distHist, ...
        'adjacentRate', adjacentRate, 'severeRate', severeRate), ...
    'n', n, 'targetName', targetName);

if haveProbs
    results.aucReferable = localROCAUC(sum(probs(:,3:5), 2), double(trueRef));
    aucs = nan(1, 5);
    for c = 0:4
        aucs(c+1) = localROCAUC(probs(:,c+1), double(yTrue == c));
    end
    results.aucMacroOVR = mean(aucs, 'omitnan');
end

% Printed report (same numbers as returned - no hidden rounding).
fprintf('=== Grading eval [%s] n=%d ===\n', targetName, n);
fprintf('QWK=%.4f  macroF1=%.4f  weightedF1=%.4f  mean|err|=%.3f (adjacent %.1f%%, severe %.1f%% of %d errors)\n', ...
    qwk, macroF1, weightedF1, mean(absErr), 100 * adjacentRate, 100 * severeRate, nErr);
fprintf('Referable(>=2): sens=%.1f%% [%.1f,%.1f] spec=%.1f%% [%.1f,%.1f] (tp=%d fn=%d tn=%d fp=%d)\n', ...
    100*sens, 100*sensLo, 100*sensHi, 100*spec, 100*specLo, 100*specHi, tp, fn, tn, fp);
fprintf('Per-class  grade recall spec prec f1 (support):\n');
for c = 1:5
    fprintf('  L%d  %.3f %.3f %.3f %.3f (%d)\n', perClass(c).grade, perClass(c).recall, ...
        perClass(c).specificity, perClass(c).precision, perClass(c).f1, perClass(c).support);
end
fprintf('Confusion (rows=true 0-4, cols=pred 0-4):\n');
disp(C);
if haveProbs
    fprintf('AUC referable=%.4f macroOVR=%.4f\n', results.aucReferable, results.aucMacroOVR);
end
end

% ------------------------------------------------------------------
function v = localSafeDiv(num, den, fallback)
if den > 0, v = num / den; else, v = fallback; end
end

% ------------------------------------------------------------------
function auc = localROCAUC(scores, binaryLabels)
% Rank-based (Mann-Whitney U) AUC with average tied ranks. Standalone copy
% of the verified routine in compareModels.m so this file has no hidden
% dependency on that file's local functions.
scores = scores(:); binaryLabels = binaryLabels(:);
n1 = sum(binaryLabels == 1); n0 = sum(binaryLabels == 0);
if n1 == 0 || n0 == 0, auc = NaN; return; end
[sortedX, sortIdx] = sort(scores);
ranks = zeros(numel(scores), 1);
i = 1;
while i <= numel(scores)
    j = i;
    while j < numel(scores) && sortedX(j+1) == sortedX(i), j = j + 1; end
    ranks(i:j) = (i + j) / 2;
    i = j + 1;
end
r = zeros(numel(scores), 1);
r(sortIdx) = ranks;
auc = (sum(r(binaryLabels == 1)) - n1 * (n1 + 1) / 2) / (n1 * n0);
end
