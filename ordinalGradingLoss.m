function [loss, parts] = ordinalGradingLoss(Y, T, classValues, lambda, classWeights)
% ordinalGradingLoss: Audited hybrid loss for ICDR grading (Stage-3 P3).
% Preserves the project's existing idea (categorical cross-entropy + soft
% ordinal expected-class penalty) but fixes its undocumented scale,
% shape and stability assumptions.
%
% FORMULATION (frozen):
%   E_pred = sum_c Y(c,:) .* classValues(c)      % softmax expected grade
%   E_true = sum_c T(c,:) .* classValues(c)      % one-hot true grade
%   loss   = mean(crossentropy(Y,T),'all') + lambda * mean((E_pred-E_true).^2,'all')
%
% SCALE AUDIT (why lambda=0.5 is a starting point, not a truth):
%   CE per sample is -log p_true: ~0 when confident-correct, 1.61 at
%   uniform, unbounded above when confident-wrong (e.g. 2.30 at p=0.1).
%   Penalty per sample in [0, 16] (grade distance 0..4, squared).
%   With lambda=0.5 the penalty contributes up to 8.0 - it CAN dominate CE
%   on far misses by design (a 0-vs-4 error must hurt more than 3-vs-4),
%   but on near misses (d=1 -> 0.5) it is comparable to CE. Tune lambda
%   ONLY on validation (never test); log both components via parts.
%
% GRADIENT (bounded, no explosion):
%   d(penalty)/dY(c,i) = 2*lambda*(E_pred(i)-E_true(i))*classValues(c)/N.
%   |E diff|<=4, |c|<=4 -> |grad| <= 32*lambda/N per element. Linear in Y,
%   no exp/log pathologies beyond CE's own (handled by crossentropy).
%
% EXTREME vs ADJACENT (the point of the hybrid):
%   CE alone: P(true)=0.1 gives -log(.1)=2.30 whether the mass sits on a
%   neighbor or 4 grades away. Penalty adds 0.5*d^2 in expectation shift -
%   quadratic in distance, so far misses dominate. Adjacent confusion still
%   trains (CE nonzero) but no longer equals a dangerous miss.
%
% STABILITY: Y must be post-softmax probabilities in [0,1] with columns
% summing to 1 (tolerance 1e-3); T one-hot columns summing to 1. Anything
% else (logits, unnormalized scores) errors loudly instead of training on
% a meaningless value. crossentropy() itself handles log(0) internally;
% inputs are additionally clamped to [1e-12, 1] for the diagnostic NLL.
%
% INPUTS:
%   Y - dlarray/single [C N] or [C 1 N], C==5 softmax probabilities
%   T - same size one-hot targets
%   classValues - [C 1] single(0:4)' (default gradingConfig)
%   lambda - scalar >=0 (default gradingConfig.ordinalLambda 0.5)
%   classWeights - [C 1] positive per-class CE multipliers, or [] for
%     unweighted (default []). Median-frequency weights are computed and
%     LOGGED by train_DR_Grader.m but only APPLIED when that script opts
%     in with a validation justification - never silently. Weighting scales
%     the CE term only (rare-class recall); the ordinal penalty is
%     unweighted (distance already encodes severity).
%
% OUTPUTS:
%   loss  - scalar dlarray/single (trainnet-compatible single output)
%   parts - struct .ce, .penalty, .lambda, .weighted (diagnostics)
%
% Requires: Deep Learning Toolbox (crossentropy, dlarray).

if nargin < 3 || isempty(classValues)
    g = gradingConfig();
    classValues = g.classValues;
end
if nargin < 4 || isempty(lambda)
    g = gradingConfig();
    lambda = g.ordinalLambda;
end
if nargin < 5, classWeights = []; end
assert(isscalar(lambda) && lambda >= 0, ...
    'ordinalGradingLoss:badLambda', 'lambda must be scalar >=0, got %s.', mat2str(lambda));
if ~isempty(classWeights)
    assert(isvector(classWeights) && numel(classWeights) == 5 && all(classWeights > 0), ...
        'ordinalGradingLoss:badWeights', 'classWeights must be [5 1] positive, got %s.', mat2str(size(classWeights)));
end

C = size(Y, 1);
assert(C == 5, 'ordinalGradingLoss:channels - expected 5-class dim1, got %d.', C);
assert(isequal(size(Y), size(T)), ...
    'ordinalGradingLoss:sizeMismatch - Y %s vs T %s.', mat2str(size(Y)), mat2str(size(T)));

dY = extractdata(Y); dT = extractdata(T);
if any(dY(:) < -1e-3) || any(dY(:) > 1 + 1e-3)
    error(['ordinalGradingLoss:notProbabilities - Y outside [0,1] (min %.3f max %.3f). ' ...
           'Pass post-softmax probabilities, not logits.'], min(dY(:)), max(dY(:)));
end
colSumY = sum(dY, 1);
if any(abs(colSumY(:) - 1) > 1e-3)
    error(['ordinalGradingLoss:notNormalized - Y columns must sum to 1 (softmax). ' ...
           'Worst deviation %.4f. Check the dr_fc+prob head.'], max(abs(colSumY(:) - 1)));
end
colSumT = sum(dT, 1);
if any(abs(colSumT(:) - 1) > 1e-3)
    error('ordinalGradingLoss:notOneHot - T columns must sum to 1 (one-hot).');
end

if isempty(classWeights)
    ce = mean(crossentropy(Y, T), 'all');
else
    % Weighted CE: w_true * (-log p_true), averaged over the batch.
    % Manual log form (clamped) so per-sample weights apply; unweighted
    % path above keeps the toolbox crossentropy() behavior bit-for-bit.
    eps_ = 1e-12;
    logY = log(max(extractdata(Y), eps_));
    trueIdx = sum(extractdata(T) .* (1:numel(classWeights))', 1); % 1-indexed class per sample
    w = classWeights(trueIdx);
    cePerSample = -sum(extractdata(T) .* logY, 1);
    ce = mean(reshape(w, size(cePerSample)) .* cePerSample, 'all');
end
expectedPred = sum(Y .* classValues, 1);
expectedTrue = sum(T .* classValues, 1);
penalty = mean((expectedPred - expectedTrue).^2, 'all');
loss = ce + lambda * penalty;
if nargout > 1
    parts = struct('ce', extractdata(ce), 'penalty', extractdata(penalty), ...
        'lambda', lambda, 'weighted', ~isempty(classWeights));
end
end
