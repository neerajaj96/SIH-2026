function runGradingSelfTests()
% runGradingSelfTests: MATLAB-side checks for the Stage-3 grading
% subsystem. Companion to runSelfTests.m (which stays untouched this
% stage) and to tests/test_grading_mirror.py (which runs without MATLAB).
%
% STATUS: UNVERIFIED in this workspace (no MATLAB/Octave runtime, no
% clinical data, no .mat weights here). Run in MATLAB with Deep Learning
% Toolbox before trusting any grading number. Each check prints PASS/FAIL
% and the suite errors out on any failure.
%
% DATA-GATED checks (buildGradingDatasets splits, real trainnet runs) are
% NOT included here - they need data/ + .mat weights. They are covered by
% evaluateGrading.m + buildGradingDatasets.m manifest logging at run time.
%
% Requires: Deep Learning Toolbox (dlarray, crossentropy) for the loss
% checks; base MATLAB + Image Processing Toolbox for fusion/eval checks.

nPassed = 0; nTotal = 0;

[nPassed, nTotal] = localCheck(nPassed, nTotal, ...
    'gradingConfig freezes 5ch order, 224x224x5, labels 0-4, referable>=2', @() localTestConfig());

[nPassed, nTotal] = localCheck(nPassed, nTotal, ...
    'buildGradingFusionTensor builds [224 224 5] single with merged lesion + loud validation', @() localTestFusion());

[nPassed, nTotal] = localCheck(nPassed, nTotal, ...
    'ordinalGradingLoss: confident-correct small, far miss > adjacent miss', @() localTestOrdinal());

[nPassed, nTotal] = localCheck(nPassed, nTotal, ...
    'evaluateGrading reproduces QWK reference + confusion + referable counts', @() localTestEval());

fprintf('\n=== %d / %d grading checks passed ===\n', nPassed, nTotal);
if nPassed < nTotal
    error('runGradingSelfTests:failures', '%d check(s) failed - see above.', nTotal - nPassed);
end
end

% ------------------------------------------------------------------
function [nPassed, nTotal] = localCheck(nPassed, nTotal, description, testFcn)
nTotal = nTotal + 1;
fprintf('[%2d] %s ... ', nTotal, description);
try
    testFcn();
    fprintf('PASS\n');
    nPassed = nPassed + 1;
catch err
    fprintf('FAIL\n      %s\n', err.message);
end
end

% ------------------------------------------------------------------
function localTestConfig()
g = gradingConfig();
assert(isequal(g.channelOrder, {'R','G','B','vessel','lesion'}), 'channel order');
assert(isequal(g.inputSize, [224 224]) && g.inputChannels == 5, 'input size/channels');
assert(isequal(g.classValues, single(0:4)'), 'class values');
assert(g.numClasses == 5 && g.referableThreshold == 2, 'classes/referable');
assert(strcmp(g.photoInterp, 'bilinear') && strcmp(g.maskInterp, 'nearest'), 'interp semantics');
assert(g.ordinalLambda == 0.5 && g.seed == 42, 'loss default/seed');
end

% ------------------------------------------------------------------
function localTestFusion()
rng(0);
rgb = uint8(repmat(linspace(0, 255, 64)', [1 64 3]));
vessel = false(64, 64); vessel(1:8, :) = true;
mahe = false(64, 64); mahe(10:12, 10:12) = true;
exud = false(64, 64); % empty exudate: lesion must still equal mahe
fused = buildGradingFusionTensor(rgb, vessel, mahe, exud);
assert(isequal(size(fused), [224 224 5]), 'fused size');
assert(isa(fused, 'single'), 'fused dtype');
ch4 = fused(:, :, 4); ch5 = fused(:, :, 5);
assert(all(ismember(unique(ch4(:))', [0 255])), 'vessel channel binary');
assert(all(ismember(unique(ch5(:))', [0 255])), 'lesion channel binary');
% Lesion fires where mahe fires (resized): any lesion mass must exist.
assert(any(ch5(:) ~= 0), 'merged lesion must carry mahe signal');
% Fractional tripwire: bicubic-style input must error, not train.
threw = false;
try
    buildGradingFusionTensor(rgb, double(vessel) * 0.5, mahe, exud);
catch
    threw = true;
end
assert(threw, 'fractional mask must raise');
% Size mismatch must error.
threw = false;
try
    buildGradingFusionTensor(rgb, true(32, 32), mahe, exud);
catch
    threw = true;
end
assert(threw, 'mask/rgb size mismatch must raise');
end

% ------------------------------------------------------------------
function localTestOrdinal()
% Confident-correct ~ small; far miss >> adjacent miss at equal p_true.
pOK = single([0.02 0.02 0.9 0.03 0.03]');
t2 = single([0 0 1 0 0]');
[lossOK, partsOK] = ordinalGradingLoss(dlarray(pOK), dlarray(t2), single(0:4)', 0.5);
assert(extractdata(partsOK.penalty) < 0.05, 'confident-correct penalty small');
pAdj = single([0.05 0.05 0.1 0.7 0.1]'); % true 4, mass on 3
pFar = single([0.7 0.1 0.05 0.05 0.1]'); % true 4, same p_true, mass 4 away
t4 = single([0 0 0 0 1]');
[~, pA] = ordinalGradingLoss(dlarray(pAdj), dlarray(t4), single(0:4)', 0.5);
[~, pF] = ordinalGradingLoss(dlarray(pFar), dlarray(t4), single(0:4)', 0.5);
assert(extractdata(pF.penalty) > 5 * extractdata(pA.penalty), 'far miss penalty >> adjacent');
% Weighted variant upweights rare-class CE.
[~, pW0] = ordinalGradingLoss(dlarray(pFar), dlarray(t4), single(0:4)', 0.5, []);
[lossW, ~] = ordinalGradingLoss(dlarray(pFar), dlarray(t4), single(0:4)', 0.5, [1 1 1 1 5]');
assert(extractdata(lossW) > extractdata(pW0), 'rare-class weight must increase loss');
end

% ------------------------------------------------------------------
function localTestEval()
% QWK reference shared with runSelfTests.m (sklearn quadratic kappa).
yTrue = [0 0 1 1 2 2 3 3 4 4 2 1]';
yPred = [0 1 1 2 2 3 3 3 4 3 2 1]';
r = evaluateGrading(yTrue, yPred);
assert(abs(r.qwk - 0.88940092) < 1e-6, 'QWK reference');
assert(isequal(size(r.confusion), [5 5]) && sum(r.confusion(:)) == 12, 'confusion');
assert(r.referable.tp + r.referable.fn + r.referable.tn + r.referable.fp == 12, 'referable counts');
assert(all(r.referable.sensitivityCI >= 0) && all(r.referable.sensitivityCI <= 1), 'Wilson bounds');
assert(r.errDist.distHist(1) == sum(yTrue == yPred), 'error histogram d=0');
end
