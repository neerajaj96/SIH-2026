function rep = testStage11Runtime(outDir)
% testStage11Runtime: Stage-11 real-runtime verification with HONEST
% tri-state accounting. A check that cannot run (no MATLAB assets, no
% weights, no Engine) is recorded UNEXECUTED - it NEVER increments PASS.
% Simulated-mode checks prove plumbing only, and are labeled SIMULATED;
% they can never satisfy a trained-runtime level.
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB here) ***
% Run in MATLAB: >> rep = testStage11Runtime()  (evidence dir defaults
% to tempdir - repository is never polluted with generated artifacts).
%
% LEVELS: A env, B simulated pipeline, C trained pipeline (needs .mat),
% D Engine bridge (needs Engine+bridge), E live HTTP (needs serving
% bridge), F repeat/reuse, G failure/recovery, H performance.
%
% OUTPUT: rep struct(passed{}, failed{}, unexecuted{}, gated{}) where
% each entry names the check; gated lists MATLAB-/DATA-GATED items.

if nargin < 1 || isempty(outDir), outDir = tempdir; end
rep = struct();
rep.passed = {}; rep.failed = {}; rep.unexecuted = {};
rep.gated = {'MATLAB runtime (all execution)', 'trained weights (.mat assets)', ...
    'MATLAB Engine + live bridge', 'deployment/TLS', 'clinical accuracy (DATA)'};

[rep, env] = rt(rep, 'A.env-detected', @() deal(true, ''), @() cEnv(outDir));
[rep, hasNets] = rtFlag(rep, 'C.assets-present', @() cAssets());
[rep] = rt(rep, 'B.simulated-quality-gate', @() deal(true, ''), @() cSimQuality());
[rep] = rt(rep, 'B.simulated-insufficient-evidence', @() deal(true, ''), @() cSimInsuff());
[rep] = rt(rep, 'B.simulated-malformed-input', @() deal(true, ''), @() cSimMalformed());
[rep] = rt(rep, 'C.trained-single-image', @() needNets(hasNets), @() cTrained());
[rep] = rt(rep, 'C.trained-5channel-fusion-order', @() needNets(hasNets), @() cFusion());
[rep] = rt(rep, 'C.trained-temperature-scaling', @() needNets(hasNets), @() cTemp());
[rep] = rt(rep, 'C.trained-gradcam-executes', @() needNets(hasNets), @() cGradcam());
[rep] = rt(rep, 'C.clinical-evidence-propagation', @() needNets(hasNets), @() cClinical());
[rep] = rt(rep, 'D.engine-conversion', @() needEngine(), @() cEngine());
[rep] = rt(rep, 'E.live-http-screen', @() needBridge(), @() cLive());
[rep] = rt(rep, 'F.repeat-state-isolation', @() deal(true, ''), @() cRepeat());
[rep] = rt(rep, 'G.failure-recovery', @() deal(true, ''), @() cFailure());
[rep] = rt(rep, 'H.performance-measured', @() needNets(hasNets), @() cPerf(outDir));
rep.env = env;

fprintf('\n=== stage11 runtime: %d PASS, %d FAIL, %d UNEXECUTED ===\n', ...
    numel(rep.passed), numel(rep.failed), numel(rep.unexecuted));
if ~isempty(rep.failed)
    fprintf('FAILED: %s\n', strjoin(rep.failed, ', '));
end
if ~isempty(rep.unexecuted)
    fprintf('UNEXECUTED (not PASS): %s\n', strjoin(rep.unexecuted, ', '));
end
end

function [rep, out] = rtFlag(rep, name, fn)
try
    out = fn();
    rep.passed{end+1} = name;
    fprintf('[PASS] %s\n', name);
catch E
    rep.failed{end+1} = sprintf('%s (%s)', name, E.message);
    fprintf('[FAIL] %s :: %s\n', name, E.message);
    out = false;
end
end

function [rep, out] = rt(rep, name, preFn, fn)
[ready, reason] = preFn();
if ~ready
    rep.unexecuted{end+1} = sprintf('%s [%s]', name, reason);
    fprintf('[UNEXECUTED] %s :: %s\n', name, reason);
    out = [];
    return;
end
try
    out = fn();
    rep.passed{end+1} = name;
    fprintf('[PASS] %s\n', name);
catch E
    rep.failed{end+1} = sprintf('%s (%s)', name, E.message);
    fprintf('[FAIL] %s :: %s\n', name, E.message);
    out = [];
end
end

function [ok, why] = needNets(hasNets)
ok = hasNets; why = 'no .mat assets (need unet_*.mat + trained_dr_grader.mat)';
end
function [ok, why] = needEngine()
[~, hasEng] = system('python3 -c "import matlab.engine" 2>/dev/null');
ok = (hasEng == 0); why = 'no MATLAB Engine for Python here';
end
function [ok, why] = needBridge()
ok = false; why = 'no serving bridge in a unit MATLAB run (use scripts/smoke_e2e.py live)';
end

function env = cEnv(outDir)
env = matlab_env_report(outDir);
assert(isfield(env,'matlabVersion') && isfield(env,'assetInventory'), 'env report incomplete');
end

function hasNets = cAssets()
m = loadModelsIfPresent();
hasNets = m.haveTrainedModels;
assert(isfield(m,'calibrationState'), 'cache contract drift');
end

function cSimQuality()
img = uint8(128*ones(64,64,3));
rep = assessFundusQuality(img);
assert(ismember(rep.decision, {'PASS','BORDERLINE','FAIL'}), 'bad decision enum');
end

function cSimInsuff()
img = uint8(128*ones(64,64,3));
r = runScreeningPipeline(img);
if isnan(r.ruleGrade)
    assert(strcmp(r.ruleStatus,'INSUFFICIENT_EVIDENCE') || strcmp(r.ruleStatus,'INVALID'), 'NaN grade needs insufficient status');
end
end

function cSimMalformed()
try
    assessFundusQuality(uint8([]));
    error('expected:shouldHaveErrored', 'empty input accepted');
catch E
    assert(contains(E.identifier,'badInput'), 'wrong error contract');
end
end

function cTrained()
img = uint8(128*ones(64,64,3));
r = runScreeningPipeline(img);
assert(strcmp(r.status,'ok') || strcmp(r.status,'ungradeable'), 'unexpected status');
assert(~isnan(r.dlGrade) || strcmp(r.status,'ungradeable'), 'trained run without DL grade');
end

function cFusion()
img = uint8(128*ones(64,64,3));
[~, eRGB, eGray, ~, ~, roi] = assessAndEnhanceImage(img, -Inf, -Inf);
m = loadModelsIfPresent();
v = runSegmentationNet(m.vesselNet, eGray, roi);
ma = runSegmentationNet(m.maheNet, eGray, roi);
ex = runSegmentationNet(m.exudateNet, eGray, roi);
f = buildGradingFusionTensor(eRGB, v, ma, ex);
assert(isequal(size(f), [224 224 5]), 'fusion shape drift');
assert(isa(f,'single'), 'fusion dtype drift');
end

function cTemp()
m = loadModelsIfPresent();
assert(isfinite(m.temperatureT) && m.temperatureT > 0, 'non-positive temperature');
assert(ismember(m.calibrationState, {'CALIBRATED_VALID','UNCALIBRATED_FALLBACK','CALIBRATED_MISMATCH'}), 'bad calibration state');
end

function cGradcam()
img = uint8(128*ones(64,64,3));
[~, eRGB, eGray, ~, ~, roi] = assessAndEnhanceImage(img, -Inf, -Inf);
m = loadModelsIfPresent();
v = runSegmentationNet(m.vesselNet, eGray, roi);
ma = runSegmentationNet(m.maheNet, eGray, roi);
ex = runSegmentationNet(m.exudateNet, eGray, roi);
fused = buildGradingFusionTensor(eRGB, v, ma, ex);
dlX = dlarray(fused, 'SSC');
[~, idx] = max(extractdata(softmax(predict(m.drNet, dlX, 'Outputs', 'dr_fc') ./ m.temperatureT)));
[mapOut, rep] = explainGradCAM(m.drNet, dlX, idx, roi, ma|ex, [32 32], 5);
assert(ismember(rep.status, {'VALID','DEGRADED','UNAVAILABLE','INVALID'}), 'bad explanation status');
assert(isempty(mapOut) || isequal(size(mapOut,[1 2]), size(roi,[1 2])), 'map/ROI size drift');
end

function cClinical()
img = uint8(128*ones(64,64,3));
r = runScreeningPipeline(img);
assert(ismember(r.ruleStatus, {'SUFFICIENT','INSUFFICIENT_EVIDENCE','PROXY','INVALID'}), 'bad rule status');
assert(isfield(r,'nvStatus') && isfield(r,'quadrantValid'), 'clinical propagation drift');
end

function cEngine()
error('testStage11Runtime:manualStep', 'run scripts/smoke_e2e.py against a live bridge (see runbook)');
end

function cLive()
error('testStage11Runtime:manualStep', 'needs serving bridge + fixture (see runbook)');
end

function cRepeat()
a = runScreeningPipeline(uint8(128*ones(48,48,3)));
b = runScreeningPipeline(uint8(128*ones(48,48,3)));
assert(strcmp(a.status,b.status) && strcmp(a.qualityDecision,b.qualityDecision), 'repeat unstable');
assert(isequal(size(a.enhancedGray),size(b.enhancedGray)), 'shape unstable');
end

function cFailure()
r = runScreeningPipeline(zeros(32,32,3,'uint8'));
assert(ismember(r.status, {'ok','ungradeable','error'}), 'bad status enum');
if strcmp(r.status,'error')
    assert(~isempty(r.errorMessage), 'error without message');
end
end

function cPerf(outDir)
fixture = fullfile(outDir, 'stage11_perf_fixture.png');
imwrite(uint8(128*ones(96,96,3)), fixture);
bench = benchmarkStage11(fixture, outDir, 2);
assert(bench.coldSec > 0 && bench.warmSecMean > 0, 'non-positive timing');
end
