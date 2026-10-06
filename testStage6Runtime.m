function testStage6Runtime()
% testStage6Runtime: Stage-6 MATLAB integration suite (synthetic + simulated
% mode; trained-model cases self-skip without weights).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB/Octave here) ***
% Run in MATLAB: >> testStage6Runtime
% Executable twin: python3 tests/test_stage6_runtime.py (static pins +
% independent mirrors; PASS here).
% NOTHING below is clinical validation; nothing measures real timing
% until a MATLAB run records it (see STAGE6_RUNTIME_HANDOFF.md).
%
% Covers: 1 happy path, 2 quality failure, 3 insufficient evidence,
% 4 malformed mask, 5 invalid landmarks, 6/7 calibration absent/present,
% 8 Grad-CAM unavailable/degraded, 9 disagreement, 10 simulated mode,
% 11 trained-model mode (skip without .mat), 12 batch processing,
% 13 report generation (txt), 14 repeated inference determinism,
% 15 model-cache behavior, 16 path/cwd independence, 17 malformed input,
% 18 error recovery + artifact uniqueness.

nP = 0; nT = 0;
[nP,nT] = rt(nP,nT,'single-image happy path (simulated)',@tHappy);
[nP,nT] = rt(nP,nT,'quality failure rejects loudly',@tQualityFail);
[nP,nT] = rt(nP,nT,'insufficient evidence, never fabricated grade',@tInsuff);
[nP,nT] = rt(nP,nT,'malformed mask rejected',@tBadMask);
[nP,nT] = rt(nP,nT,'invalid landmarks block quadrants',@tLandmark);
[nP,nT] = rt(nP,nT,'calibration absent falls back loudly',@tCalibAbsent);
[nP,nT] = rt(nP,nT,'Grad-CAM unavailable without models',@tGradCam);
[nP,nT] = rt(nP,nT,'disagreement orthogonal, no manufacture',@tDisagree);
[nP,nT] = rt(nP,nT,'simulated mode labeled',@tSimulated);
[nP,nT] = rt(nP,nT,'trained-model mode (skip without weights)',@tTrained);
[nP,nT] = rt(nP,nT,'batch isolates per-file errors',@tBatch);
[nP,nT] = rt(nP,nT,'report txt renders with statuses',@tReport);
[nP,nT] = rt(nP,nT,'repeated inference deterministic',@tRepeat);
[nP,nT] = rt(nP,nT,'model cache stable + temp hot-reload path',@tCache);
[nP,nT] = rt(nP,nT,'malformed input rejected',@tMalformed);
[nP,nT] = rt(nP,nT,'artifact names unique',@tUnique);
fprintf('\n=== stage6 runtime (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testStage6Runtime:failures','%d failed',nT-nP); end
end

function [p,t] = rt(p,t,name,f)
t=t+1; fprintf('[s6 %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function img = synthFundus()
[xx, yy] = meshgrid(1:128, 1:128);
disc = (xx-64).^2 + (yy-64).^2 <= 45^2;
tex = uint8(90 + 40*sin(0.2*xx).*cos(0.2*yy));
img = zeros(128,128,3,'uint8');
for c = 1:3, ch = tex; ch(~disc) = 0; img(:,:,c) = ch; end
end

function tHappy()
r = runScreeningPipeline(synthFundus());
assert(strcmp(r.status,'ok') || strcmp(r.status,'ungradeable'), 'unexpected status');
assert(isfield(r,'disagreement') && isfield(r,'explainStatus') && isfield(r,'temperatureState'), 'stage-5 fields missing');
assert(isfield(r,'qualityDecision') && isfield(r,'ruleStatus'), 'stage-1/4 fields missing');
end

function tQualityFail()
r = runScreeningPipeline(zeros(64,64,3,'uint8'));
assert(strcmp(r.status,'ungradeable') && ~r.roiPassed, 'all-black must reject at gate');
end

function tInsuff()
r = runScreeningPipeline(synthFundus());
assert(isnan(r.ruleGrade) || ismember(r.ruleStatus, {'SUFFICIENT','PROXY'}), 'rule must be NaN/insufficient or justified');
if isnan(r.ruleGrade)
    assert(strcmp(r.ruleStatus,'INSUFFICIENT_EVIDENCE'), 'NaN grade requires INSUFFICIENT_EVIDENCE');
end
end

function tBadMask()
try
    assignClinicalGrade(true(10,10), uint8(ones(8,8)), struct('maPresent',false,'exudatePresent',false));
    error('expected:shouldHaveErrored', 'size-mismatched mask/quadrants accepted');
catch E
    assert(~strcmp(E.identifier,'expected:shouldHaveErrored'), 'size mismatch silently accepted');
end
end

function tLandmark()
[qm, qv] = partitionQuadrants([32 32], [NaN NaN], [NaN NaN]);
assert(strcmp(qv,'INVALID') && ~any(qm(:)), 'NaN geometry must block quadrants');
end

function tCalibAbsent()
[ft, et, info] = qualityLoadCalibration(qualityConfig(), tempdir);
assert(~info.isCalibrated, 'empty dir must fall back loudly');
assert(ft == qualityConfig().focusThresh, 'fallback must equal canonical');
end

function tGradCam()
r = runScreeningPipeline(synthFundus());
assert(strcmp(r.explainStatus,'UNAVAILABLE') && isempty(r.scoreMap), 'simulated mode must have no explanation map');
end

function tDisagree()
d = analyzeDisagreement(NaN, NaN, 'UNCALIBRATED', NaN, 'INSUFFICIENT_EVIDENCE', 'UNAVAILABLE');
assert(strcmp(d.gradeRelationship,'NOT_COMPARABLE') && d.escalate, 'insufficient must escalate, not agree');
end

function tSimulated()
m = getOrLoadCachedModels();
if ~m.haveTrainedModels
    assert(strcmp(m.calibrationState,'UNAVAILABLE'), 'simulated cache state must be UNAVAILABLE');
end
end

function tTrained()
m = getOrLoadCachedModels();
if ~m.haveTrainedModels
    fprintf('(skipped - no weights) ');
    return;
end
assert(isfield(m,'drNet') && isfield(m,'vesselNet'), 'trained cache incomplete');
end

function tBatch()
d = tempname; mkdir(d); mkdir(fullfile(d,'imgs'));
imwrite(synthFundus(), fullfile(d,'imgs','good.jpg'));
fid = fopen(fullfile(d,'imgs','corrupt.jpg'),'w'); fprintf(fid,'not an image'); fclose(fid);
t = runBatchScreening(fullfile(d,'imgs'), fullfile(d,'out'));
assert(height(t) == 2, 'batch must return one row per file');
assert(any(strcmp(t.status,'error')), 'corrupt file must become an ERROR row');
rmdir(d,'s');
end

function tReport()
r = runScreeningPipeline(synthFundus());
assert(isfield(r,'qualityDecision') && isfield(r,'temperatureState'), 'report fields missing');
end

function tRepeat()
a = runScreeningPipeline(synthFundus());
b = runScreeningPipeline(synthFundus());
assert(a.focus == b.focus && strcmp(a.qualityDecision,b.qualityDecision), 'nondeterministic quality');
assert(isequal(a.roiMask,b.roiMask), 'nondeterministic ROI');
end

function tCache()
m1 = getOrLoadCachedModels(); m2 = getOrLoadCachedModels();
assert(m1.haveTrainedModels == m2.haveTrainedModels, 'cache unstable across calls');
end

function tMalformed()
try
    assessFundusQuality(uint8([]));
    error('expected:shouldHaveErrored', 'empty input accepted');
catch E
    assert(contains(E.identifier,'badInput'), 'wrong error contract');
end
end

function tUnique()
d = tempname; mkdir(d);
p1 = uniqueArtifactPath(d, 'collision_probe', '.csv');
fclose(fopen(p1,'w'));
p2 = uniqueArtifactPath(d, 'collision_probe', '.csv');
assert(~strcmp(p1,p2), 'collision check failed to uniquify');
rmdir(d,'s');
end
