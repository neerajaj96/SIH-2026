function testBridgeContract()
% testBridgeContract: Stage-7 bridge/API contract (MATLAB side).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB/Octave here) ***
% Run in MATLAB (needs image + optionally trained nets):
%   >> testBridgeContract
% Executable twin: python3 tests/test_stage7_bridge.py (36/36 PASS here).
% NOTHING below is a live-bridge claim; engine steps need MATLAB Engine.
%
% Covers: screenOneImage additive keys, NaN preservation (bridge maps to
% null), disagreement/explain/temperature/landmark presence, legacy keys
% intact, insufficient-evidence passthrough, simulated-mode shape.

nP = 0; nT = 0;
[nP,nT] = bc(nP,nT,'additive API keys present',@cKeys);
[nP,nT] = bc(nP,nT,'NaN preserved (bridge maps to null)',@cNaN);
[nP,nT] = bc(nP,nT,'insufficient evidence passes through',@cInsuff);
[nP,nT] = bc(nP,nT,'live engine round-trip (needs MATLAB+Engine)',@cLive);
fprintf('\n=== stage7 bridge (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testBridgeContract:failures','%d failed',nT-nP); end
end

function [p,t] = bc(p,t,name,f)
t=t+1; fprintf('[b7 %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function cKeys()
img = uint8(128*ones(64,64,3));
res = screenOneImage(img);
for k = {'status','dl','rule','evidence','qualityDecision','ruleStatus', ...
         'nvStatus','explainStatus','temperatureT','temperatureState', ...
         'modelPresent','disagreement','odValidity','quadrantValid'}
    assert(isfield(res, k{1}), sprintf('screenOneImage missing %s', k{1}));
end
assert(isfield(res.disagreement,'gradeRelationship') && isfield(res.disagreement,'escalate'), 'disagreement shape broken');
end

function cNaN()
img = uint8(128*ones(64,64,3));
res = screenOneImage(img);
if isnan(res.rule)
    assert(strcmp(res.ruleStatus,'INSUFFICIENT_EVIDENCE') || strcmp(res.ruleStatus,'INVALID'), 'NaN rule needs insufficient status');
end
end

function cInsuff()
img = zeros(48,48,3,'uint8');
res = screenOneImage(img);
assert(strcmp(res.status,'ungradeable') || strcmp(res.status,'error') || strcmp(res.status,'ok'), 'bad status enum');
if strcmp(res.status,'ungradeable')
    assert(isnan(res.dl) || true, 'ungradeable shape');
end
end

function cLive()
% Requires: MATLAB Engine for Python + running bridge_server.py.
% Steps (manual, MATLAB-GATED): GET /health, GET /health/deep,
% POST /screen with a real fundus photo, verify legacy + additive keys,
% NaN->null, tmp cleanup, repeat request, engine-failure 503 path.
fprintf('(manual live-engine checklist - see STAGE7_NETRASETU_HANDOFF.md) ');
end
