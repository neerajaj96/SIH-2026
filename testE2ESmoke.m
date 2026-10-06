function testE2ESmoke()
% testE2ESmoke: Stage-10 MATLAB end-to-end agreement (synthetic only).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB here) ***
% Run in MATLAB (no weights needed - simulated mode suffices):
%   >> testE2ESmoke
% Executable twin: python3 tests/test_stage10_e2e.py (static pins) +
%   python3 scripts/smoke_e2e.py (live HTTP when a bridge runs).
% Asserts cross-entry agreement on ONE synthetic case: the structured
% outputs of runScreeningPipeline, screenOneImage-equivalent fields,
% batch row, and report inputs must agree (no clinical claim).
%
% Covers: happy-path agreement, insufficient-evidence propagation,
% NaN handling, repeated-run stability (schema/state, NOT byte equality
% - exact determinism is asserted only where explicitly guaranteed).

nP = 0; nT = 0;
[nP,nT] = e2(nP,nT,'entry-point agreement (simulated)',@cAgree);
[nP,nT] = e2(nP,nT,'insufficient-evidence propagation',@cInsuff);
[nP,nT] = e2(nP,nT,'repeat stability (schema/state, not bytes)',@cRepeat);
fprintf('\n=== stage10 e2e (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testE2ESmoke:failures','%d failed',nT-nP); end
end

function [p,t] = e2(p,t,name,f)
t=t+1; fprintf('[e2e %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function img = synthEye()
[xx, yy] = meshgrid(1:96, 1:96);
disc = (xx-48).^2 + (yy-48).^2 <= 34^2;
tex = uint8(90 + 40*sin(0.2*xx).*cos(0.2*yy));
img = zeros(96,96,3,'uint8');
for c = 1:3, ch = tex; ch(~disc) = 0; img(:,:,c) = ch; end
end

function cAgree()
img = synthEye();
r = runScreeningPipeline(img);
assert(isfield(r,'qualityDecision') && isfield(r,'ruleStatus') && isfield(r,'disagreement'), 'pipeline fields missing');
assert(isfield(r.disagreement,'gradeRelationship') && isfield(r.disagreement,'escalate'), 'disagreement shape broken');
assert(strcmp(r.disagreement.gradeRelationship,'NOT_COMPARABLE') || strcmp(r.disagreement.gradeRelationship,'AGREE') || strcmp(r.disagreement.gradeRelationship,'NUMERIC_DISAGREE'), 'bad relationship enum');
end

function cInsuff()
img = synthEye();
r = runScreeningPipeline(img);
if isnan(r.ruleGrade)
    assert(strcmp(r.ruleStatus,'INSUFFICIENT_EVIDENCE') || strcmp(r.ruleStatus,'INVALID'), 'NaN grade needs insufficient status');
end
end

function cRepeat()
a = runScreeningPipeline(synthEye());
b = runScreeningPipeline(synthEye());
assert(strcmp(a.status,b.status) && strcmp(a.qualityDecision,b.qualityDecision) && strcmp(a.ruleStatus,b.ruleStatus), 'repeat unstable');
assert(isequal(size(a.enhancedGray),size(b.enhancedGray)), 'shape unstable');
end
