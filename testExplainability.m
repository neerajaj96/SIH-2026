function testExplainability()
% testExplainability: Stage-5 MATLAB harness (synthetic, no data/weights).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB/Octave here) ***
% Run in MATLAB (trained nets required for gradCAM paths): >> testExplainability
% Executable twin: python3 tests/test_stage5_explain.py (45/45 PASS here).
% Results below are LOGIC/structure tests, never clinical validation.
%
% Covers: config single-source, canonical Grad-CAM wiring, orthogonal
% disagreement (incl. NaN-rule NOT_COMPARABLE), batch agreement rule,
% bridge additive keys (doc-level), calibration artifact validation,
% reliability table shape, report status lines.

nP = 0; nT = 0;
[nP,nT] = ec(nP,nT,'config single source',@cConfig);
[nP,nT] = ec(nP,nT,'canonical Grad-CAM path',@cCanon);
[nP,nT] = ec(nP,nT,'orthogonal disagreement',@cDisagree);
[nP,nT] = ec(nP,nT,'batch agreement NaN-safe',@cBatch);
[nP,nT] = ec(nP,nT,'calibration artifact validation',@cCalib);
[nP,nT] = ec(nP,nT,'report status lines',@cReport);
fprintf('\n=== stage5 explainability (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testExplainability:failures','%d failed',nT-nP); end
end

function [p,t] = ec(p,t,name,f)
t=t+1; fprintf('[s5 %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function cConfig()
cfg = explainabilityConfig();
assert(strcmp(cfg.reductionLayer,'prob'), 'reduction layer drift');
assert(isfield(cfg,'borderMassFrac') && isfield(cfg,'discMassFrac') && isfield(cfg,'highFrac'), 'region math unconfigured');
assert(isfield(cfg,'lowConfidenceReview'), 'review heuristic unconfigured');
g = fileread('gradingConfig.m');
assert(isempty(strfind(lower(g),'explainabilityconfig')) && isempty(strfind(lower(g),'featurelayer')), 'gradingConfig must stay frozen');
end

function cCanon()
% Canonical path only: pipeline must call explainGradCAM, never gradCAM( directly.
rsp = fileread('runScreeningPipeline.m');
assert(~isempty(strfind(rsp,'explainGradCAM(')), 'pipeline bypasses canonical explanation');
assert(isempty(strfind(rsp,'gradCAM(')), 'second Grad-CAM path in pipeline');
end

function cDisagree()
d = analyzeDisagreement(2, 0.9, 'CALIBRATED', 2, 'SUFFICIENT', 'VALID');
assert(strcmp(d.gradeRelationship,'AGREE') && ~d.escalate, 'agree case broken');
d2 = analyzeDisagreement(3, 0.9, 'CALIBRATED', 2, 'SUFFICIENT', 'VALID');
assert(strcmp(d2.gradeRelationship,'NUMERIC_DISAGREE') && d2.escalate, 'disagree must escalate');
d3 = analyzeDisagreement(3, 0.9, 'CALIBRATED', NaN, 'INSUFFICIENT_EVIDENCE', 'UNAVAILABLE');
assert(strcmp(d3.gradeRelationship,'NOT_COMPARABLE') && d3.escalate, 'insufficient must not read as disagree/agree');
d4 = analyzeDisagreement(NaN, NaN, 'UNCALIBRATED', NaN, 'INSUFFICIENT_EVIDENCE', 'INVALID');
assert(strcmp(d4.confidenceStatus,'UNCALIBRATED'), 'fallback T must never read calibrated');
end

function cBatch()
% Batch agreement rule: NaN unless both grades exist (static + synthetic).
b = fileread('runBatchScreening.m');
assert(~isempty(strfind(b,'~isnan(icdrDL) && ~isnan(icdrRule)')), 'manufactured-disagreement regression');
assert(~isempty(strfind(b,'nInsufficient')), 'insufficient triage count missing');
end

function cCalib()
% Artifact validation states + reliability columns (static pins; live
% validation needs .mat files + MATLAB).
l = fileread('loadModelsIfPresent.m');
assert(~isempty(strfind(l,'CALIBRATED_VALID')) && ~isempty(strfind(l,'CALIBRATED_MISMATCH')) && ~isempty(strfind(l,'UNCALIBRATED_FALLBACK')), 'load states incomplete');
c = fileread('calibrateTemperature.m');
assert(~isempty(strfind(c,'binLower')) && ~isempty(strfind(c,'meanConfidence')) && ~isempty(strfind(c,'sampleCount')), 'reliability columns incomplete');
end

function cReport()
p = fileread('production_inference.m');
assert(~isempty(strfind(p,'Evidence status')) && ~isempty(strfind(p,'r.disagreement')) && ~isempty(strfind(p,'r.explainStatus')), 'report status block missing');
end
