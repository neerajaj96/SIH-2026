function testClinicalReasoning()
% testClinicalReasoning: Stage-4 MATLAB harness (synthetic, no data).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB/Octave here) ***
% Run in MATLAB: >> testClinicalReasoning
% Executable twin: python3 tests/test_stage4_clinical.py (59/59 PASS here).
% TRUTH-TABLE results below are LOGIC tests, never clinical validation.
%
% Covers: config centralization, speckle guard (0/10/21 1px dots),
% insufficient-evidence on production-shape input, PROXY NV wording,
% landmark validity states, quadrant INVALID blocking, determinism.

nP = 0; nT = 0;
[nP,nT] = cc(nP,nT,'config single source',@cConfig);
[nP,nT] = cc(nP,nT,'speckle guard blocks 1px severe fabrication',@cSpeckle);
[nP,nT] = cc(nP,nT,'insufficient evidence on UNAVAILABLE',@cInsuff);
[nP,nT] = cc(nP,nT,'NV proxy wording + INVALID gates',@cNV);
[nP,nT] = cc(nP,nT,'landmark/quadrant validity',@cLandmark);
[nP,nT] = cc(nP,nT,'legacy 6-output wrapper delegates',@cCompat);
fprintf('\n=== stage4 clinical (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testClinicalReasoning:failures','%d failed',nT-nP); end
end

function [p,t] = cc(p,t,name,f)
t=t+1; fprintf('[s4 %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function cConfig()
cfg = clinicalConfig();
assert(cfg.connectivity == 8 && cfg.minBlobAreaPx == 6, 'speckle policy drift');
assert(cfg.severeHemPerQuadrant == 20, '4-threshold drift');
assert(strcmp(cfg.vbStatusDefault,'UNAVAILABLE') && strcmp(cfg.irmaStatusDefault,'UNAVAILABLE'), 'VB/IRMA must default UNAVAILABLE');
assert(~cfg.allowFallbackQuadrants, 'fallback policy must default conservative');
end

function cSpeckle()
H = 40; W = 40; q = ones(H,W,'uint8');
q(1:20,1:20)=1; q(1:20,21:40)=2; q(21:40,1:20)=3; q(21:40,21:40)=4;
info = struct('maPresent',false,'exudatePresent',false,'venousBeadingQuadrants',0,'venousBeadingStatus','VERIFIED', ...
    'irmaQuadrants',0,'irmaStatus','VERIFIED','neovascularization',false,'neovascularizationStatus','NOT_DETECTED', ...
    'vitreousHemorrhage',false,'vitreousStatus','VERIFIED');
for n = [0 10 21]
    dots = false(H,W);
    for i = 1:n
        dots(mod(3*i,H)+1, mod(7*i,W)+1) = true; % isolated 1px speckles
    end
    [g, ~, rep] = assignClinicalGrade(dots, q, info);
    assert(~(g == 3 && strcmp(rep.trigger,'severe-trigger')), sprintf('%d 1px speckles must not fire Severe', n));
end
end

function cInsuff()
H = 40; W = 40; q = ones(H,W,'uint8')*2;
prodInfo = struct('maPresent',true,'exudatePresent',false,'venousBeadingQuadrants',0, ...
    'irmaQuadrants',0,'neovascularization',false,'vitreousHemorrhage',false);
[g, ev, rep] = assignClinicalGrade(false(H,W), q, prodInfo);
assert(isnan(g) && strcmp(rep.status,'INSUFFICIENT_EVIDENCE'), 'production input must be insufficient, not 0-4');
assert(any(contains(ev,'INCOMPLETE')), 'evidence must state incompleteness');
end

function cNV()
odC = [50 50]; odR = 5;
[flag, ~, ~, rep] = detectNeovascularization(false(100,100), odC, odR);
assert(~flag && strcmp(rep.status,'INVALID'), 'empty vessel mask must be INVALID, not negative');
[flag2, ~, ~, rep2] = detectNeovascularization(true(100,100), [NaN NaN], odR);
assert(~flag2 && strcmp(rep2.status,'INVALID'), 'NaN OD geometry must be INVALID');
end

function cLandmark()
[imgOut, mOut] = deal(zeros(32,32,3,'uint8'), false(32,32));
[~, ~, ~, lm] = localizeOpticDiscFovea(imgOut, mOut, []);
assert(isfield(lm,'odValidity') && isfield(lm,'foveaValidity'), 'validity struct missing');
[qm, qv] = partitionQuadrants([32 32], [NaN NaN], [NaN NaN]);
assert(strcmp(qv,'INVALID') && ~any(qm(:)), 'NaN geometry must give INVALID zero quadrants');
[qm2, qv2] = partitionQuadrants([32 32], [16 16], [24 16], 'FALLBACK', 'CONFIDENT');
assert(strcmp(qv2,'INVALID'), 'FALLBACK must be blocked by default policy');
end

function cCompat()
img = uint8(128*ones(48,48,3));
[isG, eRGB, eGray, fs, es, roi] = assessAndEnhanceImage(img, 8, 3.5);
rep = assessFundusQuality(img, 'FocusThresh', 8, 'EntropyThresh', 3.5, 'CalibrationDir', []);
assert(isG == rep.legacyGradeable && isequal(eGray, rep.enhancedGray) && isequal(roi, rep.roiMask), 'wrapper/canonical diverged');
assert(isG == (fs >= 8 && es >= 3.5), 'legacy boolean semantics broken');
end
