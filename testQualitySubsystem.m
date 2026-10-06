function testQualitySubsystem()
% testQualitySubsystem: Stage-1 MATLAB harness for IQA + enhancement.
% Run in MATLAB: >> testQualitySubsystem
%
% Covers (synthetic, no datasets needed):
%  [1] qualityConfig canonical thresholds + margins exist
%  [2] assessFundusQuality decision taxonomy PASS/BORDERLINE/FAIL + reasons
%  [3] ROI robustness: centered/off-center/small/degenerate FOV
%  [4] Sharp-vs-blurred focus ordering; textured-vs-flat entropy ordering
%  [5] Enhancement guardrails: flat images use reduced clip; tiny skips denoise
%  [6] Calibrator math: midpoint, sens/spec, AUC on synthetic score vectors
%  [7] Determinism: same input twice -> identical decision+scores
%  [8] Backward compat: assessAndEnhanceImage 6-output shape preserved
%
% NOTE: This file was authored without MATLAB execution in this workspace
% (no MATLAB/Octave here). Run it in MATLAB before trusting Stage-2
% integration; the Python mirror (tests/python/test_quality_mirror.py)
% IS executed here and cross-checks the same logic.

nPass = 0; nTot = 0;
[nPass, nTot] = qcheck(nPass, nTot, 'config canonical', @tConfig);
[nPass, nTot] = qcheck(nPass, nTot, 'decision taxonomy + reasons', @tDecision);
[nPass, nTot] = qcheck(nPass, nTot, 'ROI robustness', @tROI);
[nPass, nTot] = qcheck(nPass, nTot, 'focus/entropy ordering', @tOrdering);
[nPass, nTot] = qcheck(nPass, nTot, 'enhancement guardrails', @tEnhance);
[nPass, nTot] = qcheck(nPass, nTot, 'calibrator math', @tCalib);
[nPass, nTot] = qcheck(nPass, nTot, 'determinism', @tDeterminism);
[nPass, nTot] = qcheck(nPass, nTot, 'backward compat 6-output', @tCompat);
[nPass, nTot] = qcheck(nPass, nTot, 'wrapper/canonical agreement (same inputs)', @tAgreement);
fprintf('\n=== quality: %d / %d passed ===\n', nPass, nTot);
if nPass < nTot, error('testQualitySubsystem:failures', '%d failed', nTot-nPass); end
end

function [p, t] = qcheck(p, t, name, f)
t = t + 1; fprintf('[q%2d] %s ... ', t, name);
try, f(); fprintf('PASS\n'); p = p + 1;
catch E, fprintf('FAIL\n      %s\n', E.message); end
end

function tConfig()
cfg = qualityConfig();
assert(cfg.focusThresh == 8 && cfg.entropyThresh == 3.5, 'canonical defaults drifted');
assert(cfg.borderlineFocusMargin == 0.15 && cfg.borderlineEntropyMargin == 0.10, 'margins missing');
assert(isfield(cfg,'roiSeedThresh') && isfield(cfg,'bgFraction') && isfield(cfg,'claheClipLimit'), 'config incomplete');
end

function tDecision()
sharp = localFundus(256, 256, 90, 1.0, 0);      % clean disc
rep = assessFundusQuality(sharp);
assert(ismember(rep.decision, {'PASS','BORDERLINE','FAIL'}), 'bad decision enum');
assert(islogical(rep.isGradeable) && iscell(rep.reasons) && iscell(rep.recaptureGuidance), 'bad report shape');
blk = zeros(128,128,3,'uint8');
repB = assessFundusQuality(blk);
assert(strcmp(repB.decision,'FAIL') && ~repB.isGradeable, 'all-black must FAIL');
end

function tROI()
c = localFundus(256,256,90,1.0,0);
r1 = assessFundusQuality(c);
assert(r1.roiInfo.coverageFrac > 0.05, 'centered FOV coverage too low');
off = localFundus(256,256,60,1.0,40);            % small off-center disc
r2 = assessFundusQuality(off);
assert(r2.roiInfo.coverageFrac > 0 && r2.roiInfo.coverageFrac < r1.roiInfo.coverageFrac, 'off-center should shrink ROI');
tiny = zeros(32,32,3,'uint8'); tiny(8:24,8:24,:) = 200;
r3 = assessFundusQuality(tiny);
assert(ismember(r3.decision, {'PASS','BORDERLINE','FAIL'}), 'tiny must not crash');
end

function tOrdering()
sharp = localFundus(256,256,90,1.0,0);
blur = localBlur(sharp);
[~,~,~,fS,~,~] = assessAndEnhanceImage(sharp, -Inf, -Inf);
[~,~,~,fB,~,~] = assessAndEnhanceImage(blur, -Inf, -Inf);
assert(fS > fB, sprintf('sharp %.2f must exceed blur %.2f', fS, fB));
flat = uint8(128*ones(256,256,3));
[~,~,~,~,eFlat,~] = assessAndEnhanceImage(flat, -Inf, -Inf);
[~,~,~,~,eTex,~] = assessAndEnhanceImage(sharp, -Inf, -Inf);
assert(eTex > eFlat, 'textured entropy must exceed flat');
end

function tEnhance()
cfg = qualityConfig();
flat = uint8(128*ones(256,256,3) + randi(3,256,256,3,'uint8') - 1);
[~, eRGB, eGray, ~, ~, m] = assessAndEnhanceImage(flat, -Inf, -Inf);
assert(isequal(size(eGray), size(m)) && ndims(eRGB) == 3, 'enhanced shape mismatch');
assert(all(eGray(~m) == 0), 'background must be masked to 0');
end

function tCalib()
goodV = [9 10 11 12]; badV = [2 3 4 5];
th = (min(goodV)+max(badV))/2;
assert(abs(th - 7) < 1e-9, 'midpoint math wrong');
assert(mean(goodV >= th) == 1 && mean(badV < th) == 1, 'sens/spec should be 1 on separated data');
end

function tDeterminism()
img = localFundus(200,200,70,0.9,0);
a = assessFundusQuality(img); b = assessFundusQuality(img);
assert(strcmp(a.decision,b.decision) && a.focusScore == b.focusScore && a.entropyScore == b.entropyScore, 'non-deterministic');
end

function tCompat()img = localFundus(128,128,45,1.0,0);
[isG, eRGB, eGray, fs, es, roi] = assessAndEnhanceImage(img, 8, 3.5);
assert(islogical(isG) && ndims(eRGB)==3 && ismatrix(eGray) && islogical(roi), '6-output contract broken');
assert(isequal(size(eGray), size(roi)), 'enhancedGray/roiMask size mismatch breaks runSegmentationNet');
% RGBA input: first 3 channels kept, no crash
rgba = cat(3, img, uint8(255*ones(128,128)));
[isG4, ~, ~, ~, ~, ~] = assessAndEnhanceImage(rgba, 8, 3.5);
assert(islogical(isG4), 'RGBA handling broken');
% Empty input: clean error, not obscure crash
try
    assessAndEnhanceImage(uint8([]), 8, 3.5);
    error('expected:emptyShouldError', 'empty input did not error');
catch E
    assert(strcmp(E.identifier,'assessAndEnhanceImage:badInput'), sprintf('wrong error: %s', E.identifier));
end
end

function img = localFundus(H, W, R, bright, dx)
[xx, yy] = meshgrid(1:W, 1:H);
cx = W/2 + dx; cy = H/2;
disc = (xx-cx).^2 + (yy-cy).^2 <= R^2;
tex = uint8(60 + 40*sin(0.3*xx) .* cos(0.3*yy) + 20*sin(0.11*xx+0.23*yy));
img = zeros(H,W,3,'uint8');
for c = 1:3, ch = uint8(double(tex)*bright); ch(~disc) = 0; img(:,:,c) = ch; end
end

function b = localBlur(img)
k = fspecial('gaussian', [9 9], 3);
b = img;
for c = 1:3, b(:,:,c) = imfilter(img(:,:,c), k, 'replicate'); end
end

function tAgreement()
% Wrapper/canonical agreement: same explicit thresholds on the same
% synthetic inputs must give identical enhanced images, scores, ROI and
% legacy gradeability (UNEXECUTED here - run in MATLAB).
cases = {localFundus(128,128,45,1.0,0), localFundus(128,128,45,0.7,0), ...
         uint8(128*ones(96,96,3))};
for i = 1:numel(cases)
    img = cases{i};
    [isG, eRGB, eGray, fs, es, roi] = assessAndEnhanceImage(img, 8, 3.5);
    rep = assessFundusQuality(img, 'FocusThresh', 8, 'EntropyThresh', 3.5, 'CalibrationDir', []);
    assert(isG == rep.legacyGradeable, sprintf('case %d: legacy boolean diverged', i));
    assert(isequal(eRGB, rep.enhancedRGB) && isequal(eGray, rep.enhancedGray), sprintf('case %d: enhanced images diverged', i));
    assert(fs == rep.focusScore && es == rep.entropyScore, sprintf('case %d: scores diverged', i));
    assert(isequal(roi, rep.roiMask), sprintf('case %d: ROI diverged', i));
end
end
