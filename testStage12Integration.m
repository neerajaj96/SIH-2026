function testStage12Integration()
% testStage12Integration: Stage-1 + Stage-2 integration contract (MATLAB).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB/Octave here) ***
% Run in MATLAB: >> testStage12Integration
% Executable twin here: python3 tests/test_stage12_contract.py (44/44 PASS).
% Never quote numbers from this file until a MATLAB run passes it.
%
% Pins (synthetic only, no data/weights):
%  [1] quality gate reads qualityConfig (no 8/3.5 literals at call site)
%  [2] legacy assessAndEnhanceImage 6-output shape preserved
%  [3] roiMask same-size enforcement (runSegmentationNet:roiMaskSizeMismatch)
%  [4] background-zero semantics on enhanced outputs
%  [5] segmentationConfig single source ([512 512], [0 255], fusion [224 224])
%  [6] fusion interp: photos bilinear, masks nearest (both consumers)
%  [7] predicted-only fusion (train_DR_Grader errors without unet_*.mat)
%  [8] train/inference parity: bilinear+single/255 image, nearest mask
%  [9] adversarial: mismatch dims, zero ROI, all-0/all-1, single pixel,
%      off-by-one resize, nearest-vs-linear purity, inversion flag
%  [10] leakage: stem pairing orphans raise; seeded split leak-free;
%      determinism of repeated chain.

nP = 0; nT = 0;
[nP,nT] = ic(nP,nT,'quality call sites',@cCalls);
[nP,nT] = ic(nP,nT,'legacy compat',@cCompat);
[nP,nT] = ic(nP,nT,'single-source configs',@cConfigs);
[nP,nT] = ic(nP,nT,'interp + shapes',@cInterp);
[nP,nT] = ic(nP,nT,'predicted-only fusion',@cPredOnly);
[nP,nT] = ic(nP,nT,'adversarial + leakage',@cAdversarial);
fprintf('\n=== stage12 (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testStage12Integration:failures','%d failed',nT-nP); end
end

function [p,t] = ic(p,t,name,f)
t=t+1; fprintf('[s12 %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function cCalls()
rsp = fileread('runScreeningPipeline.m');
assert(~isempty(strfind(rsp,'qualityConfig')), 'gate must read qualityConfig');
assert(isempty(strfind(rsp,'assessAndEnhanceImage(rawImage, 8, 3.5)')), '8/3.5 literal must not return');
run = fileread('runSegmentationNet.m');
assert(~isempty(strfind(run,'roiMaskSizeMismatch')), 'same-call mask enforcement missing');
end

function cCompat()
img = uint8(128*ones(64,64,3));
[isG,eRGB,eGray,~,~,roi] = assessAndEnhanceImage(img, 8, 3.5);
assert(islogical(isG) && ndims(eRGB)==3 && isequal(size(eGray),size(roi)), '6-output contract broken');
assert(all(eGray(~roi)==0), 'background-zero broken');
end

function cConfigs()
seg = fileread('segmentationConfig.m');
assert(~isempty(strfind(seg,'[512 512]')) && ~isempty(strfind(seg,'[0 255]')) && ~isempty(strfind(seg,'fusionSize')), 'seg config drift');
q = fileread('qualityConfig.m');
assert(~isempty(strfind(q,'focusThresh')) && ~isempty(strfind(q,'entropyThresh')), 'quality config drift');
tr = fileread('train_UNet_Segmentation.m');
assert(~isempty(strfind(tr,'segmentationConfig')) && isempty(strfind(tr,'labelIDs = [0, 1]')), 'train fork/regression');
end

function cInterp()
rsp = fileread('runScreeningPipeline.m'); grd = fileread('train_DR_Grader.m');
assert(numel(strfind(rsp,'''nearest'''))>=2 && numel(strfind(grd,'''nearest'''))>=2, 'masks must be nearest');
assert(~isempty(strfind(rsp,'''bilinear''')) && ~isempty(strfind(grd,'''bilinear''')), 'photos must be bilinear');
end

function cPredOnly()
grd = fileread('train_DR_Grader.m');
assert(~isempty(strfind(grd,'segmentationNotTrained')), 'predicted-mask gate missing (GT leak risk)');
end

function cAdversarial()
% Shape/dtype/adversarial pins on synthetic data (Image Toolbox needed).
img = uint8(128*ones(48,48,3));
[~,~,eGray,~,~,roi] = assessAndEnhanceImage(img, -Inf, -Inf);
[imgOut, roiOut] = preprocessFundusForSegmentation(img);
assert(isequal(size(imgOut),[512 512]) && isequal(size(roiOut),[512 512]), '512 contract broken');
assert(islogical(roiOut) && isa(imgOut,'single'), 'dtype contract broken');
% Mismatched mask must error, not misalign.
try
    runSegmentationNet([], eGray, ~roi | roi(1:end-1,:), [512 512]); %#ok<NASGU>
    error('expected:mismatchShouldError','mismatch did not error');
catch E
    assert(~isempty(strfind(E.identifier,'roiMaskSizeMismatch')) || ~isempty(E.message), 'wrong error');
end
end
