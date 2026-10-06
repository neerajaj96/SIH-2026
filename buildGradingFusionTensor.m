function fused = buildGradingFusionTensor(enhancedRGB, vesselMask, maheMask, exudateMask, targetSize)
% buildGradingFusionTensor: CANONICAL 5-channel fusion builder for
% TRAINING (Stage-3 P0). Byte-identical semantics to the inference path in
% runScreeningPipeline.m:103-108, which is deliberately NOT rewired to call
% this (shared orchestration stays untouched this stage - see header note).
% If you change one, change the other and update tests/test_grading_mirror.py.
%
% CHANNEL ORDER (frozen, see gradingConfig.m):
%   1-3 enhanced R,G,B (bilinear) | 4 vessel*255 (nearest) | 5 lesion*255
%   (nearest), lesion = maheMask | exudateMask (Stage-2 contract).
%
% MASKS MUST BE PREDICTED segmentation outputs, never ground-truth masks.
% This function cannot tell predicted from GT pixels - the guarantee comes
% from the caller (train_DR_Grader.m builds masks via runSegmentationNet on
% trained U-Nets and errors if the .mat files are missing). Passing GT
% masks here would let the grader cheat on features inference never has.
%
% LOUD INVALID DETECTION (this errors instead of training on garbage):
%   - enhancedRGB not HxWx3 uint8/single, or NaN/Inf present
%   - any mask not HxW logical (or binary numeric 0/1, 0/255, true/false)
%   - mask size != enhancedRGB size (caller passed mismatched resolutions)
%   - fractional mask values after *255 (indicates a bicubic-resize leak
%     upstream - masks must stay categorical via nearest)
%   - empty ROI (all-zero enhancedRGB) warns (degenerate input, not fatal)
%
% RAW-BYPASS NOTE: enhanced vs raw RGB cannot be distinguished pixel-wise
% here. The no-raw guarantee comes from the caller passing assessAndEnhanceImage
% output (train_DR_Grader.buildFusionTensor uses -Inf thresholds = enhance,
% never reject; runScreeningPipeline uses the gated enhanced image). Do not
% pass imread() output directly - that silently trains on unenhanced pixels
% while inference sees enhanced ones (covariate shift). The deliberate
% baseline ablation (train_Baseline_ResNet50.m raw images) is the ONLY
% sanctioned raw path, and it never calls this function.
%
% INPUTS:
%   enhancedRGB - HxWx3 uint8 (or single 0-255), enhanced (not raw)
%   vesselMask, maheMask, exudateMask - HxW logical (or binary numeric)
%   targetSize - [H W], default gradingConfig.inputSize ([224 224])
%
% OUTPUT:
%   fused - single [H W 5]: ch1-3 photo 0-255, ch4-5 masks {0,255}
%
% Requires: Image Processing Toolbox (imresize).

if nargin < 5 || isempty(targetSize)
    g = gradingConfig();
    targetSize = g.inputSize;
end
g = gradingConfig();

% --- Validate enhancedRGB ---
assert(ndims(enhancedRGB) == 3 && size(enhancedRGB, 3) == 3, ...
    'buildGradingFusionTensor:badRGB - enhancedRGB must be HxWx3, got %s.', mat2str(size(enhancedRGB)));
dRGB = double(enhancedRGB);
assert(all(isfinite(dRGB(:))), 'buildGradingFusionTensor:nanInf - enhancedRGB contains NaN/Inf.');
assert(~isempty(dRGB), 'buildGradingFusionTensor:empty - enhancedRGB is empty.');
if ~any(dRGB(:) ~= 0)
    warning(['buildGradingFusionTensor:allZero - enhancedRGB is all zeros (empty ROI or degenerate input). ' ...
             'Proceeding, but inspect the image - downstream landmarks will fall back.']);
end
if max(dRGB(:)) <= 1 && min(dRGB(:)) >= 0
    warning(['buildGradingFusionTensor:range01 - enhancedRGB max <= 1. Expected 0-255 single/uint8. ' ...
             'If you passed a 0-1 normalized image, channels 1-3 will be ~0 and the pretrained conv filters ' ...
             'will see out-of-distribution input. Pass 0-255 (this function does NOT rescale photos).']);
end

% --- Validate + normalize masks (logical canonical) ---
[vesselMask, vesselSrc] = localAsLogical(vesselMask, 'vesselMask');
[maheMask, maheSrc] = localAsLogical(maheMask, 'maheMask');
[exMask, exSrc] = localAsLogical(exudateMask, 'exudateMask'); %#ok<ASGLU>
sz = size(enhancedRGB, [1 2]);
for k = 1:3
    mk = {vesselMask, maheMask, exMask}; nm = {'vesselMask', 'maheMask', 'exudateMask'};
    assert(isequal(size(mk{k}, [1 2]), sz), ...
        ['buildGradingFusionTensor:sizeMismatch - %s is %s but enhancedRGB is %s. ' ...
         'Both must come from the same assessAndEnhanceImage call at the same resolution.'], ...
        nm{k}, mat2str(size(mk{k})), mat2str(sz));
end
lesionMask = maheMask | exMask; %#ok<NASGU>

% --- Resize with frozen interp semantics ---
photoSmall = imresize(enhancedRGB, targetSize, g.photoInterp);
vesselSmall = imresize(uint8(vesselMask) * 255, targetSize, g.maskInterp);
lesionSmall = imresize(uint8(lesionMask) * 255, targetSize, g.maskInterp);

% Fractional-value tripwire: nearest must yield exactly {0,255}.
for k = 1:2
    mk = {vesselSmall, lesionSmall}; nm = {'vessel', 'lesion'};
    u = unique(mk{k}(:))';
    if ~all(ismember(u, [0 255]))
        error(['buildGradingFusionTensor:nonBinaryMask - %s channel after %s resize contains %s, not {0,255}. ' ...
               'Masks are categorical - use nearest, never bicubic/bilinear.'], nm{k}, g.maskInterp, mat2str(u(1:min(9,numel(u)))));
    end
end

fused = single(cat(3, photoSmall, vesselSmall, lesionSmall));
assert(isequal(size(fused), [targetSize(1) targetSize(2) 5]), ...
    'buildGradingFusionTensor:shape - fused must be [H W 5], got %s.', mat2str(size(fused)));
assert(all(isfinite(fused(:))), 'buildGradingFusionTensor:nanInfOut - fused contains NaN/Inf.');
end

% ------------------------------------------------------------------
function [m, src] = localAsLogical(x, name)
% Accepts logical, binary double/single 0/1, or 0/255 uint8/double.
% Anything else (fractional, NaN/Inf, out-of-range) errors loudly.
if islogical(x)
    m = x; src = 'logical'; return;
end
assert(isnumeric(x) && isreal(x), ...
    'buildGradingFusionTensor:badMaskType - %s must be logical or binary numeric, got %s.', name, class(x));
assert(all(isfinite(x(:))), 'buildGradingFusionTensor:nanInf - %s contains NaN/Inf.', name);
u = unique(x(:))';
if all(ismember(u, [0 1]))
    m = logical(x); src = '0/1';
elseif all(ismember(u, [0 255]))
    m = logical(x ~= 0); src = '0/255';
elseif all(ismember(u, [false true]))
    m = logical(x); src = 'logical-numeric';
else
    error(['buildGradingFusionTensor:nonBinaryMask - %s has values %s. Masks must be binary ' ...
           '(logical, 0/1, or 0/255). Fractional values indicate a bicubic/bilinear resize leak - use nearest.'], ...
        name, mat2str(u(1:min(9,numel(u)))));
end
end
