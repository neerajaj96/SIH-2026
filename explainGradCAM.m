function [mapOrig, gradReport] = explainGradCAM(net, dlX, targetIdx, roiMaskOrig, lesionMask, odCenter, odRadius, varargin)
% explainGradCAM: CANONICAL Grad-CAM explanation builder (Stage-5).
% Single implementation consumed by runScreeningPipeline.m; no second
% Grad-CAM computation/interpretation path may exist elsewhere.
%
% What it guarantees over a raw gradCAM call:
%  - FeatureLayer validated against the actual graph when configured;
%    unknown names return UNAVAILABLE/INVALID - never a silent
%    substitution. "" (default) = framework auto-select, recorded as
%    'auto' in provenance (MATLAB-GATED: pin the exact DenseNet name in
%    explainabilityConfig once confirmed via net.Layers).
%  - Map resized to ORIGINAL image coordinates, then ROI-masked BEFORE
%    any peak/region measurement (border peaks cannot dominate).
%  - Dominance decided by activation-REGION mass fractions (not the
%    single peak pixel): border band + disc zone + lesion overlap are all
%    measured on the thresholded active region, deterministically.
%  - Lesion evidence (lesionMask) is an explicit input so disc-dominance
%    is evaluated against lesions elsewhere, not in a vacuum.
%  - Reliability states VALID / DEGRADED / UNAVAILABLE / INVALID with
%    reasons; NaN/degenerate inputs give INVALID, never a plausible map.
%
% INPUTS:
%   net         - trained dlnetwork (DenseNet-121 fusion)
%   dlX         - dlarray fusion input, 'SSC' (same tensor softmax used)
%   targetIdx   - 1-based class index explained (caller's predicted class)
%   roiMaskOrig - logical ROI at ORIGINAL resolution
%   lesionMask  - logical lesion (mahe|exudate) at ORIGINAL resolution
%   odCenter    - [x y] (NaN => INVALID)
%   odRadius    - pixels, >0 (else INVALID)
%   'Config'    - explainabilityConfig (default fresh)
%
% OUTPUTS:
%   mapOrig    - double activation map at ORIGINAL resolution, ROI-masked
%                (background 0), normalized to [0,1]; [] when INVALID/
%                UNAVAILABLE
%   gradReport - struct(status, peakInROI, peakInDiscZone, borderMassFrac,
%                discMassFrac, lesionOverlapFrac, activeFrac, targetClass,
%                reductionLayer, featureLayerUsed, reasons{}, provenance)
%
% Requires: Deep Learning Toolbox (gradCAM) + Image Processing Toolbox.

p = inputParser;
addParameter(p, 'Config', explainabilityConfig(), @isstruct);
parse(p, varargin{:});
cfg = p.Results.Config;

bad = @(c, r) any(isnan(c(:))) || ~isfinite(r) || r <= 0;
origSize = size(roiMaskOrig, [1 2]);
if ~isequal(size(lesionMask, [1 2]), origSize)
    mapOrig = [];
    gradReport = localReport(cfg.explainInvalid, false, false, NaN, NaN, NaN, NaN, targetIdx, cfg, ...
        {'lesionMask/ROI size mismatch - caller passed misaligned resolutions'}, 'size-mismatch');
    return;
end
if bad(odCenter, odRadius)
    mapOrig = [];
    gradReport = localReport(cfg.explainInvalid, false, false, NaN, NaN, NaN, NaN, targetIdx, cfg, ...
        {'unusable OD geometry (NaN/nonpositive) - disc zone undefined'}, 'bad-geometry');
    return;
end

% --- Feature-layer validation (never substitute silently) ---
featureUsed = 'auto';
if ~isempty(cfg.featureLayer)
    try
        layerNames = string({net.Layers.Name});
    catch
        layerNames = strings(0);
    end
    if ~any(layerNames == string(cfg.featureLayer))
        mapOrig = [];
        gradReport = localReport(cfg.explainUnavailable, false, false, NaN, NaN, NaN, NaN, targetIdx, cfg, ...
            {sprintf('configured FeatureLayer "%s" not in model graph - refusing to substitute', cfg.featureLayer)}, 'unknown-layer');
        return;
    end
    featureUsed = cfg.featureLayer;
end

% --- Activation (any gradCAM failure => INVALID, not a plausible map) ---
try
    if isempty(cfg.featureLayer)
        rawMap = extractdata(gradCAM(net, dlX, targetIdx, 'ReductionLayer', cfg.reductionLayer));
    else
        rawMap = extractdata(gradCAM(net, dlX, targetIdx, 'ReductionLayer', cfg.reductionLayer, 'FeatureLayer', cfg.featureLayer));
    end
catch ME
    mapOrig = [];
    gradReport = localReport(cfg.explainInvalid, false, false, NaN, NaN, NaN, NaN, targetIdx, cfg, ...
        {sprintf('gradCAM failed (%s) - wrong architecture/model object or unavailable gradients', ME.message)}, 'gradcam-error');
    return;
end
if isempty(rawMap) || ~any(isfinite(rawMap(:))) || max(rawMap(:)) <= 0
    mapOrig = [];
    gradReport = localReport(cfg.explainUnavailable, false, false, NaN, NaN, NaN, 0, targetIdx, cfg, ...
        {'empty/near-empty activation map - no explanation available'}, 'empty-map');
    return;
end

% --- Back to original coordinates, THEN mask (border cannot dominate) ---
mapFull = imresize(rawMap, origSize, 'bilinear');
mapFull(~roiMaskOrig) = 0;
mx = max(mapFull(:));
if ~(mx > 0)
    mapOrig = [];
    gradReport = localReport(cfg.explainUnavailable, false, false, NaN, NaN, NaN, 0, targetIdx, cfg, ...
        {'activation vanishes inside ROI - explanation outside retina only'}, 'outside-roi');
    return;
end
mapOrig = mapFull / mx; % [0,1]

% --- Deterministic activation-region mathematics ---
[H, W] = size(mapOrig);
[xx, yy] = meshgrid(1:W, 1:H);
active = mapOrig >= cfg.highFrac; % normalized map: active region = above fraction of max
activeFrac = sum(active(:)) / max(nnz(roiMaskOrig), 1);
band = round(cfg.borderBandFrac * min(H, W));
borderZone = xx <= band | xx > W-band | yy <= band | yy > H-band;
borderMass = sum(mapOrig(active & borderZone));
totalMass = sum(mapOrig(active));
borderMassFrac = borderMass / max(totalMass, eps);
discZone = hypot(xx-odCenter(1), yy-odCenter(2)) < cfg.discZoneRadii*odRadius;
discMassFrac = sum(mapOrig(active & discZone)) / max(totalMass, eps);
lesionMass = sum(mapOrig(active & logical(lesionMask)));
lesionOverlapFrac = lesionMass / max(totalMass, eps);
[~, peakIdx] = max(mapOrig(:));
[peakY, peakX] = ind2sub(size(mapOrig), peakIdx);
peakInROI = roiMaskOrig(peakY, peakX);
peakInDiscZone = hypot(peakX-odCenter(1), peakY-odCenter(2)) < cfg.discZoneRadii*odRadius;

reasons = {};
status = cfg.explainValid;
if borderMassFrac > cfg.borderMassFrac
    status = cfg.explainDegraded;
    reasons{end+1} = sprintf('border-dominated: %.0f%% of active mass in outer frame band', 100*borderMassFrac);
end
if discMassFrac > cfg.discMassFrac && lesionOverlapFrac < cfg.discMassFrac
    status = cfg.explainDegraded;
    reasons{end+1} = sprintf(['disc-dominated: %.0f%% of active mass in disc zone with only %.0f%% on lesions - ' ...
        'possible spurious attention, recommend manual review'], 100*discMassFrac, 100*lesionOverlapFrac);
end
gradReport = localReport(status, peakInROI, peakInDiscZone, borderMassFrac, discMassFrac, ...
    lesionOverlapFrac, activeFrac, targetIdx, cfg, reasons, 'measured');
gradReport.peakXY = [peakX, peakY];
end

function rep = localReport(status, inROI, inDisc, borderF, discF, lesF, actF, tgt, cfg, reasons, prov)
% E. Photo-vs-derived-mask circularity caveat (always attached): the
% grader input is RGB + vessel + lesion channels, so a 2D overlay cannot
% prove highlighted areas come from photographic retinal evidence rather
% than derived-mask influence. No per-channel attribution is claimed.
rep = struct('status', status, 'peakInROI', inROI, 'peakInDiscZone', inDisc, ...
    'borderMassFrac', borderF, 'discMassFrac', discF, 'lesionOverlapFrac', lesF, ...
    'activeFrac', actF, 'targetClass', tgt, 'reductionLayer', cfg.reductionLayer, ...
    'featureLayerUsed', cfg.featureLayer, 'reasons', {reasons}, 'provenance', prov, ...
    'inputChannels', {{'R','G','B','vessel','lesion'}}, ...
    'channelsCaveat', ['Attribution is computed from the fused RGB + derived segmentation channels ' ...
        'and cannot by itself distinguish photographic evidence from derived-mask influence.'], ...
    'configVersion', cfg.version);
if isempty(rep.featureLayerUsed), rep.featureLayerUsed = 'auto'; end
end
