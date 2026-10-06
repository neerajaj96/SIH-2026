function [imgOut, roiMaskOut] = preprocessFundusForSegmentation(rawImage, netInputSize)
% preprocessFundusForSegmentation: THE canonical image -> U-Net-input
% transformation. Called from BOTH train_UNet_Segmentation.m's datastore
% transform and (via runSegmentationNet.m) production_inference.m, so
% training and inference are guaranteed to see identically-processed
% pixels.
%
% WHY THIS FILE EXISTS - two separate bugs this closes:
%
%   Bug 1 (resolution mismatch, flagged by the last project review):
%   train_UNet_Segmentation.m declares imageSize = [512 512 1] for the
%   unet(...) call, but its imageDatastore was built directly from raw
%   dataset folders with NO resize step, and production_inference.m fed
%   enhancedGray straight into predictBinaryMask() at whatever resolution
%   assessAndEnhanceImage.m happened to output (1280x1280 for the sample
%   photo, 4288x2848 for IDRiD, 584x565 for DRIVE...). A network trained
%   at one spatial resolution and run at another doesn't error - it just
%   silently sees lesions at the wrong effective physical scale relative
%   to its receptive field, which degrades accuracy without ever raising
%   an exception.
%
%   Bug 2 (train/inference DISTRIBUTION mismatch - not previously
%   flagged, found while fixing Bug 1): production_inference.m feeds the
%   segmentation nets enhancedGray - the CLAHE'd, flat-fielded, denoised
%   output of assessAndEnhanceImage.m - but train_UNet_Segmentation.m's
%   imageDatastore read raw DRIVE/IDRiD images with NO enhancement at
%   all. A network trained on raw pixel statistics and deployed on
%   enhanced pixel statistics is a covariate-shift bug, not just a resize
%   bug, and arguably the more serious of the two: it changes the actual
%   pixel-value distribution the network learned to expect, not just the
%   sampling grid.
%
% Both are fixed the same way: ONE function both paths are required to call.
%
% NOTE ON QUALITY GATING: this function does NOT reject low-quality
% images (thresholds are passed as -Inf) - during TRAINING you generally
% want the network to see some degraded images too, since production
% traffic will include borderline-but-accepted cases. Curate which images
% enter the training set (via calibrateQualityThresholds.m / manual
% review) at the file-list stage, before they reach this function - don't
% rely on this function to do that filtering silently on every call.
%
% INPUTS:
%   rawImage     - RGB or grayscale fundus image, any resolution
%   netInputSize - [H W], default segmentationConfig.inputSize ([512 512]).
%                  MUST match the size passed to runSegmentationNet.m at
%                  inference - both default to the same config, so this
%                  holds unless a caller overrides one side explicitly
%                  (which logs a warning - see below).
%
% OUTPUTS:
%   imgOut     - enhanced grayscale image resized to netInputSize, single
%                precision, scaled to [0,1]
%   roiMaskOut - logical fundus-circle mask, resized to netInputSize
%                (nearest-neighbor - a mask is categorical, not
%                continuous, so bilinear/bicubic would invent meaningless
%                fractional boundary values)
%
% Requires: Image Processing Toolbox, and assessAndEnhanceImage.m on the path.

cfgDefault = segmentationConfig();
if nargin < 2 || isempty(netInputSize)
    netInputSize = cfgDefault.inputSize;
end

% -Inf thresholds: this call site NEVER rejects, it only enhances - see
% the NOTE above on why quality curation belongs at the file-list stage,
% not baked into this function.
[~, ~, enhancedGray, ~, ~, roiMask] = assessAndEnhanceImage(rawImage, -Inf, -Inf);

% Parity contract with runSegmentationNet.m (inference): image = bilinear,
% mask = nearest, scale = single/255. Both files read segmentationConfig;
% change inputSize there, not here. An explicit non-default netInputSize
% is honored (for the 512-vs-768 benchmark) but logged.
if ~isequal(netInputSize, cfgDefault.inputSize)
    fprintf(['preprocessFundusForSegmentation: non-default netInputSize [%d %d] ' ...
             '(config default [%d %d]) - ensure runSegmentationNet is called with the same size.\n'], ...
        netInputSize(1), netInputSize(2), cfgDefault.inputSize(1), cfgDefault.inputSize(2));
end
imgResized = imresize(enhancedGray, netInputSize, 'bilinear');
roiMaskOut = logical(imresize(roiMask, netInputSize, 'nearest'));
imgOut = single(imgResized) / 255;
end
