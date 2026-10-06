function [maskOrig, maskNet] = runSegmentationNet(net, enhancedGray, roiMask, netInputSize)
% runSegmentationNet: THE canonical inference wrapper for the three
% binary lesion/vessel U-Nets - resizes to the network's trained
% resolution, runs prediction, then resizes the mask BACK to the
% original image's coordinate space before returning it. Replaces the
% old predictBinaryMask() local function in production_inference.m,
% which fed enhancedGray to the network at whatever resolution it
% happened to be - see preprocessFundusForSegmentation.m's header for
% the full story of why that was a bug.
%
% WHY THE RESIZE-BACK MATTERS: every downstream consumer of these masks
% (partitionQuadrants.m, assignClinicalGrade.m, detectNeovascularization.m,
% localizeOpticDiscFovea.m) indexes pixels in the ORIGINAL image's
% coordinate system - odCenter, foveaCenter and quadrantMask are all
% computed at the resolution assessAndEnhanceImage.m produced. If this
% function returned the mask at netInputSize instead, every one of those
% downstream calls would silently misalign (a hemorrhage at pixel (800,
% 650) in the original image is NOT at pixel (800,650) in a 512x512
% resize of a 1280x1280 original). Resizing back here, once, in the one
% place segmentation happens, means nothing downstream has to know or
% care that the network internally runs at a fixed resolution.
%
% See preprocessFundusForSegmentation.m for the matching TRAINING-side
% function (same resize target, same enhancement path) - this function
% and that one must be changed together if the target resolution ever
% changes.
%
% INPUTS:
%   net           - trained (or placeholder) dlnetwork, binary U-Net,
%                   output convention: class 2 = "Foreground" (matches
%                   train_UNet_Segmentation.m's classNames ordering)
%   enhancedGray  - single-channel enhanced image, ORIGINAL resolution,
%                   from assessAndEnhanceImage.m
%   roiMask       - logical fundus-circle mask, SAME original resolution
%                   as enhancedGray (from the SAME assessAndEnhanceImage.m
%                   call - a size mismatch here is a caller bug and is
%                   checked below rather than silently misaligned)
%   netInputSize  - [H W] the network was trained at (default
%                   segmentationConfig.inputSize, currently [512 512] -
%                   both this file and preprocessFundusForSegmentation.m
%                   default to the same config)
%
% OUTPUTS:
%   maskOrig - logical mask at enhancedGray's ORIGINAL resolution
%              (nearest-neighbor resize-back, then re-masked by roiMask)
%   maskNet  - the raw mask at netInputSize, before resize-back. Not
%              needed by the main pipeline; returned for debugging/
%              visualizing exactly what the network saw.
%
% Requires: Image Processing Toolbox, Deep Learning Toolbox.

cfgDefault = segmentationConfig();
if nargin < 4 || isempty(netInputSize)
    netInputSize = cfgDefault.inputSize;
end

origSize = size(enhancedGray, [1 2]);
if ~isequal(size(roiMask, [1 2]), origSize)
    error('runSegmentationNet:roiMaskSizeMismatch', ...
        ['roiMask is %dx%d but enhancedGray is %dx%d - both must come from the SAME ' ...
         'assessAndEnhanceImage.m call on the SAME image.'], ...
        size(roiMask,1), size(roiMask,2), origSize(1), origSize(2));
end
if ~isequal(netInputSize, cfgDefault.inputSize)
    fprintf(['runSegmentationNet: non-default netInputSize [%d %d] ' ...
             '(config default [%d %d]) - ensure training used the same size.\n'], ...
        netInputSize(1), netInputSize(2), cfgDefault.inputSize(1), cfgDefault.inputSize(2));
end

% Parity contract with preprocessFundusForSegmentation.m (training):
% identical bilinear resize + single/255 scaling, so train and inference
% see identically-processed pixels. Mask resize-back is nearest +
% re-masked by the ORIGINAL-resolution roiMask (categorical, not photo).
imgResized = imresize(enhancedGray, netInputSize, 'bilinear');
dlIn = dlarray(single(imgResized) / 255, 'SSC');
dlOut = extractdata(predict(net, dlIn));
% dlOut is [H W 2 N] softmax probabilities (class 1 = Background,
% class 2 = Foreground per segmentationConfig.classNames ordering).
% squeeze handles the N==1 single-image case; callers batching N>1
% should loop per image (documented - this wrapper is single-image).
[~, classIdx] = max(dlOut, [], 3);
maskNet = logical(squeeze(classIdx) == 2); % class 2 = "Foreground"

maskOrig = logical(imresize(maskNet, origSize, 'nearest')) & logical(roiMask);
end
