% =========================================================================
% MODULE 2 (remodeled): U-Net ensemble for vessel/lesion segmentation
% Requires: Deep Learning Toolbox, Image Processing Toolbox, Computer
%           Vision Toolbox - assumed fully available and licensed.
%
% REMODEL (dataset access changed the plan): APTOS/IDRiD/DRIVE turned out
% to sit behind access walls that didn't clear; only Messidor-2 actually
% downloaded. Rather than hardcode a new fixed dataset list that might hit
% the same problem again, this script now pulls whichever vessel/lesion
% sources are ACTUALLY PRESENT from datasetRegistry.m and combines them:
%   Vessels: STARE + CHASE_DB1 + HRF + DRIVE, whichever are found
%            (direct university hosting - no Kaggle-competition wall)
%   Lesions: DDR's 757-image segmentation subset (primary) + DIARETDB1
%            (supplement) - both far exceed IDRiD's 81-image segmentation
%            subset if they download successfully
% Multiple sources for the same purpose are combined via imageDatastore's
% cell-array-of-folders support, not trained as separate networks.
%
% TRAINING IS NOW ACTIVE (previously left commented out while this
% project had no way to verify a training call wouldn't silently fail
% partway through in an environment with no real MATLAB to test against).
% With a real license and toolboxes assumed, trainnet(...) runs for real
% below, with checkpointing so a multi-hour run survives a crash/restart.
%
% BEFORE RUNNING: call datasetRegistry() on its own first and check what
% it reports as FOUND. Each raw download needs a one-time manual
% reorganization into the images/ + masks/ folder pairs this script
% expects (STARE/CHASE_DB1/HRF/DDR each ship with their own internal
% layout that I have not seen firsthand and cannot script a parser for
% sight-unseen - open the extracted folder, see what's actually in it,
% and sort images into <registry localDir>/images and masks into
% <registry localDir>/masks, binary PNG, 0=background/255=foreground).
% =========================================================================
disp('Building segmentation U-Net ensemble from available datasets...');
registry = datasetRegistry();
availByName = containers.Map({registry.name}, {registry.available});
dirByName = containers.Map({registry.name}, {registry.localDir});

vesselSources = {};
for name = ["STARE","CHASE_DB1","HRF","DRIVE"]
    if availByName(char(name))
        vesselSources{end+1} = dirByName(char(name)); %#ok<AGROW>
    end
end
lesionSources = {};
for name = ["DDR_seg","DIARETDB1","IDRiD"]
    if availByName(char(name))
        lesionSources{end+1} = dirByName(char(name)); %#ok<AGROW>
    end
end

fprintf('Vessel sources found: %d  |  Lesion sources found: %d\n', numel(vesselSources), numel(lesionSources));

imageSize = [512 512 1];
numClasses = 2; % Background, Foreground
classNames = ["Background","Foreground"];
labelIDs = [0, 1];

lesionTargets = struct('name', {}, 'imageFolders', {}, 'maskFolders', {}, 'alpha', {}, 'beta', {});
if ~isempty(vesselSources)
    lesionTargets(end+1) = struct('name','Vessels', ...
        'imageFolders', {fullfile(vesselSources,'images')}, ...
        'maskFolders',  {fullfile(vesselSources,'masks')}, ...
        'alpha', 0.3, 'beta', 0.7);
end
if ~isempty(lesionSources)
    % Kept as TWO separate targets (not collapsed to one "Lesions" class)
    % specifically because production_inference.m and runSegmentationNet.m
    % already expect unet_Vessels.mat / unet_MicroaneurysmsHemorrhages.mat
    % / unet_Exudates.mat as three separate files - matching that instead
    % of introducing a fourth naming scheme this project would then have
    % to reconcile everywhere else that loads these .mat files.
    %
    % DDR_seg ships MA, HE, EX, and SE as four SEPARATE mask classes (see
    % datasetRegistry.m's DDR_seg entry) - combine MA+HE masks into one
    % maho/ subfolder and EX(+SE) masks into one exudate/ subfolder under
    % each source's local dir before running this (I have not seen DDR's
    % actual per-class file layout firsthand, so I can't script that
    % merge sight-unseen - open the extracted folder and see what's there).
    lesionTargets(end+1) = struct('name','MicroaneurysmsHemorrhages', ...
        'imageFolders', {fullfile(lesionSources,'images')}, ...
        'maskFolders',  {fullfile(lesionSources,'masks_maho')}, ...
        'alpha', 0.3, 'beta', 0.7);
    lesionTargets(end+1) = struct('name','Exudates', ...
        'imageFolders', {fullfile(lesionSources,'images')}, ...
        'maskFolders',  {fullfile(lesionSources,'masks_exudate')}, ...
        'alpha', 0.3, 'beta', 0.6);
end

if isempty(lesionTargets)
    error(['train_UNet_Segmentation:noData - datasetRegistry() found zero vessel or lesion ' ...
           'sources. Run datasetRegistry() on its own to see what''s missing before running this.']);
end

baseOptions = trainingOptions('adam', ...
    InitialLearnRate = 1e-3, ...
    MaxEpochs = 30, ...
    MiniBatchSize = 8, ...
    Shuffle = 'every-epoch', ...
    ValidationFrequency = 20, ...
    Plots = 'training-progress', ...
    VerboseFrequency = 10, ...
    CheckpointPath = fullfile(pwd,'checkpoints'), ...
    CheckpointFrequency = 5, ...
    CheckpointFrequencyUnit = 'epoch');

if ~isfolder('checkpoints'), mkdir('checkpoints'); end

for i = 1:numel(lesionTargets)
    target = lesionTargets(i);
    fprintf('\n--- %s (%d source folder(s)) ---\n', target.name, numel(target.imageFolders));

    if numel(target.maskFolders) > 1
        fprintf('Checking mask convention consistency across %d source(s) before training...\n', numel(target.maskFolders));
        validateMaskConventions(target.maskFolders);
    end

    net = unet(imageSize, numClasses, EncoderDepth=4);
    lossFcn = @(Y,T) tverskyLoss(Y, T, target.alpha, target.beta);

    imdsPlain = imageDatastore(target.imageFolders);
    n = numel(imdsPlain.Files);
    % preprocessFundusForSegmentation applies the SAME enhancement path
    % (CLAHE, flat-fielding, denoise via assessAndEnhanceImage.m) that
    % production_inference.m's runSegmentationNet.m feeds the trained
    % network at inference time. Training on raw pixels while inference
    % runs on enhanced pixels is a covariate-shift bug distinct from (and
    % on top of) the resize mismatch - both are closed by routing both
    % paths through the same function. Returns [imgOut, roiMaskOut]; only
    % imgOut is needed for the datastore, hence the wrapper below.
    imds = transform(imdsPlain, @(img) firstOutputOnly(img, imageSize(1:2)));
    pxds = pixelLabelDatastore(target.maskFolders, classNames, labelIDs);
    pxds = transform(pxds, @(lbl) imresize(lbl, imageSize(1:2), 'nearest')); % masks: plain nearest-neighbor resize only - no enhancement, they're categorical labels, not photos
    cds = combine(imds, pxds);

    nVal = max(round(0.15*n), 1);
    [cdsTrain, cdsVal] = splitEachLabel_manual(cds, n, nVal);
    % splitEachLabel_manual: see local function below - pixelLabelDatastore
    % doesn't support MATLAB's splitEachLabel the way plain classification
    % datastores do, so this does an index-based split instead. Uses
    % subset(), which is a general datastore method - if your MATLAB
    % release's subset() doesn't accept a CombinedDatastore directly,
    % apply subset() to imds and pxds separately (same indices) and
    % combine() the two subsets instead.

    fprintf('Training %s on %d images (%d held out for validation)...\n', target.name, n-nVal, nVal);
    trainOpts = baseOptions;
    trainOpts.ValidationData = cdsVal;
    net = trainnet(cdsTrain, net, lossFcn, trainOpts);

    outFile = sprintf('unet_%s.mat', target.name);
    saveModelWithMetadata(outFile, net, struct('lesionTarget', target.name, 'alpha', target.alpha, 'beta', target.beta, 'nSourceFolders', numel(target.imageFolders)));
    fprintf('Saved %s\n', outFile);
end

disp(' ');
disp('Done. production_inference.m expects unet_Vessels.mat, unet_MicroaneurysmsHemorrhages.mat,');
disp('and unet_Exudates.mat next to it - matching the target names above.');

% ------------------------------------------------------------------
function imgOut = firstOutputOnly(rawImg, netInputSize)
% preprocessFundusForSegmentation returns [imgOut, roiMaskOut]; a
% datastore transform function needs exactly one output per read, and
% training doesn't need the ROI mask (that matters at inference time for
% re-masking the resized prediction - see runSegmentationNet.m).
[imgOut, ~] = preprocessFundusForSegmentation(rawImg, netInputSize);
end

% ------------------------------------------------------------------
function loss = tverskyLoss(Y, T, alpha, beta)
smooth = 1e-6;
TP = sum(sum(Y .* T, 1), 2);
FP = sum(sum(Y .* (1 - T), 1), 2);
FN = sum(sum((1 - Y) .* T, 1), 2);
tverskyIndex = (TP + smooth) ./ (TP + alpha.*FP + beta.*FN + smooth);
loss = mean(1 - tverskyIndex, 'all');
end

% ------------------------------------------------------------------
function [cdsTrain, cdsVal] = splitEachLabel_manual(cds, n, nVal)
idx = randperm(n);
valIdx = idx(1:nVal);
trainIdx = idx(nVal+1:end);
cdsTrain = subset(cds, trainIdx);
cdsVal = subset(cds, valIdx);
end
