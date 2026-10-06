% =========================================================================
% MODULE 2 (Stage-2 hardened): U-Net ensemble for vessel/lesion segmentation
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
% Multiple sources for the same purpose are combined via explicit paired
% file lists (buildSegmentationFileLists.m), not multi-folder datastores
% hoping alphabetical order aligns images with masks.
%
% STAGE-2 HARDENING (this revision):
%   - Single config: segmentationConfig.m owns inputSize/imageSize/
%     classNames/labelIDs ([0 255] canonical - fixes the old [0 1] bug),
%     Tversky alpha/beta, seed, val fraction. No literals duplicated.
%   - Paired file lists by basename stem (never order-assumed combine).
%   - validateMaskConventions ALWAYS runs (even single source), ERRORS on
%     inversion or 0/1-vs-0/255 mix instead of warning-and-continuing.
%   - Leakage-safe split via splitSegmentationDataset.m (seeded,
%     group-aware, no subset() on CombinedDatastore - splits file lists
%     FIRST, then builds per-split datastores).
%   - Per-target checkpoint dirs + resume-from-latest (crash restart now
%     actually works, not just writes files nobody reads).
%   - Tversky guards (probability range, 2-channel shape) + synchronized
%     train-time augmentation (same flip to image AND mask).
%   - Extended .meta.json (resolution, counts, file lists hash, val spec).
%
% BEFORE RUNNING: call datasetRegistry() on its own first and check what
% it reports as FOUND. Each raw download needs a one-time manual
% reorganization into the images/ + masks/ folder pairs this script
% expects. For DDR_seg's 4 per-class masks, run mergeDDRSegMasks.m once
% (MA|HE -> masks_maho/, EX|SE -> masks_exudate/) instead of hand-merging.
% DIARETDB1 ships per-grader confidence markings, not clean binary masks
% - inspect before assuming pixelLabelDatastore can read it directly; the
% validator will flag non-binary sources rather than silently training.
% =========================================================================
disp('Building segmentation U-Net ensemble from available datasets...');
cfg = segmentationConfig();
rng(cfg.seed); % global determinism baseline; per-target reseed below
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

imageSize = cfg.imageSize;
numClasses = cfg.numClasses;
classNames = cfg.classNames;
labelIDs = cfg.labelIDs;

lesionTargets = struct('name', {}, 'imageFolders', {}, 'maskFolders', {}, 'alpha', {}, 'beta', {});
if ~isempty(vesselSources)
    lesionTargets(end+1) = struct('name','Vessels', ...
        'imageFolders', {localExpandSub(vesselSources, 'images')}, ...
        'maskFolders',  {localExpandSub(vesselSources, 'masks')}, ...
        'alpha', cfg.tversky.vessels.alpha, 'beta', cfg.tversky.vessels.beta);
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
    % datasetRegistry.m's DDR_seg entry) - run mergeDDRSegMasks.m once to
    % build maho/ (MA|HE) and exudate/ (EX|SE) subfolders under each
    % source's local dir before running this.
    lesionTargets(end+1) = struct('name','MicroaneurysmsHemorrhages', ...
        'imageFolders', {localExpandSub(lesionSources, 'images')}, ...
        'maskFolders',  {localExpandSub(lesionSources, 'masks_maho')}, ...
        'alpha', cfg.tversky.mahe.alpha, 'beta', cfg.tversky.mahe.beta);
    lesionTargets(end+1) = struct('name','Exudates', ...
        'imageFolders', {localExpandSub(lesionSources, 'images')}, ...
        'maskFolders',  {localExpandSub(lesionSources, 'masks_exudate')}, ...
        'alpha', cfg.tversky.exudate.alpha, 'beta', cfg.tversky.exudate.beta);
end

if isempty(lesionTargets)
    error(['train_UNet_Segmentation:noData - datasetRegistry() found zero vessel or lesion ' ...
           'sources. Run datasetRegistry() on its own to see what''s missing before running this.']);
end

baseOptions = trainingOptions('adam', ...
    InitialLearnRate = cfg.train.initialLearnRate, ...
    MaxEpochs = cfg.train.maxEpochs, ...
    MiniBatchSize = cfg.train.miniBatchSize, ...
    Shuffle = 'every-epoch', ...
    ValidationFrequency = cfg.train.validationFrequency, ...
    Plots = 'training-progress', ...
    VerboseFrequency = 10, ...
    CheckpointFrequency = cfg.train.checkpointFrequency, ...
    CheckpointFrequencyUnit = cfg.train.checkpointFrequencyUnit);

for i = 1:numel(lesionTargets)
    target = lesionTargets(i);
    fprintf('\n--- %s (%d image folder(s), %d mask folder(s)) ---\n', ...
        target.name, numel(target.imageFolders), numel(target.maskFolders));
    rng(cfg.seed + i); % per-target deterministic stream (split + init + shuffle)

    % 1) Paired file lists (fixes nested-cell + misalignment bugs).
    [imagePaths, maskPaths, listReport] = buildSegmentationFileLists( ...
        target.imageFolders, target.maskFolders, target.name);

    % 2) Mask-convention gate: ALWAYS runs, even single source. Errors on
    % inversion or mixed 0/1-vs-0/255 (unrecoverable label corruption).
    fprintf('Checking mask conventions before training...\n');
    conventionReport = validateMaskConventions(target.maskFolders, target.imageFolders);

    % 3) Leakage-safe seeded split on FILE LISTS (no subset() fragility).
    % PatientIdFcn: segmentation sources publish no patient/eye key, so
    % this is an explicit image-level split - the report says so. If a
    % source-specific key becomes available (e.g. Messidor pairing for a
    % future seg source), pass it here to upgrade to group-level.
    [trainTbl, valTbl, splitReport] = splitSegmentationDataset( ...
        imagePaths, maskPaths, ...
        'ValFraction', cfg.valFraction, 'Seed', cfg.seed + i, ...
        'TargetName', target.name);

    % 4) Per-split datastores. Images: SAME enhancement path as inference
    % (preprocessFundusForSegmentation - closes the covariate-shift bug).
    % Masks: nearest-neighbor resize only (categorical, never enhanced).
    netInputSize = cfg.inputSize;
    imdsTrainPlain = imageDatastore(trainTbl.imagePaths);
    imdsTrain = transform(imdsTrainPlain, @(img) localPreprocess(img, netInputSize));
    pxdsTrainPlain = pixelLabelDatastore(trainTbl.maskPaths, classNames, labelIDs);
    pxdsTrain = transform(pxdsTrainPlain, @(lbl) imresize(lbl, netInputSize, 'nearest'));
    cdsTrain = combine(imdsTrain, pxdsTrain);
    % Synchronized augmentation: same random horizontal flip to image AND
    % mask (flipping one without the other would corrupt supervision).
    cdsTrain = transform(cdsTrain, @localAugmentPair);

    imdsValPlain = imageDatastore(valTbl.imagePaths);
    imdsVal = transform(imdsValPlain, @(img) localPreprocess(img, netInputSize));
    pxdsValPlain = pixelLabelDatastore(valTbl.maskPaths, classNames, labelIDs);
    pxdsVal = transform(pxdsValPlain, @(lbl) imresize(lbl, netInputSize, 'nearest'));
    cdsVal = combine(imdsVal, pxdsVal);

    % 5) Per-target checkpoint dir + resume-from-latest.
    ckptDir = fullfile(pwd, 'checkpoints', ['seg_' char(target.name)]);
    if ~isfolder(ckptDir), mkdir(ckptDir); end
    trainOpts = baseOptions;
    trainOpts.CheckpointPath = ckptDir;
    trainOpts.ValidationData = cdsVal;
    startNet = unet(imageSize, numClasses, EncoderDepth=cfg.encoderDepth);
    resumeFile = localLatestCheckpoint(ckptDir);
    if ~isempty(resumeFile)
        try
            S = load(resumeFile);
            % Checkpoint files store 'net' (trainnet format). Accept both
            % 'net' and legacy bare-network saves.
            if isfield(S, 'net')
                startNet = S.net;
                fprintf('Resuming %s from checkpoint %s\n', target.name, resumeFile);
            end
        catch ME
            warning('train_UNet_Segmentation:resumeFailed', ...
                'Found %s but could not load it (%s) - training from scratch.', resumeFile, ME.message);
        end
    end
    lossFcn = @(Y,T) tverskyLoss(Y, T, target.alpha, target.beta);

    fprintf('Training %s on %d images (%d held out for validation)...\n', ...
        target.name, numel(trainTbl.imagePaths), numel(valTbl.imagePaths));
    net = trainnet(cdsTrain, startNet, lossFcn, trainOpts);

    outFile = sprintf('unet_%s.mat', target.name);
    saveModelWithMetadata(outFile, net, struct( ...
        'lesionTarget', target.name, 'alpha', target.alpha, 'beta', target.beta, ...
        'inputSize', netInputSize, 'imageSize', imageSize, ...
        'encoderDepth', cfg.encoderDepth, 'seed', cfg.seed + i, ...
        'nTrain', numel(trainTbl.imagePaths), 'nVal', numel(valTbl.imagePaths), ...
        'nPaired', listReport.nPaired, 'splitPatientLevel', splitReport.isPatientLevel, ...
        'classNames', classNames, 'labelIDs', labelIDs));
    fprintf('Saved %s\n', outFile);
end

disp(' ');
disp('Done. production_inference.m expects unet_Vessels.mat, unet_MicroaneurysmsHemorrhages.mat,');
disp('and unet_Exudates.mat next to it - matching the target names above.');

% ------------------------------------------------------------------
function expanded = localExpandSub(sources, sub)
% Correctly expands {'<dir1>','<dir2>'} + 'images' -> {'<dir1>/images',
% '<dir2>/images'}. Replaces the buggy fullfile(cell,'images')-in-{…}
% pattern that produced nested cells and a count of 1 for N sources.
expanded = cellfun(@(d) fullfile(d, sub), sources, 'UniformOutput', false);
end

% ------------------------------------------------------------------
function imgOut = localPreprocess(rawImg, netInputSize)
% preprocessFundusForSegmentation returns [imgOut, roiMaskOut]; a
% datastore transform needs exactly one output per read, and training
% doesn't need the ROI mask (that matters at inference time for
% re-masking the resized prediction - see runSegmentationNet.m).
[imgOut, ~] = preprocessFundusForSegmentation(rawImg, netInputSize);
% Guard the U-Net channel contract: unet([H W 1]) needs HxWx1 single.
% Grayscale preprocess returns 2D HxW; add the singleton channel dim.
if ndims(imgOut) == 2
    imgOut = reshape(imgOut, [size(imgOut,1), size(imgOut,2), 1]);
end
end

% ------------------------------------------------------------------
function pair = localAugmentPair(pairIn)
% Synchronized train-time augmentation for a combined {image, label} read.
% Applies the SAME random horizontal flip to both (image bilinear content
% tolerates flip; mask is flipped identically so supervision stays aligned).
% Vertical flips and rotations are deliberately NOT applied: fundus
% orientation (superior/inferior) matters to downstream quadrant logic, and
% arbitrary rotations would require matching ROI handling. Extend here
% (with identical geometry to both) if a future benchmark justifies it.
img = pairIn{1}; lbl = pairIn{2};
if rand() < 0.5
    img = fliplr(img);
    lbl = fliplr(lbl);
end
pair = {img, lbl};
end

% ------------------------------------------------------------------
function loss = tverskyLoss(Y, T, alpha, beta)
% Y: network output (softmax probabilities, [H W 2 N]).
% T: one-hot targets, same size (pixelLabelDatastore + trainnet encoding).
% Guards: Y must be probabilities in [0,1] with 2 channels; if trainnet
% ever passes logits (negatives, rows not summing to 1), this errors
% loudly instead of training on a meaningless loss.
smooth = 1e-6;
assert(size(Y, 3) == 2, 'tverskyLoss:channels - expected 2-channel softmax output, got %d.', size(Y, 3));
dY = extractdata(Y);
if any(dY(:) < -1e-3) || any(dY(:) > 1 + 1e-3)
    error(['tverskyLoss:notProbabilities - network output outside [0,1] (min %.3f, max %.3f). ' ...
           'This loss expects post-softmax probabilities; check the U-Net head.'], min(dY(:)), max(dY(:)));
end
TP = sum(sum(Y .* T, 1), 2);
FP = sum(sum(Y .* (1 - T), 1), 2);
FN = sum(sum((1 - Y) .* T, 1), 2);
tverskyIndex = (TP + smooth) ./ (TP + alpha.*FP + beta.*FN + smooth);
% Mean over classes AND batch: background-heavy batches still contribute
% 50% background by design (screening recalls foreground at the cost of
% FP - see segmentationConfig). Class-imbalance handling beyond Tversky
% beta (foreground sampling) is a future benchmark, not a silent default.
loss = mean(1 - tverskyIndex, 'all');
end

% ------------------------------------------------------------------
function latest = localLatestCheckpoint(ckptDir)
latest = '';
d = dir(fullfile(ckptDir, '*.mat'));
if isempty(d), return; end
[~, order] = sort([d.datenum]);
latest = fullfile(d(order(end)).folder, d(order(end)).name);
end
