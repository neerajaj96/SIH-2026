% =========================================================================
% MODULE 3 (fixed): DenseNet-121 transfer learning for DR severity grading
% Requires: Deep Learning Toolbox, Deep Learning Toolbox Model for
%           DenseNet-121 Network (Add-On Explorer will offer to install it
%           if imagePretrainedNetwork errors on the model name below)
%
% FIXES vs. the original draft:
%   - densenet121() is (not recommended); imagePretrainedNetwork is the
%     modern loader and returns a dlnetwork directly.
%   - connectLayers(...,'pool5',...) referenced a ResNet/AlexNet-style
%     layer name that does not exist on DenseNet's graph. DenseNet's own
%     pooling layer before the classification head is 'avg_pool'.
%   - The 3->5 channel input swap only replaced imageInputLayer; the very
%     next conv layer's weights were still shaped for 3 channels, which
%     is a hard shape mismatch. Weight expansion is now done explicitly,
%     seeding the two new (mask) channels from the mean of the pretrained
%     RGB filters rather than leaving a dimension the network can't use.
%   - BIGGEST FIX - the design doc has this network as pure ordinal
%     regression (regressionLayer, one scalar output, no softmax) here in
%     Module 3, then in Module 4 describes calibrating "softmax output
%     probabilities" via temperature scaling and running gradCAM with a
%     'prob' reduction layer. Those two don't fit together: a scalar
%     regression output has no per-class probability vector to calibrate
%     or compute a class-conditional Grad-CAM gradient from. Fixed by
%     using a genuine 5-way softmax head trained with a HYBRID loss:
%     standard cross-entropy plus a soft ordinal penalty on the softmax's
%     own expected class value. This keeps the ordinality-awareness
%     Module 3 wanted while producing the real per-class probabilities
%     Module 4's calibration and Grad-CAM math actually need.
% =========================================================================

gcfg = gradingConfig(); % frozen contract: 5ch order, 224x224x5, labels 0-4, lambda, seed
numClasses = gcfg.numClasses; % ICDR levels 0-4
classValues = gcfg.classValues;
rng(gcfg.seed); % determinism: head init + shuffle + dropout order all derive from here

disp('Loading pretrained DenseNet-121 as a dlnetwork...');
try
    net = imagePretrainedNetwork("densenet121", NumClasses=numClasses);
catch loadErr
    warning(['imagePretrainedNetwork("densenet121") failed: %s\n' ...
             'Falling back to the legacy densenet121() + dag2dlnetwork path. ' ...
             'Worth confirming interactively whether your MATLAB release lists ' ...
             '"densenet121" as a valid imagePretrainedNetwork model name (it is ' ...
             'confirmed for "densenet201"; DenseNet-121 support was not directly ' ...
             'verifiable from documentation alone) before relying on this ' ...
             'fallback long-term.'], loadErr.message); %#ok<CTPCT>
    legacyNet = densenet121; %#ok<UNRCH>
    net = dag2dlnetwork(legacyNet);
end

% --- Replace the classification head with 5 ordinal classes + real softmax ---
% (imagePretrainedNetwork's NumClasses argument above should already have
% done this on releases that support it; this block is close to a no-op
% in that case, and the actual fix if the fallback path ran instead.)
existingNames = string({net.Layers.Name});
if ~ismember("dr_fc", existingNames)
    headNames = existingNames(contains(lower(existingNames), ["fc","prob","classification","softmax"]));
    if ~isempty(headNames)
        net = removeLayers(net, cellstr(headNames));
    end
    net = addLayers(net, [
        fullyConnectedLayer(numClasses, 'Name','dr_fc', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10)
        softmaxLayer('Name','prob')]);
    % DenseNet-121 shares its head-pooling layer name with DenseNet-201 in
    % every MATLAB DenseNet transfer-learning example: 'avg_pool'.
    % ('pool5' is the ResNet/AlexNet convention and does not exist on this
    % graph - if this errors, run net.Layers and swap in the real name.)
    net = connectLayers(net, 'avg_pool', 'dr_fc');
    existingNames = string({net.Layers.Name});
end

% --- Expand the input layer AND the first conv layer's weights from
%     3 (RGB) to 5 (RGB + vessel mask + lesion mask) channels ---
net = replaceLayer(net, net.InputNames{1}, ...
    imageInputLayer([224 224 5], 'Name','fusion_input', 'Normalization','zscore'));
existingNames = string({net.Layers.Name});

% First-conv detection: first convolution2dLayer in graph order with 3
% input channels (not name-substring matching, which can hit a dense-block
% conv on renamed graphs). DenseNet-121's stem is a 7x7 stride-2 conv.
firstConv = [];
for L = 1:numel(net.Layers)
    lyr = net.Layers(L);
    if isa(lyr, 'nnet.cnn.layer.Convolution2DLayer') && size(lyr.Weights, 3) == 3
        firstConv = lyr;
        break;
    end
end
if ~isempty(firstConv)
    fprintf('First 3-channel conv layer is "%s" (filter %s, %d filters).\n', ...
        firstConv.Name, mat2str(firstConv.FilterSize), firstConv.NumFilters);
end

if ~isempty(firstConv) && isprop(firstConv, 'Weights') && ~isempty(firstConv.Weights) && size(firstConv.Weights,3) == 3
    oldW = firstConv.Weights; % [kH kW 3 numFilters]
    % Seed the two new (mask) channels from the mean of the pretrained RGB
    % filters, rather than random noise the pretrained network never saw -
    % a standard, defensible starting point for expanding a pretrained
    % conv layer's input channel count.
    extraChannel = mean(oldW, 3);
    newW = cat(3, oldW, extraChannel, extraChannel); % -> 5 channels
    newConv = convolution2dLayer(firstConv.FilterSize, firstConv.NumFilters, ...
        'Name', firstConv.Name, 'Stride', firstConv.Stride, ...
        'Padding', firstConv.PaddingSize, 'Weights', newW, 'Bias', firstConv.Bias);
    net = replaceLayer(net, firstConv.Name, newConv);
    fprintf('Expanded %s from 3 to 5 input channels.\n', firstConv.Name);
else
    warning(['Could not auto-locate a 3-channel first conv layer to expand. ' ...
             'Inspect net.Layers, find the real first conv layer''s name, and ' ...
             'redo the weight-expansion block above manually before training - ' ...
             'otherwise the 5-channel input and the network''s first conv layer ' ...
             'will disagree on shape.']);
end

net = initialize(net); % fills only uninitialized params (new head); pretrained + expanded conv kept
disp('DenseNet-121 fusion architecture ready: 5-channel input, real softmax head, weights expanded.');
try
    plot(net);
catch ME
    warning('train_DR_Grader:noPlot', 'plot(net) skipped (%s) - headless/logged runs do not need it.', ME.message);
end
assert(isequal(net.Layers(end).Name, gcfg.headProbName) || ismember(gcfg.headFcName, string({net.Layers.Name})), ...
    'train_DR_Grader:headCheck - classifier head names unexpected; check dr_fc/prob wiring.');
outSizes = net.Layers(strcmp({net.Layers.Name}, gcfg.headFcName));
if ~isempty(outSizes)
    assert(outSizes.OutputSize == 5, 'train_DR_Grader:headOutputs - dr_fc must have exactly 5 outputs, got %d.', outSizes.OutputSize);
end

% --- Ordinal-aware training via the audited hybrid loss (ordinalGradingLoss.m) ---
% Same formulation as before (CE + lambda*E-grade MSE), now with shape,
% scale and stability guards. Tune lambda ONLY on validation, never test.
lambda = gcfg.ordinalLambda;
classWeights = []; % default: unweighted. Median-frequency weights are computed below and only used if useClassWeights=true (val-tuned).
useClassWeights = false;
lossFcn = @(Y,T) ordinalGradingLoss(Y, T, classValues, lambda, classWeights);

trainOpts = trainingOptions('adam', ...
    InitialLearnRate = gcfg.train.initialLearnRate, ...
    LearnRateSchedule = gcfg.train.learnRateSchedule, ...
    LearnRateDropFactor = gcfg.train.learnRateDropFactor, ...
    LearnRateDropPeriod = gcfg.train.learnRateDropPeriod, ...
    MaxEpochs = gcfg.train.maxEpochs, ...
    MiniBatchSize = gcfg.train.miniBatchSize, ...
    Shuffle = gcfg.train.shuffle, ...
    ValidationFrequency = gcfg.train.validationFrequency, ...
    Plots = 'training-progress', ...
    CheckpointFrequency = gcfg.train.checkpointFrequency, ...
    CheckpointFrequencyUnit = gcfg.train.checkpointFrequencyUnit);
ckptDir = fullfile(pwd, 'checkpoints', 'grader_densenet121');
if ~isfolder(ckptDir), mkdir(ckptDir); end
trainOpts.CheckpointPath = ckptDir;

% --- TRAINING IS NOW ACTIVE (a real MATLAB license + toolboxes are
% assumed, so there's no more reason to leave this as guidance-only) ---
% Module 3 depends on Module 2: building the 5-channel fusion tensor for
% every training image needs the TRAINED vessel/lesion U-Nets to already
% exist, since the classifier's input includes their predicted masks, not
% ground-truth ones (matching what happens at real inference time - using
% ground-truth masks here would let the grader cheat on features the real
% pipeline never has access to).
if ~isfile('unet_Vessels.mat') || ~isfile('unet_MicroaneurysmsHemorrhages.mat') || ~isfile('unet_Exudates.mat')
    error(['train_DR_Grader:segmentationNotTrained - run train_UNet_Segmentation.m first. ' ...
           'The 5-channel input needs PREDICTED vessel/lesion masks, which needs those networks ' ...
           'trained already - using ground-truth masks here would be information the real pipeline ' ...
           'never has at inference time.']);
end
S = load('unet_Vessels.mat','net'); vesselNet = S.net;
S = load('unet_MicroaneurysmsHemorrhages.mat','net'); maheNet = S.net;
S = load('unet_Exudates.mat','net'); exudateNet = S.net;

[trainSet, valSet, ~, validationStory, dataManifest] = buildGradingDatasets();
fprintf('Grading training set built. %s\n', validationStory);

% Class-imbalance audit (P4): report only by default. Enabling weights
% changes optimization - do so ONLY with a val-measured justification, and
% the choice is recorded in .meta.json either way (no silent default).
trainCounts = histcounts([trainSet.icdrGrade], -0.5:4.5);
medFreq = median(trainCounts(trainCounts > 0));
classWeightsComputed = medFreq ./ max(trainCounts, 1);
fprintf('Train class weights (median-frequency, for reference; NOT applied unless useClassWeights=true): %s\n', ...
    mat2str(round(classWeightsComputed * 100) / 100));
if useClassWeights
    classWeights = classWeightsComputed(:);
    lossFcn = @(Y,T) ordinalGradingLoss(Y, T, classValues, lambda, classWeights);
    fprintf('Class weights ENABLED (val-justified run) - recorded in metadata.\n');
end

fusionFcn = @(rawImg) buildFusionTensor(rawImg, vesselNet, maheNet, exudateNet);
imdsTrain = transform(imageDatastore({trainSet.imagePath}), fusionFcn);
imdsVal   = transform(imageDatastore({valSet.imagePath}),   fusionFcn);

oneHotTrain = onehotencode(categorical([trainSet.icdrGrade], 0:numClasses-1), 1)'; % N x numClasses
oneHotVal   = onehotencode(categorical([valSet.icdrGrade], 0:numClasses-1), 1)';
labelsTrain = arrayDatastore(oneHotTrain, 'IterationDimension', 1);
labelsVal   = arrayDatastore(oneHotVal,   'IterationDimension', 1);

dsTrain = combine(imdsTrain, labelsTrain);
dsVal   = combine(imdsVal, labelsVal);
trainOpts.ValidationData = dsVal;

% Resume-from-latest: a multi-hour run survives crash/restart (per-model
% dir, so grader and baseline checkpoints never collide).
resumeFile = localLatestCheckpoint(ckptDir);
if ~isempty(resumeFile)
    try
        S = load(resumeFile);
        if isfield(S, 'net')
            net = S.net;
            fprintf('Resuming grader from checkpoint %s\n', resumeFile);
        end
    catch ME
        warning('train_DR_Grader:resumeFailed', 'Found %s but could not load it (%s) - training from scratch.', resumeFile, ME.message);
    end
end

fprintf('Training DenseNet-121 grader on %d images (%d validation)...\n', numel(trainSet), numel(valSet));
net = trainnet(dsTrain, net, lossFcn, trainOpts);
extra = struct( ...
    'channelOrder', {gcfg.channelOrder}, 'inputSize', gcfg.inputSize, ...
    'classMapping', 'categorical(grades,0:4) -> onehot, pred=argmax-1, referable>=2', ...
    'loss', 'ordinalGradingLoss CE + lambda*E-grade-MSE', 'lambda', lambda, ...
    'useClassWeights', useClassWeights, 'classWeights', classWeightsComputed, ...
    'trainCounts', trainCounts, 'valCounts', histcounts([valSet.icdrGrade], -0.5:4.5), ...
    'seed', gcfg.seed, 'validationStory', validationStory, ...
    'dataManifest', dataManifest, 'toolboxVersions', localToolboxVersions());
saveModelWithMetadata('trained_dr_grader.mat', net, extra);
fprintf('Saved trained_dr_grader.mat\n');

% Temperature calibration is a SEPARATE step (calibrateTemperature.m) run
% on the VALIDATION set only, never on Messidor-2 if Messidor-2 is your
% test set (see buildGradingDatasets.m's validationStory) - do not fold
% it into this script, and do not skip it before trusting Module 4's
% confidence numbers.
disp('Next: run calibrateTemperature.m on the validation split before trusting any calibrated-confidence output.');

% ------------------------------------------------------------------
function fused = buildFusionTensor(rawImg, vesselNet, maheNet, exudateNet)
% Training-time fusion via the canonical builder (same enhance path +
% same bilinear/nearest semantics as inference; loud validation inside).
[~, enhancedRGB, enhancedGray, ~, ~, roiMask] = assessAndEnhanceImage(rawImg, -Inf, -Inf); % never rejects: quality curation belongs at the file-list stage, not per-read
vesselMask = runSegmentationNet(vesselNet, enhancedGray, roiMask);
maheMask = runSegmentationNet(maheNet, enhancedGray, roiMask);
exudateMask = runSegmentationNet(exudateNet, enhancedGray, roiMask);
fused = buildGradingFusionTensor(enhancedRGB, vesselMask, maheMask, exudateMask);
end

% ------------------------------------------------------------------
function latest = localLatestCheckpoint(ckptDir)
latest = '';
d = dir(fullfile(ckptDir, '*.mat'));
if isempty(d), return; end
[~, order] = sort([d.datenum]);
latest = fullfile(d(order(end)).folder, d(order(end)).name);
end

% ------------------------------------------------------------------
function v = localToolboxVersions()
% Records what the model was trained with (P7 reproducibility).
v = struct();
try
    vs = ver;
    for i = 1:numel(vs)
        name = matlab.lang.makeValidName(vs(i).Name);
        v.(name) = vs(i).Version;
    end
    v.MATLAB = version;
catch
    v.note = 'ver() unavailable - versions not recorded';
end
end
