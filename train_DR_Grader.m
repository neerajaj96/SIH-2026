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

numClasses = 5; % ICDR levels 0-4
classValues = single(0:numClasses-1)';

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

convCandidates = existingNames(contains(lower(existingNames), "conv") & ...
    ~contains(lower(existingNames), ["bn","relu","pool","concat"]));
firstConv = [];
if ~isempty(convCandidates)
    firstConv = getLayer(net, convCandidates(1));
    fprintf('Auto-detected first conv layer as "%s" - double-check this against net.Layers if anything looks off.\n', firstConv.Name);
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

net = initialize(net);
disp('DenseNet-121 fusion architecture ready: 5-channel input, real softmax head, weights expanded.');
plot(net);

% --- Ordinal-aware training via a hybrid loss (see header comment) ---
lambda = 0.5; % relative weight of the ordinal penalty vs. cross-entropy - tune on a validation set
lossFcn = @(Y,T) ordinalSoftmaxLoss(Y, T, classValues, lambda);

trainOpts = trainingOptions('adam', ...
    InitialLearnRate = 1e-4, ...
    LearnRateSchedule = 'piecewise', ...
    LearnRateDropFactor = 0.1, ...
    LearnRateDropPeriod = 5, ...
    MaxEpochs = 30, ...
    MiniBatchSize = 16, ...
    Shuffle = 'every-epoch', ...
    ValidationFrequency = 30, ...
    Plots = 'training-progress', ...
    CheckpointPath = fullfile(pwd,'checkpoints'), ...
    CheckpointFrequency = 5, ...
    CheckpointFrequencyUnit = 'epoch');
if ~isfolder('checkpoints'), mkdir('checkpoints'); end

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

[trainSet, valSet, ~, validationStory] = buildGradingDatasets();
fprintf('Grading training set built. %s\n', validationStory);

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

fprintf('Training DenseNet-121 grader on %d images (%d validation)...\n', numel(trainSet), numel(valSet));
net = trainnet(dsTrain, net, lossFcn, trainOpts);
saveModelWithMetadata('trained_dr_grader.mat', net, struct('lambda', lambda, 'validationStory', validationStory));
fprintf('Saved trained_dr_grader.mat\n');

% Temperature calibration is a SEPARATE step (calibrateTemperature.m) run
% on the VALIDATION set only, never on Messidor-2 if Messidor-2 is your
% test set (see buildGradingDatasets.m's validationStory) - do not fold
% it into this script, and do not skip it before trusting Module 4's
% confidence numbers.
disp('Next: run calibrateTemperature.m on the validation split before trusting any calibrated-confidence output.');

% ------------------------------------------------------------------
function fused = buildFusionTensor(rawImg, vesselNet, maheNet, exudateNet)
[isGradeable, enhancedRGB, enhancedGray, ~, ~, roiMask] = assessAndEnhanceImage(rawImg, -Inf, -Inf); %#ok<ASGLU> - never rejects (see preprocessFundusForSegmentation.m's NOTE on quality gating belonging upstream, at the file-list stage)
vesselMask = runSegmentationNet(vesselNet, enhancedGray, roiMask);
maheMask = runSegmentationNet(maheNet, enhancedGray, roiMask);
exudateMask = runSegmentationNet(exudateNet, enhancedGray, roiMask);
lesionMask = maheMask | exudateMask; % same combination production_inference.m uses for the 5th channel
% Stage-2 fix (matches runScreeningPipeline.m): masks nearest, photos bilinear.
fused = single(cat(3, imresize(enhancedRGB, [224 224], 'bilinear'), ...
                      imresize(uint8(vesselMask)*255, [224 224], 'nearest'), ...
                      imresize(uint8(lesionMask)*255, [224 224], 'nearest')));
end

% ------------------------------------------------------------------
function loss = ordinalSoftmaxLoss(Y, T, classValues, lambda)
% Y: dlarray, softmax probabilities. T: dlarray, one-hot targets, same size.
% classValues: [0;1;2;3;4], the actual ICDR level each row of Y/T stands for.
%
% Plain cross-entropy treats "predicted 0, true 4" the same as "predicted
% 3, true 4" - equally wrong, even though one is a much more dangerous
% miss than the other. The added term penalizes the softmax's EXPECTED
% class value (Y weighted by classValues) for being numerically far from
% the true class - a soft, differentiable stand-in for ordinal-regression
% behavior, without giving up the genuine per-class probabilities a plain
% scalar regression head can't provide.
ce = crossentropy(Y, T);
expectedClass = sum(Y .* classValues, 1);
trueClass = sum(T .* classValues, 1);
ordinalPenalty = mean((expectedClass - trueClass).^2, 'all');
loss = ce + lambda * ordinalPenalty;
end
