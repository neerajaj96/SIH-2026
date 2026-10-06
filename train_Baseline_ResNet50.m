% =========================================================================
% BASELINE (for the ablation study): plain ResNet-50 on raw images
% Requires: Deep Learning Toolbox, Deep Learning Toolbox Model for
%           ResNet-50 Network
%
% This is deliberately the SIMPLE, single-technique approach: raw
% (unenhanced) images, one pretrained backbone, standard 5-way softmax
% classification, plain cross-entropy loss. No quality gating, no
% segmentation fusion, no ordinal-aware loss, no calibration.
%
% Its only job is to exist as a fair comparison point for compareModels.m,
% because the official PS's Expected Solution explicitly asks for
% "validation against published benchmarks showing the integrated
% pipeline outperforms any single technique approach" - which needs an
% actual single-technique model trained on the actual same data split to
% compare against, not just an assertion.
% =========================================================================

numClasses = 5;
disp('Loading pretrained ResNet-50 as a dlnetwork (baseline)...');
net = imagePretrainedNetwork("resnet50", NumClasses=numClasses);
net = initialize(net);
disp('Baseline ResNet-50 ready: raw 3-channel input, plain softmax head.');

gcfgB = gradingConfig(); % seed + schedule shared with the pipeline model (paired comparison)
rng(gcfgB.seed);
trainOpts = trainingOptions('adam', ...
    InitialLearnRate = 1e-4, ...
    MaxEpochs = 30, ...
    MiniBatchSize = 16, ...
    Shuffle = 'every-epoch', ...
    ValidationFrequency = 30, ...
    Plots = 'training-progress', ...
    CheckpointFrequency = 5, ...
    CheckpointFrequencyUnit = 'epoch');
ckptDirB = fullfile(pwd, 'checkpoints', 'baseline_resnet50'); % per-model dir: never collides with grader_densenet121
if ~isfolder(ckptDirB), mkdir(ckptDirB); end
trainOpts.CheckpointPath = ckptDirB;

% Uses buildGradingDatasets() - the SAME function, same fixed seed (42),
% as train_DR_Grader.m, so this gets the IDENTICAL train/val split and
% the identical Messidor-2 test set. That identity is what makes
% compareModels.m's McNemar's test valid - it's a paired comparison on
% the same held-out cases, not two different train sets with a coincidence.
% Do NOT change the seed/fractions here without changing the grader identically.
[trainSet, valSet, ~, validationStory, dataManifestB] = buildGradingDatasets();
fprintf('Baseline training set built (same split as train_DR_Grader.m). %s\n', validationStory);

rawFcn = @(rawImg) single(imresize(rawImg, [224 224])); % raw RGB only - no enhancement, no mask fusion, that's the whole point of a baseline
imdsTrain = transform(imageDatastore({trainSet.imagePath}), rawFcn);
imdsVal   = transform(imageDatastore({valSet.imagePath}),   rawFcn);

oneHotTrain = onehotencode(categorical([trainSet.icdrGrade], 0:numClasses-1), 1)';
oneHotVal   = onehotencode(categorical([valSet.icdrGrade], 0:numClasses-1), 1)';
dsTrain = combine(imdsTrain, arrayDatastore(oneHotTrain, 'IterationDimension', 1));
dsVal   = combine(imdsVal,   arrayDatastore(oneHotVal,   'IterationDimension', 1));
trainOpts.ValidationData = dsVal;

fprintf('Training baseline ResNet-50 on %d images (%d validation)...\n', numel(trainSet), numel(valSet));
net = trainnet(dsTrain, net, "crossentropy", trainOpts);
saveModelWithMetadata('trained_baseline_resnet50.mat', net, struct( ...
    'inputSize', [224 224], 'inputChannels', 3, 'inputPolicy', 'raw RGB (deliberate ablation - no enhancement, no fusion)', ...
    'classMapping', 'categorical(grades,0:4) -> onehot, pred=argmax-1, referable>=2', ...
    'loss', 'crossentropy (no ordinal penalty - single-technique baseline)', ...
    'seed', gcfgB.seed, 'trainCounts', histcounts([trainSet.icdrGrade], -0.5:4.5), ...
    'valCounts', histcounts([valSet.icdrGrade], -0.5:4.5), ...
    'validationStory', validationStory, 'dataManifest', dataManifestB));
fprintf('Run compareModels.m once trained_dr_grader.mat also exists.\n');
