function [trainTbl, valTbl, splitReport] = splitSegmentationDataset(imagePaths, maskPaths, varargin)
% splitSegmentationDataset: Leakage-safe train/val splitter for
% segmentation. Replaces train_UNet_Segmentation.m's inline
% splitEachLabel_manual (pure randperm, no seed, no patient grouping).
%
% WHY: the same eye/patient appearing on both sides of a split silently
% inflates Dice/IoU (see buildPatientLevelSplit.m). Segmentation never
% used that utility. This wraps the same grouping principle for the
% paired image/mask lists built by buildSegmentationFileLists.m.
%
% INPUTS:
%   imagePaths, maskPaths - paired cell arrays (same length, index-aligned)
%   Name/Value options:
%     'ValFraction'  - default segmentationConfig.valFraction (0.15)
%     'Seed'         - default segmentationConfig.seed (42)
%     'PatientIdFcn' - fcn handle imagePath -> group ID string. Default []
%                      (= image-level split, logged as explicit choice).
%                      Pass @(p)fileparts-based or dataset-specific parser
%                      where a patient/eye key exists (e.g. Messidor pairs).
%     'SourceLabels' - optional cellstr per image (e.g. 'STARE','HRF') for
%                      reporting only; split is group-random, proportions
%                      are reported, not forced.
%     'TargetName'   - for messages (default 'segmentation')
%
% OUTPUTS:
%   trainTbl, valTbl - structs with .imagePaths, .maskPaths (cell arrays)
%   splitReport - struct with .nTrain, .nVal, .nGroups, .seed,
%                 .isPatientLevel, .sourceBreakdown
%
% Requires: nothing beyond base MATLAB.

p = inputParser();
p.addParameter('ValFraction', [], @isnumeric);
p.addParameter('Seed', [], @isnumeric);
p.addParameter('PatientIdFcn', [], @(f) isempty(f) || isa(f,'function_handle'));
p.addParameter('SourceLabels', {}, @(c) iscell(c) || isstring(c));
p.addParameter('TargetName', 'segmentation', @(s) ischar(s) || isstring(s));
p.parse(varargin{:});
opt = p.Results;

cfg = segmentationConfig();
if isempty(opt.ValFraction), opt.ValFraction = cfg.valFraction; end
if isempty(opt.Seed),        opt.Seed = cfg.seed; end
targetName = char(opt.TargetName);

n = numel(imagePaths);
assert(numel(maskPaths) == n, ...
    'splitSegmentationDataset:pairMismatch', ...
    'imagePaths (%d) and maskPaths (%d) lengths differ - build them with buildSegmentationFileLists.', n, numel(maskPaths));
assert(n >= 2, 'splitSegmentationDataset:tooFew', '%s: need >=2 pairs, got %d.', targetName, n);
assert(opt.ValFraction > 0 && opt.ValFraction < 1, ...
    'splitSegmentationDataset:badFraction', 'ValFraction must be in (0,1), got %.3f.', opt.ValFraction);

% Group IDs: patient-level if a key fcn is given, else image-level.
if isempty(opt.PatientIdFcn)
    groupIds = imagePaths(:); % every image its own group
    isPatientLevel = false;
else
    groupIds = cell(n, 1);
    for i = 1:n
        groupIds{i} = char(string(opt.PatientIdFcn(imagePaths{i})));
    end
    groupIds = groupIds(:);
    isPatientLevel = true;
end

[uniqueGroups, ~, groupIdx] = unique(groupIds, 'stable');
nGroups = numel(uniqueGroups);

rng(opt.Seed); % deterministic: same seed + same sorted file list = same split
order = randperm(nGroups);
nValGroups = max(round(opt.ValFraction * nGroups), 1);
nValGroups = min(nValGroups, nGroups - 1); % keep >=1 train group
valGroups = uniqueGroups(order(1:nValGroups));
trainGroups = uniqueGroups(order(nValGroups+1:end));
isValGroup = ismember(groupIds, valGroups);

trainIdx = find(~isValGroup);
valIdx   = find(isValGroup);

trainTbl = struct('imagePaths', {imagePaths(trainIdx)}, 'maskPaths', {maskPaths(trainIdx)});
valTbl   = struct('imagePaths', {imagePaths(valIdx)},   'maskPaths', {maskPaths(valIdx)});

% Leakage audit: no group may appear on both sides.
assert(isempty(intersect(unique(groupIds(trainIdx)), unique(groupIds(valIdx)))), ...
    'splitSegmentationDataset:leakage', 'INTERNAL ERROR: group appears on both sides.');

% Source breakdown for reporting (not enforced).
sourceBreakdown = struct();
if ~isempty(opt.SourceLabels)
    src = cellstr(opt.SourceLabels(:));
    assert(numel(src) == n, 'SourceLabels length must match imagePaths.');
    for s = unique(src)'
        sName = s{1};
        sourceBreakdown.(matlab.lang.makeValidName(sName)) = struct( ...
            'train', sum(strcmp(src(trainIdx), sName)), ...
            'val',   sum(strcmp(src(valIdx), sName)));
    end
end

splitReport = struct('nTrain', numel(trainIdx), 'nVal', numel(valIdx), ...
    'nGroups', nGroups, 'nTrainGroups', numel(trainGroups), 'nValGroups', numel(valGroups), ...
    'seed', opt.Seed, 'valFraction', opt.ValFraction, ...
    'isPatientLevel', isPatientLevel, 'sourceBreakdown', sourceBreakdown);

fprintf(['%s split (seed %d): %d pairs -> %d train / %d val (%d groups: %d train / %d val). %s\n'], ...
    targetName, opt.Seed, n, numel(trainIdx), numel(valIdx), ...
    nGroups, numel(trainGroups), numel(valGroups), ...
    ternary(isPatientLevel, 'Patient/group-level (no group crosses the split).', ...
        'NOTE: image-level split (no PatientIdFcn - every image its own group). See buildPatientLevelSplit.m header for which datasets lack a grouping key.'));
end

% ------------------------------------------------------------------
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
