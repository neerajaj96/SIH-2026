function [imagePaths, maskPaths, report] = buildSegmentationFileLists(imageFolders, maskFolders, targetName)
% buildSegmentationFileLists: Robust paired image<->mask file-list builder
% for train_UNet_Segmentation.m. Replaces the fragile pattern of passing
% raw folder lists straight into imageDatastore + pixelLabelDatastore and
% hoping alphabetical order aligns them.
%
% FIXES (all proven by code inspection, Stage-2):
%   1. Nested-cell bug: train_UNet_Segmentation.m built vesselSources as a
%      cell array then called fullfile(cell,'images') wrapped in {…},
%      producing a nested cell imageDatastore miscounts. This flattens
%      arbitrarily nested cells first.
%   2. Silent misalignment: imageDatastore(folder) + pixelLabelDatastore
%      (folder) + combine() pairs by read ORDER, with no check that
%      image #k matches mask #k. If one folder has an extra file, every
%      pair after it is wrong with no error. This pairs by basename stem
%      and ERRORS on any orphan.
%   3. Extension blindness: previous code relied on datastore defaults;
%      this accepts .png/.jpg/.jpeg/.tif/.tiff/.bmp/.gif explicitly.
%   4. 0/1 vs 0/255: canonical is 0/255 (see segmentationConfig.m). If a
%      source ships 0/1 masks, this detects it from a sample and converts
%      copies to 0/255 in a sibling <folder>_converted_255/ dir, returning
%      the converted paths - with a logged warning, not a silent mismatch
%      against labelIDs=[0 255].
%
% INPUTS:
%   imageFolders - char/string/cell (possibly nested) of image dirs
%   maskFolders  - char/string/cell (possibly nested) of mask dirs
%   targetName   - 'Vessels' | 'MicroaneurysmsHemorrhages' | 'Exudates'
%                  (used only for messages + DDR merge hint)
%
% OUTPUTS:
%   imagePaths, maskPaths - matched cell arrays (same length, paired by
%                           basename stem, sorted for reproducibility)
%   report - struct with .nImages, .nMasks, .nPaired, .convertedFolders,
%            .orphanImages, .orphanMasks
%
% Requires: Image Processing Toolbox (imread for the 0/1 probe only).

if nargin < 3, targetName = 'target'; end
imageFolders = localFlatten(imageFolders);
maskFolders  = localFlatten(maskFolders);

imgFiles  = localListImages(imageFolders);
maskFiles = localListImages(maskFolders);

% Index masks by stem (filename without extension). First occurrence wins;
% duplicates across sources are an error (ambiguous pairing).
maskByStem = containers.Map('KeyType','char','ValueType','char');
dupStems = {};
for i = 1:numel(maskFiles)
    [~, stem, ~] = fileparts(maskFiles{i});
    key = lower(stem); % case-insensitive: DRIVE ".tif" vs ".TIF" layouts
    if isKey(maskByStem, key)
        dupStems{end+1} = stem; %#ok<AGROW>
    else
        maskByStem(key) = maskFiles{i};
    end
end
if ~isempty(dupStems)
    error(['buildSegmentationFileLists:duplicateMaskStems - %d mask basename(s) appear in more than one mask folder ' ...
           '(e.g. "%s"). Combine sources with disjoint filenames or rename before training - silent overwrite would corrupt labels.'], ...
        numel(unique(dupStems)), dupStems{1});
end

imagePaths = {}; maskPaths = {};
orphanImages = {}; orphanMasks = {}; % orphans computed below
for i = 1:numel(imgFiles)
    [~, stem, ~] = fileparts(imgFiles{i});
    key = lower(stem);
    if isKey(maskByStem, key)
        imagePaths{end+1} = imgFiles{i}; %#ok<AGROW>
        maskPaths{end+1}  = maskByStem(key); %#ok<AGROW>
        remove(maskByStem, key);
    else
        orphanImages{end+1} = imgFiles{i}; %#ok<AGROW>
    end
end
% Whatever mask keys remain have no matching image.
orphanMasks = values(maskByStem);

if ~isempty(orphanImages) || ~isempty(orphanMasks)
    % Orphans are common when a source folder contains extra non-paired
    % files (e.g. DDR grading images mixed into a seg folder). Fail loudly
    % rather than training on a shifted pairing.
    msg = sprintf(['buildSegmentationFileLists:unpairedFiles - %s: %d paired, %d orphan image(s), %d orphan mask(s). ' ...
        'Paired by basename stem (case-insensitive, extension ignored). '], ...
        targetName, numel(imagePaths), numel(orphanImages), numel(orphanMasks));
    if ~isempty(orphanImages)
        msg = [msg sprintf('First orphan image: %s. ', orphanImages{1})]; %#ok<AGROW>
    end
    if ~isempty(orphanMasks)
        msg = [msg sprintf('First orphan mask: %s. ', orphanMasks{1})]; %#ok<AGROW>
    end
    if targetName == "MicroaneurysmsHemorrhages" || targetName == "Exudates"
        msg = [msg sprintf(['If this is DDR_seg, you likely have not yet merged its 4 per-class masks (MA/HE/EX/SE) into ' ...
            'masks_maho/ + masks_exudate/ - see mergeDDRSegMasks.m. '])]; %#ok<AGROW>
    end
    error(msg);
end

if isempty(imagePaths)
    error(['buildSegmentationFileLists:noPairs - %s: zero paired image/mask files. Checked %d image folder(s), %d mask folder(s). ' ...
           'Verify the manual sort into images/ + masks/ per train_UNet_Segmentation.m header.'], ...
        targetName, numel(imageFolders), numel(maskFolders));
end

% Sort pairs by image path for reproducibility (datastore order + randperm
% split with fixed seed then fully determines the train/val assignment).
[imagePaths, sIdx] = sort(imagePaths);
maskPaths = maskPaths(sIdx);

% 0/1 vs 0/255 probe: sample up to 8 masks, check max value.
convertedFolders = {};
sampleN = min(numel(maskPaths), 8);
looksZeroOne = true;
for k = 1:sampleN
    try
        m = imread(maskPaths{k});
    catch
        looksZeroOne = false; break;
    end
    if ndims(m) == 3, m = m(:,:,1); end
    if double(max(m(:))) > 1
        looksZeroOne = false; break;
    end
end
if looksZeroOne
    % All sampled masks max out at 1 - this source ships 0/1, not 0/255.
    % Convert COPIES (never overwrite source) so pixelLabelDatastore with
    % labelIDs=[0 255] reads correctly.
    warning(['buildSegmentationFileLists:zeroOneMasks - %s: sampled %d mask(s) max out at 1, not 255. ' ...
             'Converting copies to 0/255 in sibling _converted_255/ folders. Source folders untouched.'], ...
        targetName, sampleN);
    newMaskPaths = cell(size(maskPaths));
    for k = 1:numel(maskPaths)
        [mp, mn, me] = fileparts(maskPaths{k});
        convDir = [mp '_converted_255'];
        if ~isfolder(convDir), mkdir(convDir); end
        if ~ismember(convDir, convertedFolders), convertedFolders{end+1} = convDir; %#ok<AGROW>
        end
        dest = fullfile(convDir, [mn me]);
        if ~isfile(dest)
            m = imread(maskPaths{k});
            if ndims(m) == 3, m = m(:,:,1); end
            imwrite(uint8(logical(m)) * 255, dest);
        end
        newMaskPaths{k} = dest;
    end
    maskPaths = newMaskPaths;
end

report = struct('nImages', numel(imgFiles), 'nMasks', numel(maskFiles), ...
    'nPaired', numel(imagePaths), 'convertedFolders', {convertedFolders}, ...
    'orphanImages', {orphanImages}, 'orphanMasks', {orphanMasks});
fprintf('%s: paired %d image/mask file(s) from %d image folder(s) + %d mask folder(s).\n', ...
    targetName, report.nPaired, numel(imageFolders), numel(maskFolders));
if ~isempty(convertedFolders)
    fprintf('  Converted 0/1 -> 0/255 in: %s\n', strjoin(convertedFolders, ', '));
end
end

% ------------------------------------------------------------------
function flat = localFlatten(c)
% Accepts char/string/cell (arbitrarily nested, including the
% {fullfile(cell,'x')} double-wrap) and returns a flat cellstr.
if ischar(c) || isstring(c)
    flat = cellstr(c);
    return;
end
flat = {};
stack = {c};
while ~isempty(stack)
    cur = stack{1}; stack(1) = [];
    if iscell(cur)
        for i = 1:numel(cur), stack{end+1} = cur{i}; %#ok<AGROW>
        end
    elseif ischar(cur) || isstring(cur)
        % fullfile over a cell produces a cell; a plain path stays plain.
        % Expand any multi-element cellstr element-wise.
        if iscellstr(cur) || (isstring(cur) && numel(cur) > 1)
            for i = 1:numel(cur), flat{end+1} = char(cur(i)); %#ok<AGROW>
            end
        else
            flat{end+1} = char(cur); %#ok<AGROW>
        end
    end
end
% Also expand fullfile-style vectorized results: if any entry contains a
% cell-vector artifact, it is already split above. Dedupe + drop empties.
flat = unique(flat(~cellfun(@isempty, flat)));
end

% ------------------------------------------------------------------
function files = localListImages(folders)
exts = {'*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp','*.gif', ...
        '*.PNG','*.JPG','*.JPEG','*.TIF','*.TIFF','*.BMP'};
files = {};
for f = 1:numel(folders)
    if ~isfolder(folders{f})
        error('buildSegmentationFileLists:missingFolder - folder not found: %s', folders{f});
    end
    for e = 1:numel(exts)
        d = dir(fullfile(folders{f}, exts{e}));
        for i = 1:numel(d)
            % Skip directories + the _converted_255 siblings unless asked.
            if ~d(i).isdir
                files{end+1} = fullfile(d(i).folder, d(i).name); %#ok<AGROW>
            end
        end
    end
end
% Dedupe (upper/lowercase ext double-glob on case-insensitive FS).
files = unique(files);
end
