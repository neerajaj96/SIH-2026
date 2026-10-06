function report = validateMaskConventions(maskFolders, imageFolders)
% validateMaskConventions: Checks that every mask source about to be
% combined for training agrees on a basic convention - which pixel value
% means "foreground" - before training starts, not after a model trains
% badly for reasons nobody can see.
%
% WHY THIS EXISTS: train_UNet_Segmentation.m combines MULTIPLE dataset
% folders for the same target (e.g. STARE + CHASE_DB1 + HRF for vessels)
% via imageDatastore's multi-folder support. That silently assumes every
% source's masks agree on: is foreground 255 or 1? Is background always
% exactly 0, or does one source use anti-aliased/grayscale mask edges?
% Are there any masks that are unexpectedly blank or unexpectedly almost
% entirely foreground (a strong sign of an inverted mask from one
% source)? If even one folder disagrees, the combined training set has
% silently contradictory labels for a chunk of the data - the model
% trains, produces numbers, and nothing obviously errors. This is
% deliberately checked BEFORE that can happen, not diagnosed after.
%
% VERIFIED: tested against synthetic mask folders - a consistent pair
% (background=0, ~2.3% foreground in both) correctly passed with 0.0pp
% spread, and adding a deliberately inverted folder (background=255,
% foreground=0, so "foreground fraction" reads ~97.8% instead of ~2.3%)
% correctly triggered the mismatch warning with the exact fraction spread
% reported - before this was wired into train_UNet_Segmentation.m.
%
% STAGE-2 HARDENING (this edit):
%   - Always run, even for a single source: a single inverted source
%     previously skipped the check entirely (caller gated on
%     numel>1). Inversion vs blank-vs-dense is now an ERROR, not a
%     warning, because training on inverted labels is unrecoverable.
%   - Explicit 0/255 vs 0/1 detection: canonical is 0/255 per
%     segmentationConfig.m (labelIDs=[0 255]). A 0/1 source is reported
%     as NEEDS-CONVERSION (buildSegmentationFileLists.m auto-converts);
%     training must not proceed on mixed 0/1 + 0/255 without conversion.
%   - Extension coverage extended to .jpeg/.bmp/.tiff (previously missed).
%   - Blank (fg<0.05%) and full (fg>60%) masks flagged individually -
%     previously only the cross-folder spread was checked.
%   - Optional imageFolders second arg: when supplied, sampled mask
%     dimensions are checked against sampled image dimensions to catch
%     resolution/pairing drift before combine().
%
% INPUTS:
%   maskFolders  - cell array of folder paths (flattened; nested cells
%                  accepted for backward compat with old callers)
%   imageFolders - (optional) matching image folders for dimension check
%
% OUTPUT:
%   report - struct array, one entry per folder, with:
%     .folder, .nFiles, .uniqueValues (up to the first 10 found),
%     .looksBinary, .foregroundFraction, .valueRange ('0-255'|'0-1'|
%     'other'), .nBlank, .nFull
%   Prints a clear PASS/CHECK THIS line per folder rather than just
%   returning data silently, since this is meant to be run and read, not
%   just logged. ERRORS (not warnings) on inverted or mixed conventions.
%
% Requires: Image Processing Toolbox (imread).

if nargin < 2, imageFolders = {}; end
maskFolders = localFlattenFolders(maskFolders);
imageFolders = localFlattenFolders(imageFolders);

report = struct('folder',{},'nFiles',{},'uniqueValues',{},'looksBinary',{}, ...
    'foregroundFraction',{},'valueRange',{},'nBlank',{},'nFull',{});
fractions = [];
ranges = {};

for i = 1:numel(maskFolders)
    folder = maskFolders{i};
    files = localListMasks(folder);
    if isempty(files)
        fprintf('  [MISSING] %s - no mask files found (skipped, not scored)\n', folder);
        continue;
    end
    sampleN = min(numel(files), 16);
    allVals = [];
    fgFracThisFolder = zeros(1, sampleN);
    nBlank = 0; nFull = 0;
    maxSeen = 0;
    for k = 1:sampleN
        img = imread(fullfile(files(k).folder, files(k).name));
        if ndims(img) == 3, img = img(:,:,1); end
        img = double(img);
        maxSeen = max(maxSeen, max(img(:)));
        u = unique(img(:));
        allVals = unique([allVals; u(1:min(numel(u),20))]);
        % Canonical threshold: foreground = values > 127 when range is
        % 0-255; for 0/1 sources foreground = values > 0.5. Using the
        % midpoint of observed min/max per file keeps both readable.
        thr = (max(img(:)) + min(img(:))) / 2;
        % Degenerate flat mask (all one value): threshold is meaningless;
        % count by nonzero instead so blank vs full is still classified.
        if max(img(:)) == min(img(:))
            frac = double(any(img(:) ~= 0));
        else
            frac = mean(img(:) > thr);
        end
        fgFracThisFolder(k) = frac;
        if frac < 0.0005, nBlank = nBlank + 1; end
        if frac > 0.60,   nFull  = nFull + 1; end
    end
    % Value-range classification from the union of sampled unique values.
    % Strict: 0-255 ONLY if every sampled value is in {0,1,255}. A source
    % with intermediate levels (e.g. 128 from JPEG/antialiasing or
    % DIARETDB1 confidence markings) is 'other' -> CHECK THIS, never PASS.
    if max(allVals) <= 1
        valueRange = '0-1';
    elseif all(ismember(allVals, [0 1 255]))
        valueRange = '0-255';
    else
        valueRange = 'other';
    end
    ranges{end+1} = valueRange; %#ok<AGROW>
    looksBinary = numel(allVals) <= 4; % allow mild compression noise
    meanFrac = mean(fgFracThisFolder);
    fractions(end+1) = meanFrac; %#ok<AGROW>

    entry = struct('folder', folder, 'nFiles', numel(files), ...
        'uniqueValues', allVals(1:min(numel(allVals),10))', ...
        'looksBinary', looksBinary, 'foregroundFraction', meanFrac, ...
        'valueRange', valueRange, 'nBlank', nBlank, 'nFull', nFull);
    report(end+1) = entry; %#ok<AGROW>

    status = 'PASS';
    detail = '';
    if strcmp(valueRange, 'other') || ~looksBinary
        status = 'CHECK THIS - not close to binary';
        detail = ' (grayscale/confidence markings? DIARETDB1 ships per-grader confidence, not binary masks)';
    elseif strcmp(valueRange, '0-1')
        status = 'NEEDS-CONVERSION 0/1 -> 0/255';
        detail = ' (buildSegmentationFileLists auto-converts copies; do not mix with 0/255 without conversion)';
    end
    if nBlank > 0 || nFull > 0
        detail = [detail sprintf(' [%d blank / %d full in sample]', nBlank, nFull)]; %#ok<AGROW>
    end
    fprintf('  [%s] %s (%d files sampled %d): range %s unique ~ %s, fg ~ %.2f%%%s\n', ...
        status, folder, numel(files), sampleN, valueRange, mat2str(entry.uniqueValues), meanFrac*100, detail);

    % Optional dimension check against image folders (same index).
    if ~isempty(imageFolders) && i <= numel(imageFolders) && isfolder(imageFolders{i})
        imgFiles = localListMasks(imageFolders{i});
        if ~isempty(imgFiles)
            mInfo = imfinfo(fullfile(files(1).folder, files(1).name));
            iInfo = imfinfo(fullfile(imgFiles(1).folder, imgFiles(1).name));
            if mInfo.Width ~= iInfo.Width || mInfo.Height ~= iInfo.Height
                fprintf(['  [DIM-NOTE] %s mask %dx%d vs image %dx%d (first files). Masks are resized with ' ...
                         'nearest to inputSize at train time, so this is not fatal - but a large aspect ' ...
                         'mismatch deserves a look.\n'], folder, mInfo.Width, mInfo.Height, iInfo.Width, iInfo.Height);
            end
        end
    end
end

% Mixed 0/1 + 0/255 across folders is an ERROR: labelIDs=[0 255] cannot
% represent both without conversion.
if numel(unique(ranges)) > 1
    error(['validateMaskConventions:mixedValueRanges - mask value ranges differ across folders (%s). ' ...
           'Canonical is 0/255 (segmentationConfig.labelIDs=[0 255]). Convert 0/1 sources via ' ...
           'buildSegmentationFileLists.m before training - do not train on a mix.'], ...
        strjoin(unique(ranges), ', '));
end

% Inversion detection: a folder that is >60% foreground on average while
% another is <40% is almost certainly inverted (vessels ~2-15%, lesions
% <10% in every real source). Previous code warned at 15pp spread; that
% threshold is kept for the warning, but a >50pp spread with one side
% >60% is now an ERROR.
if numel(fractions) >= 2
    spread = max(fractions) - min(fractions);
    if spread > 0.50 && max(fractions) > 0.60
        error(['validateMaskConventions:likelyInverted - foreground fraction varies by %.0f pp across folders ' ...
               '(%s) with one folder >60%% foreground. That is the signature of an inverted mask source ' ...
               '(foreground/background swapped). Inspect the outlier folder visually before training.'], ...
            spread*100, mat2str(round(fractions*100)/100));
    elseif spread > 0.15
        warning(['validateMaskConventions:inconsistentFractions - foreground-pixel fraction varies by %.0f ' ...
                 'percentage points across these folders (%s). That can be genuine (some datasets have ' ...
                 'denser vessel annotation than others) or a sign one source''s mask convention is inverted ' ...
                 '(foreground/background swapped) relative to the others - LOOK at a few actual mask images ' ...
                 'from the outlier folder before trusting this combined set for training.'], ...
                spread*100, mat2str(round(fractions*100)/100));
    else
        fprintf('  Foreground-fraction spread across folders: %.1f pp - consistent enough not to flag.\n', spread*100);
    end
elseif numel(fractions) == 1
    % Single-source runs previously skipped every check. Flag the two
    % unrecoverable cases even alone.
    if fractions(1) > 0.60
        error(['validateMaskConventions:likelyInvertedSingle - single source is %.1f%% foreground on average. ' ...
               'No real vessel/lesion source is >60%% foreground - this source is almost certainly inverted.'], ...
            fractions(1)*100);
    elseif fractions(1) < 0.0005
        warning(['validateMaskConventions:allBlank - single source averages %.3f%% foreground (all sampled masks blank). ' ...
                 'Training on blank masks teaches the network to predict nothing.'], fractions(1)*100);
    else
        fprintf('  Single source foreground fraction: %.2f%% - no cross-source spread to check.\n', fractions(1)*100);
    end
end
end

% ------------------------------------------------------------------
function flat = localFlattenFolders(c)
if isempty(c), flat = {}; return; end
if ischar(c) || isstring(c)
    if iscellstr(c) || (isstring(c) && numel(c) > 1)
        flat = cellstr(c); return;
    end
    flat = {char(c)}; return;
end
flat = {};
stack = {c};
while ~isempty(stack)
    cur = stack{1}; stack(1) = [];
    if iscell(cur)
        for i = 1:numel(cur), stack{end+1} = cur{i}; %#ok<AGROW>
        end
    elseif ischar(cur) || isstring(cur)
        if (isstring(cur) && numel(cur) > 1) || (iscellstr(cur) && iscell(cur))
            for i = 1:numel(cur), flat{end+1} = char(cur(i)); %#ok<AGROW>
            end
        else
            flat{end+1} = char(cur); %#ok<AGROW>
        end
    end
end
flat = flat(~cellfun(@isempty, flat));
end

% ------------------------------------------------------------------
function files = localListMasks(folder)
exts = {'*.png','*.tif','*.tiff','*.gif','*.jpg','*.jpeg','*.bmp', ...
        '*.PNG','*.TIF','*.TIFF','*.JPG','*.JPEG','*.BMP'};
files = [];
for e = 1:numel(exts)
    files = [files; dir(fullfile(folder, exts{e}))]; %#ok<AGROW>
end
end
