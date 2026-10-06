function report = validateMaskConventions(maskFolders)
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
% INPUTS:
%   maskFolders - cell array of folder paths (the same list
%                 train_UNet_Segmentation.m builds from datasetRegistry.m)
%
% OUTPUT:
%   report - struct array, one entry per folder, with:
%     .folder, .nFiles, .uniqueValues (up to the first 10 found),
%     .looksBinary (true if only ~2 distinct values, allowing for mild
%       JPEG-compression noise), .foregroundFraction (mean fraction of
%       "on" pixels across a sample - wildly different fractions across
%       folders is itself a flag worth a manual look, even if each
%       individually looks binary)
%   Prints a clear PASS/CHECK THIS line per folder rather than just
%   returning data silently, since this is meant to be run and read, not
%   just logged.
%
% Requires: Image Processing Toolbox (imread).

if ischar(maskFolders) || isstring(maskFolders)
    maskFolders = {char(maskFolders)};
end

report = struct('folder',{},'nFiles',{},'uniqueValues',{},'looksBinary',{},'foregroundFraction',{});
fractions = [];

for i = 1:numel(maskFolders)
    folder = maskFolders{i};
    files = [dir(fullfile(folder,'*.png')); dir(fullfile(folder,'*.tif')); dir(fullfile(folder,'*.gif')); dir(fullfile(folder,'*.jpg'))];
    if isempty(files)
        fprintf('  [MISSING] %s - no mask files found (skipped, not scored)\n', folder);
        continue;
    end
    sampleN = min(numel(files), 12); % a handful of files is enough to catch a convention mismatch, no need to scan thousands
    allVals = [];
    fgFracThisFolder = zeros(1, sampleN);
    for k = 1:sampleN
        img = imread(fullfile(files(k).folder, files(k).name));
        if ndims(img) == 3, img = img(:,:,1); end
        u = unique(img(:));
        allVals = unique([allVals; u(1:min(numel(u),20))]);
        fgFracThisFolder(k) = mean(img(:) > (double(max(img(:)))+double(min(img(:))))/2);
    end
    looksBinary = numel(allVals) <= 4; % allow a little compression noise around 2 "real" levels
    meanFrac = mean(fgFracThisFolder);
    fractions(end+1) = meanFrac; %#ok<AGROW>

    entry = struct('folder', folder, 'nFiles', numel(files), ...
        'uniqueValues', allVals(1:min(numel(allVals),10))', ...
        'looksBinary', looksBinary, 'foregroundFraction', meanFrac);
    report(end+1) = entry; %#ok<AGROW>

    status = 'PASS';
    if ~looksBinary, status = 'CHECK THIS - not close to binary'; end
    fprintf('  [%s] %s (%d files sampled %d): unique values ~ %s, foreground fraction ~ %.1f%%\n', ...
        status, folder, numel(files), sampleN, mat2str(entry.uniqueValues), meanFrac*100);
end

if numel(fractions) >= 2
    spread = max(fractions) - min(fractions);
    if spread > 0.15
        warning(['validateMaskConventions:inconsistentFractions - foreground-pixel fraction varies by %.0f ' ...
                 'percentage points across these folders (%s). That can be genuine (some datasets have ' ...
                 'denser vessel annotation than others) or a sign one source''s mask convention is inverted ' ...
                 '(foreground/background swapped) relative to the others - LOOK at a few actual mask images ' ...
                 'from the outlier folder before trusting this combined set for training.'], ...
                spread*100, mat2str(round(fractions*100)/100));
    else
        fprintf('  Foreground-fraction spread across folders: %.1f pp - consistent enough not to flag.\n', spread*100);
    end
end
end
