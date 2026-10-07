function verdict = acceptDataset(name, varargin)
% acceptDataset: Dataset acceptance gate against structured registry
% metadata (datasetRegistry.m) - never prose sizeHint parsing, never
% URL scraping from accessNote.
%
% For each check the verdict records an explicit state; exact expected
% counts apply ONLY where the registry holds authoritative integers -
% otherwise COUNT_NOT_VERIFIED (an approximate "~" figure never becomes
% a hard requirement).
%
% *** UNEXECUTED against real data here (data/ absent): run in MATLAB
% with datasets present. Contract tests cover the decision logic on
% synthetic fixtures (MATLAB suite) and independent mirrors (Python).
%
% INPUTS: name (registry name), 'Registry' (override struct),
%   'GradeTable' (struct array with .imageId/.labelICDR, optional -
%   enables label parse/range checks), 'ImageExts' (default {jpg,jpeg,
%   png,tif,tiff,bmp}).
% OUTPUT: verdict struct(name, sourceStatus, versionStatus,
%   expectedImages, observedImages, imageCountStatus, expectedLabels,
%   parsedLabels, labelRangeValid, corruptCount, corruptFiles{},
%   available, exclusions{}, warnings{}, overallStatus, provenance).

p = inputParser;
addParameter(p, 'Registry', datasetRegistry(), @isstruct);
addParameter(p, 'GradeTable', [], @(x) true);
addParameter(p, 'ImageExts', {'jpg','jpeg','png','tif','tiff','bmp'}, @iscell);
parse(p, varargin{:});

ix = find(strcmp({p.Results.Registry.name}, name), 1);
if isempty(ix)
    error('acceptDataset:unknown - dataset %s not in registry.', name);
end
entry = p.Results.Registry(ix);
warnings = {};
exclusions = {};

% Source / version (recorded, never invented).
if strcmp(entry.sourceType, 'UNKNOWN') || strcmp(entry.sourceUrl, 'UNKNOWN')
    sourceStatus = 'UNVERIFIED (source identity not established)';
    warnings{end+1} = 'sourceUrl/sourceType UNKNOWN - confirm mirror-vs-official before citing provenance.';
else
    sourceStatus = sprintf('%s (%s)', entry.sourceUrl, entry.sourceType);
end
versionStatus = sprintf('policy=%s (record acquisition date/version at download)', entry.versionPolicy);

% File inventory.
files = [];
for e = 1:numel(p.Results.ImageExts)
    files = [files; dir(fullfile(entry.localDir, ['*.' p.Results.ImageExts{e}]))]; %#ok<AGROW>
end
observedImages = numel(files);
if isnan(entry.expectedImageCount)
    imageCountStatus = 'COUNT_NOT_VERIFIED (no authoritative count in registry)';
    expectedImages = NaN;
else
    expectedImages = entry.expectedImageCount;
    if observedImages == expectedImages
        imageCountStatus = 'MATCH';
    else
        imageCountStatus = sprintf('MISMATCH (expected %d, observed %d)', expectedImages, observedImages);
        warnings{end+1} = imageCountStatus;
    end
end

% Corrupt / unreadable scan.
corruptFiles = {};
for i = 1:numel(files)
    try
        imread(fullfile(files(i).folder, files(i).name));
    catch
        corruptFiles{end+1} = files(i).name; %#ok<AGROW>
    end
end
corruptCount = numel(corruptFiles);

% Label parse + range (only when a grade table is supplied).
parsedLabels = 0; labelRangeValid = 'NOT_APPLICABLE (no grade table supplied)';
expectedLabels = entry.expectedLabelCount;
if ~isempty(p.Results.GradeTable)
    gt = p.Results.GradeTable;
    parsedLabels = numel(gt);
    if all(ismember([gt.labelICDR], 0:4))
        labelRangeValid = 'VALID (all ICDR in {0..4})';
    else
        labelRangeValid = 'INVALID (out-of-range ICDR present)';
    end
end

available = logical(entry.available);
if ~available
    exclusions{end+1} = 'dataset absent from localDir - excluded from all roles';
end
if corruptCount > 0
    exclusions{end+1} = sprintf('%d corrupt/unreadable files excluded', corruptCount);
end
if strcmp(labelRangeValid, 'INVALID (out-of-range ICDR present)')
    overall = 'REJECTED';
else
    overall = 'ACCEPTED_WITH_NOTES';
end

verdict = struct('name', name, 'sourceStatus', sourceStatus, 'versionStatus', versionStatus, ...
    'expectedImages', expectedImages, 'observedImages', observedImages, ...
    'imageCountStatus', imageCountStatus, 'expectedLabels', expectedLabels, ...
    'parsedLabels', parsedLabels, 'labelRangeValid', labelRangeValid, ...
    'corruptCount', corruptCount, 'corruptFiles', {corruptFiles}, ...
    'available', available, 'exclusions', {exclusions}, 'warnings', {warnings}, ...
    'overallStatus', overall, ...
    'provenance', sprintf('registry=%s; policy=%s', entry.sourceType, entry.versionPolicy));
end
