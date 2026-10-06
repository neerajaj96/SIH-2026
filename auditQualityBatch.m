function summaryTable = auditQualityBatch(imagesDir, outputDir, varargin)
% auditQualityBatch: Lightweight file-list quality audit for training
% curation. Stage P1+P2+P3 hardening (NOT model training).
%
% WHY THIS EXISTS: every training entry point calls the quality core with
% -Inf thresholds (enhance-only, never rejects), so FAIL-quality files
% used to flow silently into training/validation splits. This makes that
% path auditable: each file gets a quality decision + reason codes in a
% CSV manifest WITHOUT requiring model inference (no U-Nets, no grader,
% no getOrLoadCachedModels - assessFundusQuality only).
%
% CURATION POLICY (explicit, never silent):
%   'CurationPolicy','audit-only' (default): report only; returns the full
%     table including FAIL rows. Caller decides what enters training.
%   'CurationPolicy','enforce': additionally returns ONLY gradeable rows
%     as second output and logs how many FAIL rows were excluded.
% Nothing is deleted from disk either way.
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB here) ***
%
% INPUTS:
%   imagesDir - folder of .jpg/.jpeg/.png/.tif fundus photos
%   outputDir - manifest CSV destination (created if missing)
%
% OUTPUT:
%   summaryTable - table, one row per image: filename, decision,
%     isGradeable, focus, entropy, focusThresh, entropyThresh,
%     qualityCalibrated, coverage, circularity, underexpFrac, overexpFrac,
%     specularFrac, contrastP95P5, reasons (| -joined), guidance (| -joined)
%
% Requires: Image Processing Toolbox (via assessFundusQuality). No
%   trained weights, no Deep Learning Toolbox.

p = inputParser;
addParameter(p, 'CurationPolicy', 'audit-only', @(x) ismember(x, {'audit-only','enforce'}));
parse(p, varargin{:});

if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(imagesDir, 'quality_audit');
end
if ~isfolder(outputDir), mkdir(outputDir); end

files = [dir(fullfile(imagesDir,'*.jpg')); dir(fullfile(imagesDir,'*.jpeg')); ...
         dir(fullfile(imagesDir,'*.png')); dir(fullfile(imagesDir,'*.tif'))];
if isempty(files)
    error('auditQualityBatch:noImages', 'No image files found in %s', imagesDir);
end
fprintf('Auditing quality for %d images (policy: %s, no model inference)...\n', numel(files), p.Results.CurationPolicy);

n = numel(files);
rows = cell(n, 16);
for i = 1:n
    fname = files(i).name;
    try
        raw = imread(fullfile(files(i).folder, fname));
        rep = assessFundusQuality(raw);
        reasons = strjoin(rep.reasons, ' | ');
        guide = strjoin(rep.recaptureGuidance, ' | ');
        if isempty(reasons), reasons = ''; end
        if isempty(guide), guide = ''; end
        rows(i,:) = {fname, rep.decision, double(rep.isGradeable), ...
            rep.focusScore, rep.entropyScore, rep.focusThresh, rep.entropyThresh, ...
            double(rep.isCalibrated), rep.roiInfo.coverageFrac, rep.roiInfo.circularity, ...
            rep.scores.underexposedFrac, rep.scores.overexposedFrac, ...
            rep.scores.specularFrac, rep.scores.contrastP95P5, reasons, guide};
    catch ME
        rows(i,:) = {fname, 'ERROR', 0, NaN, NaN, NaN, NaN, 0, NaN, NaN, NaN, NaN, NaN, NaN, ME.message, ''};
    end
    if mod(i,100) == 0 || i == n
        fprintf('  %d/%d audited\n', i, n);
    end
end

summaryTable = cell2table(rows, 'VariableNames', {'filename','decision','isGradeable', ...
    'focus','entropy','focusThresh','entropyThresh','qualityCalibrated', ...
    'coverage','circularity','underexpFrac','overexpFrac','specularFrac','contrastP95P5', ...
    'reasons','guidance'});

outCsv = uniqueArtifactPath(outputDir, 'quality_audit', '.csv');
writetable(summaryTable, outCsv);

nPass = sum(strcmp(summaryTable.decision,'PASS'));
nBorder = sum(strcmp(summaryTable.decision,'BORDERLINE'));
nFail = sum(strcmp(summaryTable.decision,'FAIL'));
nErr = sum(strcmp(summaryTable.decision,'ERROR'));
fprintf('\n=== Quality audit complete: %s ===\n', outCsv);
fprintf('PASS: %d | BORDERLINE: %d | FAIL: %d | ERROR: %d\n', nPass, nBorder, nFail, nErr);
if strcmp(p.Results.CurationPolicy, 'enforce')
    keep = ~strcmp(summaryTable.decision,'FAIL') & ~strcmp(summaryTable.decision,'ERROR');
    fprintf('Policy=enforce: %d/%d files gradeable; FAIL/ERROR rows EXCLUDED from returned table (nothing deleted on disk).\n', sum(keep), n);
    summaryTable = summaryTable(keep,:);
else
    fprintf('Policy=audit-only: all %d rows returned; NOTHING rejected automatically - apply curation explicitly.\n', n);
end
end
