function summaryTable = runBatchScreening(imagesDir, outputDir)
% runBatchScreening: Runs the screening pipeline over a WHOLE FOLDER of
% patient images and writes one summary CSV, instead of the single
% hardcoded sample.jpg production_inference.m processes. The official PS
% asks for a system serving "100,000+ patients annually" - nothing in
% this project could previously process more than one image without
% manually re-running the script and renaming files.
%
% Deliberately does NOT generate a full PDF per patient (that's what
% production_inference.m is for, on a single case you want to inspect in
% detail) - this is the fast triage pass: quality gate, both grading
% pathways, and the flags that matter for routing, one row per patient.
%
% BEHAVIOR ON A BAD IMAGE: logs the failure and CONTINUES to the next
% image, rather than halting the whole batch - a single corrupt or
% unreadable file in a folder of thousands should not stop the other
% 999+ from being screened. Compare this to production_inference.m's
% error() on a failed quality gate, which is correct for a single
% interactive case but wrong for unattended batch processing.
%
% INPUTS:
%   imagesDir - folder of .jpg/.png fundus photos, one file per eye
%   outputDir - where to write the summary CSV (created if missing)
%
% OUTPUT:
%   summaryTable - table with one row per image: filename, gradeable,
%     focus, entropy, icdrGrade_deepLearning (NaN if simulated),
%     confidence, icdrGrade_clinicalRules, gradesAgree, nvFlagged,
%     gradCamOnDisc, referable (either grade >= 2), status, errorMessage,
%     qualityDecision (PASS/BORDERLINE/FAIL/ERROR - filter triage on this,
%     not status alone), qualityReasons (| -joined, may be empty)
%
% Requires: same toolboxes as production_inference.m.

if nargin < 2
    outputDir = fullfile(imagesDir, 'batch_results');
end
if ~isfolder(outputDir), mkdir(outputDir); end

files = [dir(fullfile(imagesDir,'*.jpg')); dir(fullfile(imagesDir,'*.jpeg')); dir(fullfile(imagesDir,'*.png'))];
if isempty(files)
    error('runBatchScreening:noImages', 'No .jpg/.jpeg/.png files found in %s', imagesDir);
end
fprintf('Found %d images. Starting batch run...\n', numel(files));

models = getOrLoadCachedModels();
haveTrainedModels = models.haveTrainedModels;
if ~haveTrainedModels
    warning(['runBatchScreening:simulated - no trained models found. Every icdrGrade_deepLearning value ' ...
             'in the output will be NaN, and icdrGrade_clinicalRules will be based on placeholder masks. ' ...
             'This still exercises the full pipeline and timing, but results are not real predictions.']);
end

n = numel(files);
rows = cell(n, 14);
for i = 1:n
    fname = files(i).name;
    res = screenOneImage(fullfile(files(i).folder, fname), models);

    icdrDL = res.dl; icdrRule = res.rule;
    agree = NaN; if ~isnan(icdrDL), agree = double(icdrDL == icdrRule); end
    nvFlag = double(res.nv);
    onDisc = double(res.gradCamOnDisc);
    referable = NaN;
    if ~isnan(icdrRule) || ~isnan(icdrDL)
        bestGrade = icdrDL; if isnan(bestGrade), bestGrade = icdrRule; end
        referable = double(bestGrade >= 2);
    end
    rows(i,:) = {fname, res.status, res.focus, res.entropy, icdrDL, res.conf, icdrRule, agree, nvFlag, onDisc, referable, res.errorMessage, ...
        res.qualityDecision, strjoin(res.qualityReasons, ' | ')};

    if mod(i,50) == 0 || i == n
        fprintf('  %d/%d processed\n', i, n);
    end
end

summaryTable = cell2table(rows, 'VariableNames', {'filename','status','focus','entropy', ...
    'icdrGrade_deepLearning','confidence','icdrGrade_clinicalRules','gradesAgree', ...
    'nvFlagged','gradCamOnDisc','referable','errorMessage','qualityDecision','qualityReasons'});

outCsv = fullfile(outputDir, sprintf('batch_summary_%s.csv', datestr(now,'yyyymmdd_HHMMSS')));
writetable(summaryTable, outCsv);

nOk = sum(strcmp(summaryTable.status,'ok'));
nUngradeable = sum(strcmp(summaryTable.status,'ungradeable'));
nError = sum(strcmp(summaryTable.status,'error'));
nBorderline = sum(strcmp(summaryTable.qualityDecision,'BORDERLINE'));
nReferable = sum(summaryTable.referable == 1);
nDisagree = sum(summaryTable.gradesAgree == 0);
nFlaggedGradCam = sum(summaryTable.gradCamOnDisc == 1);

fprintf('\n=== Batch complete: %s ===\n', outCsv);
fprintf('Processed: %d | Ungradeable: %d | Errors: %d | BORDERLINE (gradeable-with-warning): %d\n', nOk, nUngradeable, nError, nBorderline);
if haveTrainedModels
    fprintf('Referable (Level 2+): %d | AI/rule-engine disagreements: %d | Grad-CAM-on-disc flags: %d\n', ...
        nReferable, nDisagree, nFlaggedGradCam);
else
    fprintf('Ran in SIMULATED mode (no trained models) - referable/disagreement counts above are not real.\n');
end
if nError > 0
    fprintf('%d image(s) errored - see the errorMessage column, they were skipped rather than halting the batch.\n', nError);
end
end
