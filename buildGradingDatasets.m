function [trainSet, valSet, testSet, validationStory, manifest] = buildGradingDatasets()
% buildGradingDatasets: Builds the train/val/test split for DR severity
% grading, adapting to whichever datasets datasetRegistry() finds.
%
% STAGE-3 HARDENING (split identity preserved - same seeds/fractions as
% before so train_DR_Grader.m and train_Baseline_ResNet50.m still get
% IDENTICAL paired splits for compareModels.m McNemar validity):
%   - Grade-range gate: every label asserted in 0..4 (frozen ICDR mapping).
%   - Provenance: train/val/test structs now carry .patientId + .source
%     alongside .imagePath/.icdrGrade (empty string where the source
%     publishes no key - explicit, not missing).
%   - Class-distribution report per split + imbalance ratio (max/min).
%   - Disjointness asserts: train/val/test paths pairwise disjoint (a shared
%     file across splits is leakage, not augmentation).
%   - DDR_seg overlap warning: DDR_seg pixel-annotated images are a subset
%     of DDR grading images; the grader trains on DDR with seg-PREDICTED
%     masks, so overlapping IDs see memorized (not generalized) masks -
%     flagged in manifest, not silently ignored (see SEGMENTATION_HANDOFF).
%   - 5th output manifest (backward compatible - 4-output callers unaffected)
%     with case, seed, counts, class histograms, dataset snapshot, and the
%     no-test-tuning rule restated for the training log.
%
% TWO POSSIBLE OUTCOMES, and they are NOT equally strong evidence - say
% so explicitly in whichever one you end up with:
%
%   CASE A - DDR available: DDR (China, ~10,000 images) trains the model;
%   Messidor-2 (France, 1748 images) is held out ENTIRELY as the test
%   set - never touched for training, threshold selection, or
%   temperature calibration. This is genuine cross-population external
%   validation - different country, different clinical sites, different
%   cameras. This is actually a STRONGER validation story than the
%   original APTOS/IDRiD/DRIVE plan would have given you.
%
%   CASE B - DDR unavailable: only Messidor-2 exists, so it has to serve
%   as train, val, AND test simultaneously via an internal patient-level
%   split (via buildPatientLevelSplit.m, keyed on the same patient/eye
%   pairing used in buildMessidorTestSet.m - no single patient's images
%   cross a split boundary). This is legitimate, correctly-done ML
%   practice - but it is WITHIN-DATASET validation, not external
%   validation. Say "internally validated on a held-out patient split"
%   in your pitch, not "externally validated on an independent dataset" -
%   those are different claims and a judge who knows the difference will
%   notice if you conflate them.
%
% OUTPUTS:
%   trainSet, valSet, testSet - struct arrays with .imagePath, .icdrGrade
%   validationStory - string, exactly which case ran and why - print
%                      this directly into your report/pitch materials
%                      instead of re-deriving the wording each time.
%
% Requires: nothing beyond base MATLAB (plus assessAndEnhanceImage.m's
% dependencies indirectly, if you resize/enhance downstream).

registry = datasetRegistry();
availByName = containers.Map({registry.name}, {registry.available});
dirByName = containers.Map({registry.name}, {registry.localDir});

if availByName('DDR')
    fprintf('DDR found - using it to train, Messidor-2 held out entirely as the test set.\n');
    validationStory = ['Trained on DDR (~10,000 images, multiple clinical sites in China); ' ...
        'tested on Messidor-2 (1748 images, France), held out entirely - a genuine cross-' ...
        'population external validation, not an internal split.'];

    ddrDir = dirByName('DDR');
    % --- Adjust to match your actual downloaded DDR label format - I have
    % not seen the real files, so this is a best-effort default, not a
    % confirmed schema. DDR's HuggingFace reuploads have varied between a
    % single labels.csv and a folder-per-class layout across mirrors. ---
    ddrLabelsCsv = fullfile(ddrDir, 'labels.csv');
    DDR_IMAGE_COL = "image_id";
    DDR_GRADE_COL = "diagnosis";
    if isfile(ddrLabelsCsv)
        ddrLabels = readtable(ddrLabelsCsv, 'TextType', 'string');
        ddrPaths = {}; ddrGrades = [];
        for i = 1:height(ddrLabels)
            candidate = fullfile(ddrDir, ddrLabels.(DDR_IMAGE_COL)(i));
            if ~isfile(candidate)
                candidate = candidate + ".jpg"; % try appending an extension if the CSV stores bare stems
            end
            if isfile(candidate)
                ddrPaths{end+1} = char(candidate); %#ok<AGROW>
                ddrGrades(end+1) = double(ddrLabels.(DDR_GRADE_COL)(i)); %#ok<AGROW>
            end
        end
    else
        % Fallback: folder-per-class layout, e.g. data/DDR/images/0/*.jpg .. images/4/*.jpg
        ddrPaths = {}; ddrGrades = [];
        for g = 0:4
            classDir = fullfile(ddrDir, num2str(g));
            if isfolder(classDir)
                files = dir(fullfile(classDir, '*.jpg'));
                for i = 1:numel(files)
                    ddrPaths{end+1} = fullfile(files(i).folder, files(i).name); %#ok<AGROW>
                    ddrGrades(end+1) = g; %#ok<AGROW>
                end
            end
        end
    end
    if isempty(ddrPaths)
        error(['buildGradingDatasets:ddrFormatUnknown - found the DDR folder but could not locate ' ...
               'either labels.csv or a folder-per-class layout inside it. Open %s, see what''s ' ...
               'actually there, and adjust the label-reading block in this function accordingly.'], ddrDir);
    end
    fprintf('Loaded %d labeled DDR images.\n', numel(ddrPaths));

    % DDR publishes no patient/eye identifier - same situation
    % buildPatientLevelSplit.m already documented for APTOS/IDRiD - so this
    % is an honest image-level split, not a patient-level one.
    split = buildPatientLevelSplit(ddrPaths, @(p) p, 0.85, 0.15, 42); % 85/15 train/val - test comes entirely from Messidor-2, not from here
    trainSet = toStructArray(split, ddrGrades, 'train', 'DDR', {''});
    valSet   = toStructArray(split, ddrGrades, 'val', 'DDR', {''});

    testSet = buildMessidorTestSet(fullfile(dirByName('Messidor-2')), ...
        fullfile(dirByName('Messidor-2'), '..', 'messidor2_grades.csv'), ...
        fullfile(dirByName('Messidor-2'), '..', 'messidor-2.csv'));

else
    fprintf('DDR not found - falling back to a Messidor-2-only internal split. See this function''s header for what that does and does not prove.\n');
    validationStory = ['Only Messidor-2 was available. Split internally by patient (via the left/right ' ...
        'eye pairing) into train/val/test with buildPatientLevelSplit.m - a legitimate, leakage-safe ' ...
        'split, but WITHIN one dataset, not an external validation. Say "internally validated on a ' ...
        'held-out patient split", not "externally validated".'];

    messDir = dirByName('Messidor-2');
    fullSet = buildMessidorTestSet(messDir, fullfile(messDir, '..', 'messidor2_grades.csv'), ...
        fullfile(messDir, '..', 'messidor-2.csv'));
    paths = {fullSet.imagePath};
    grades = [fullSet.icdrGrade];
    patientIds = {fullSet.patientId};
    patientOf = containers.Map(paths, patientIds);

    split = buildPatientLevelSplit(paths, @(p) patientOf(p), 0.70, 0.15, 42);
    % Per-image patient IDs aligned to split order for provenance.
    pids = cellfun(@(p) char(patientOf(p)), paths, 'UniformOutput', false);
    trainSet = toStructArray(split, grades, 'train', 'Messidor-2', pids);
    valSet   = toStructArray(split, grades, 'val', 'Messidor-2', pids);
    testSet  = toStructArray(split, grades, 'test', 'Messidor-2', pids);
end

fprintf('Final split: %d train, %d val, %d test.\n', numel(trainSet), numel(valSet), numel(testSet));
fprintf('VALIDATION STORY: %s\n', validationStory);

% --- Stage-3 gates: grade range, disjointness, class report, manifest ---
localAssertGrades(trainSet, 'train');
localAssertGrades(valSet, 'val');
localAssertGrades(testSet, 'test');
localAssertDisjoint(trainSet, valSet, testSet);
localReportClassDist('train', [trainSet.icdrGrade]);
localReportClassDist('val', [valSet.icdrGrade]);
localReportClassDist('test', [testSet.icdrGrade]);
manifest = localBuildManifest(trainSet, valSet, testSet, validationStory, registry);
if ~isempty(strfind(validationStory, 'DDR'))
    localWarnDDRSegOverlap(trainSet, valSet, registry);
end
fprintf(['RULE: never tune (thresholds, temperature, hyperparameters, early stopping) against the test set above. ' ...
         'Tune on val only; report on test once.\n']);
end

% ------------------------------------------------------------------
function out = toStructArray(split, grades, whichSplit, sourceName, patientIds)
if nargin < 4 || isempty(sourceName), sourceName = ''; end
if nargin < 5, patientIds = {}; end
mask = strcmp({split.split}, whichSplit);
idx = find(mask);
out = struct('imagePath', {}, 'icdrGrade', {}, 'patientId', {}, 'source', {});
for k = 1:numel(idx)
    j = idx(k);
    pid = '';
    if ~isempty(patientIds) && numel(patientIds) >= j
        % patientIds aligned to the pre-split path order when supplied as
        % a full-length cell; otherwise a single shared default.
        if numel(patientIds) == numel(split)
            pid = char(string(patientIds{j}));
        elseif numel(patientIds) == 1
            pid = char(string(patientIds{1}));
        end
    end
    out(k) = struct('imagePath', split(j).path, 'icdrGrade', grades(j), ...
        'patientId', pid, 'source', char(sourceName));
end
end

% ------------------------------------------------------------------
function localAssertGrades(set_, name)
if isempty(set_), return; end
grades = [set_.icdrGrade];
assert(all(grades == floor(grades)) && all(grades >= 0) && all(grades <= 4), ...
    'buildGradingDatasets:badGrade - %s split has out-of-range ICDR grades (must be integers 0..4). Got range [%g %g].', ...
    name, min(grades), max(grades));
end

% ------------------------------------------------------------------
function localAssertDisjoint(trainSet, valSet, testSet)
tr = {trainSet.imagePath}; va = {valSet.imagePath}; te = {testSet.imagePath};
assert(isempty(intersect(tr, va)), 'buildGradingDatasets:leakage - %d file(s) in BOTH train and val.', numel(intersect(tr, va)));
assert(isempty(intersect(tr, te)), 'buildGradingDatasets:leakage - %d file(s) in BOTH train and test. Test must stay held out.', numel(intersect(tr, te)));
assert(isempty(intersect(va, te)), 'buildGradingDatasets:leakage - %d file(s) in BOTH val and test.', numel(intersect(va, te)));
% Exact-duplicate tripwire within a split (same path twice = double counting).
for k = 1:3
    sets = {tr, va, te}; names = {'train','val','test'};
    if numel(unique(sets{k})) ~= numel(sets{k})
        error('buildGradingDatasets:duplicate - %s split contains duplicate image paths.', names{k});
    end
end
end

% ------------------------------------------------------------------
function localReportClassDist(name, grades)
if isempty(grades)
    fprintf('Class dist [%s]: empty split.\n', name);
    return;
end
counts = histcounts(grades, -0.5:4.5);
nz = counts(counts > 0);
imb = max(counts) / max(min(nz), 1);
fprintf('Class dist [%s] n=%d: L0=%d L1=%d L2=%d L3=%d L4=%d (max/min=%.1f).\n', ...
    name, numel(grades), counts(1), counts(2), counts(3), counts(4), counts(5), imb);
if any(counts == 0)
    fprintf('  NOTE [%s]: %d ICDR level(s) have zero samples - per-class recall for those levels will be NaN, not 0. See evaluateGrading.m.\n', ...
        name, sum(counts == 0));
end
end

% ------------------------------------------------------------------
function manifest = localBuildManifest(trainSet, valSet, testSet, validationStory, registry)
snap = struct();
for i = 1:numel(registry)
    snap.(matlab.lang.makeValidName(registry(i).name)) = registry(i).available;
end
dotAt = find(validationStory == '.', 1, 'first');
if isempty(dotAt), dotAt = min(numel(validationStory) + 1, 80); end
manifest = struct( ...
    'case', validationStory(1:dotAt-1), ...
    'validationStory', char(validationStory), ...
    'seed', 42, ...
    'nTrain', numel(trainSet), 'nVal', numel(valSet), 'nTest', numel(testSet), ...
    'trainHist', histcounts([trainSet.icdrGrade], -0.5:4.5), ...
    'valHist', histcounts([valSet.icdrGrade], -0.5:4.5), ...
    'testHist', histcounts([testSet.icdrGrade], -0.5:4.5), ...
    'datasetSnapshot', snap, ...
    'testHeldOut', true, ...
    'builtAt', datestr(now, 'yyyy-mm-ddTHH:MM:SS'));
end

% ------------------------------------------------------------------
function localWarnDDRSegOverlap(trainSet, valSet, registry)
% DDR_seg pixel masks are a subset of DDR grading images. The grader sees
% DDR images through seg-PREDICTED masks, so overlapping IDs get memorized
% (not generalized) lesion channels - an optimistic bias on exactly the
% images both stages saw. Flag it; do not silently claim full externality
% on those IDs.
try
    dirByName = containers.Map({registry.name}, {registry.localDir});
    if ~isKey(dirByName, 'DDR_seg'), return; end
    segDir = dirByName('DDR_seg');
    if ~isfolder(segDir), return; end
    segFiles = [dir(fullfile(segDir, '**', '*.png')); dir(fullfile(segDir, '*.png'))];
    if isempty(segFiles), return; end
    segStems = lower(string({segFiles.name}));
    segStems = erase(segStems, [".png", ".jpg", ".tif"]);
    trStems = lower(string(cellfun(@(p) getStem(p), {trainSet.imagePath}, 'UniformOutput', false)));
    vaStems = lower(string(cellfun(@(p) getStem(p), {valSet.imagePath}, 'UniformOutput', false)));
    nTr = sum(ismember(trStems, segStems));
    nVa = sum(ismember(vaStems, segStems));
    if nTr + nVa > 0
        warning(['buildGradingDatasets:ddrSegOverlap - %d train + %d val DDR grading images share basenames with DDR_seg ' ...
                 'pixel-annotated images, whose masks trained the Stage-2 U-Nets. The grader fusion channels on those IDs ' ...
                 'are memorized, not generalized - report this overlap alongside any cross-population claim.'], nTr, nVa);
    end
catch ME
    warning('buildGradingDatasets:overlapCheckFailed', 'DDR_seg overlap check skipped (%s).', ME.message);
end
end

% ------------------------------------------------------------------
function s = getStem(p)
[~, s, ~] = fileparts(p);
end
