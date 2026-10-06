function [trainSet, valSet, testSet, validationStory] = buildGradingDatasets()
% buildGradingDatasets: Builds the train/val/test split for DR severity
% grading, adapting to whichever datasets datasetRegistry() finds.
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
    trainSet = toStructArray(split, ddrGrades, 'train');
    valSet   = toStructArray(split, ddrGrades, 'val');

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
    trainSet = toStructArray(split, grades, 'train');
    valSet   = toStructArray(split, grades, 'val');
    testSet  = toStructArray(split, grades, 'test');
end

fprintf('Final split: %d train, %d val, %d test.\n', numel(trainSet), numel(valSet), numel(testSet));
fprintf('VALIDATION STORY: %s\n', validationStory);
end

% ------------------------------------------------------------------
function out = toStructArray(split, grades, whichSplit)
mask = strcmp({split.split}, whichSplit);
idx = find(mask);
out = struct('imagePath', {}, 'icdrGrade', {});
for k = 1:numel(idx)
    out(k) = struct('imagePath', split(idx(k)).path, 'icdrGrade', grades(idx(k)));
end
end
