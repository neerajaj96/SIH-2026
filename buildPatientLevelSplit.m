function splitAssignment = buildPatientLevelSplit(imagePaths, patientIdFcn, trainFrac, valFrac, seed)
% buildPatientLevelSplit: Assigns images to train/val/test SO THAT no
% patient (or eye) appears in more than one split - a leakage risk not
% previously covered anywhere in this project. The earlier project review
% discussed leakage at the TRAIN/VAL/TEST-set BOUNDARY level ("never use
% the held-out test set for threshold selection...") but not the more
% basic risk of the SAME eye or patient appearing on both sides of a
% split in the first place, which silently inflates every reported
% metric (QWK, sensitivity, specificity, Dice - all of them), because the
% model gets partial credit for "recognizing" an image it effectively
% already saw.
%
% WHAT I CHECKED ABOUT THE ACTUAL DATASETS THIS PROJECT USES (worth
% knowing before you write patientIdFcn for each one):
%   - APTOS 2019 (Kaggle): the public train.csv has exactly two columns,
%     id_code and diagnosis - NO patient ID or eye-laterality field is
%     published. There is nothing to group by beyond the image itself,
%     so a patient-level split literally isn't constructible from this
%     dataset's public metadata. An IMAGE-level split is the most honest
%     claim you can make here - say that, rather than imply otherwise.
%   - IDRiD Disease Grading subset: 516 images (413 train / 103 test per
%     the original 2018 IEEE-ISBI challenge split), released as an
%     image-level grading dataset - no published patient/eye identifier
%     column either. Same situation as APTOS.
%   - Messidor-2: this project already uses Messidor-2 as an EXTERNAL
%     held-out TEST set (per the design doc), which is actually the more
%     important protection here - a model that never saw ANY Messidor-2
%     image (of any patient) during training or tuning is safe from
%     within-dataset leakage regardless of Messidor-2's own internal
%     patient structure.
%   - DRIVE: only used for vessel segmentation pretraining (40 images
%     total, a 20-train/20-test split fixed by the dataset itself) - too
%     small and too standardized to second-guess.
%
% BOTTOM LINE: the leakage this function actually defends against is
% mainly WITHIN a dataset build you construct yourself (e.g. if you pool
% images across IDRiD's Segmentation / Localization / Disease Grading
% subsets, which DO share some of the same underlying images across
% subsets). Use patientIdFcn to group by whatever key you can actually
% establish. If you genuinely have no grouping key for a given dataset,
% pass an identity function (patientIdFcn = @(p) p) and this degrades
% gracefully to a plain image-level split - an explicit, honest choice
% instead of an accidental one.
%
% INPUTS:
%   imagePaths   - cell array of image file paths (or any list of IDs)
%   patientIdFcn - function handle, imagePaths{i} -> a patient/eye group
%                  ID (string or number). Two images with the SAME
%                  returned ID are always kept in the SAME split.
%   trainFrac    - fraction of PATIENTS (not images) for training
%                  (default 0.70)
%   valFrac      - fraction of PATIENTS for validation (default 0.15;
%                  the remainder, 1-trainFrac-valFrac, goes to test)
%   seed         - RNG seed for reproducibility (default 42)
%   labels       - (optional) per-image class vector for label-stratified
%                  allocation. When omitted/empty the split is UNSTRATIFIED
%                  (documented limitation: rare classes can starve VAL/TEST;
%                  check the printed class balance before citing the split).
%                  Uses a private RandStream (never the global rng state).
%
% OUTPUT:
%   splitAssignment - struct array, one entry per imagePaths{i}, with
%                      fields .path and .split ('train'/'val'/'test')
%
% Requires: nothing beyond base MATLAB.

if nargin < 3 || isempty(trainFrac), trainFrac = 0.70; end
if nargin < 4 || isempty(valFrac),   valFrac = 0.15;  end
if nargin < 5 || isempty(seed),      seed = 42;        end
if nargin < 6, labels = []; end
if trainFrac + valFrac >= 1
    error('buildPatientLevelSplit:badFractions', ...
        'trainFrac (%.2f) + valFrac (%.2f) must be < 1 so a nonzero test fraction remains.', trainFrac, valFrac);
end

n = numel(imagePaths);
patientIds = cell(n, 1);
for i = 1:n
    patientIds{i} = char(string(patientIdFcn(imagePaths{i})));
end

uniquePatients = unique(patientIds, 'stable');
numPatients = numel(uniquePatients);

% Private stream: reproducible without mutating the caller's global rng.
rs = RandStream('mt19937ar', 'Seed', seed);
order = randperm(rs, numPatients);
nTrain = min(round(trainFrac * numPatients), numPatients);
nVal   = min(round(valFrac * numPatients), numPatients - nTrain);

trainPatients = uniquePatients(order(1:nTrain));
valPatients   = uniquePatients(order(nTrain+1 : nTrain+nVal));
testPatients  = uniquePatients(order(nTrain+nVal+1 : end));
if ~isempty(labels)
    % Label-stratified allocation: split each class's patients
    % proportionally so rare grades reach VAL/TEST when present.
    assert(numel(labels) == n, 'buildPatientLevelSplit:badLabels - labels must match imagePaths.');
    trainPatients = {}; valPatients = {}; testPatients = {};
    trainPatients = {}; valPatients = {}; testPatients = {};
    cls = unique(labels(:))';
    for ci = 1:numel(cls)
        % Majority-label patients for this class (patient -> mode of its images).
        pInClass = {};
        for pi = 1:numPatients
            mem = find(strcmp(patientIds, uniquePatients{pi}));
            if mode(labels(mem)) == cls(ci), pInClass{end+1} = uniquePatients{pi}; end %#ok<AGROW>
        end
        pInClass = pInClass(randperm(rs, numel(pInClass)));
        nT = round(trainFrac * numel(pInClass));
        nV = round(valFrac * numel(pInClass));
        nT = min(nT, numel(pInClass)); nV = min(nV, numel(pInClass) - nT);
        trainPatients = [trainPatients, pInClass(1:nT)]; %#ok<AGROW>
        valPatients = [valPatients, pInClass(nT+1:nT+nV)]; %#ok<AGROW>
        testPatients = [testPatients, pInClass(nT+nV+1:end)]; %#ok<AGROW>
    end
end

patientToSplit = containers.Map(uniquePatients, repmat({''}, numPatients, 1));
for i = 1:numel(trainPatients), patientToSplit(trainPatients{i}) = 'train'; end
for i = 1:numel(valPatients),   patientToSplit(valPatients{i})   = 'val';   end
for i = 1:numel(testPatients),  patientToSplit(testPatients{i})  = 'test';  end

splitAssignment = repmat(struct('path','', 'split',''), n, 1);
for i = 1:n
    splitAssignment(i).path = imagePaths{i};
    splitAssignment(i).split = patientToSplit(patientIds{i});
end

actualTrain = sum(strcmp({splitAssignment.split}, 'train'));
actualVal   = sum(strcmp({splitAssignment.split}, 'val'));
actualTest  = sum(strcmp({splitAssignment.split}, 'test'));
fprintf('Patient-level split: %d unique group(s) -> %d train / %d val / %d test patients\n', ...
    numPatients, numel(trainPatients), numel(valPatients), numel(testPatients));
fprintf('Resulting image counts: %d train / %d val / %d test (%d total)\n', actualTrain, actualVal, actualTest, n);
if ~isempty(labels)
    for c = unique(labels(:))'
        fprintf('  label %g: %d train / %d val / %d test images\n', c, ...
            sum(labels(strcmp({splitAssignment.split}, 'train')) == c), ...
            sum(labels(strcmp({splitAssignment.split}, 'val')) == c), ...
            sum(labels(strcmp({splitAssignment.split}, 'test')) == c));
    end
else
    fprintf(['NOTE: unstratified split (no labels supplied) - rare grades may starve VAL/TEST; ' ...
             'pass labels for stratified allocation.\n']);
end
if numPatients == n
    fprintf(['NOTE: every image had a UNIQUE group ID (numPatients == numImages), so this ran as a ' ...
             'plain IMAGE-level split, not a patient-level one - patientIdFcn had nothing to group on. ' ...
             'See this file''s header for which of your datasets that is expected for (APTOS, IDRiD) ' ...
             'and which it is not.\n']);
end
end
