function testSet = buildMessidorTestSet(imagesDir, gradesCsvPath, pairsCsvPath)
% buildMessidorTestSet: Turns "a folder of Messidor-2 images" into the
% labeled held-out test set compareModels.m and calibrateTemperature.m
% actually need.
%
% WHY THIS FILE EXISTS: the Messidor-2 distribution linked from the
% official problem statement (adcis.net) ships WITHOUT diagnostic grade
% labels - it was released that way deliberately, and every serious paper
% using Messidor-2 sources grades from a separate adjudication effort
% (Krause et al. 2018 - 3 retinal specialists reaching consensus per
% image), not from the base download. If your messidor-2.csv has only
% "left;right" columns (image pairing by patient/session), that CSV is
% NOT diagnostic labels - it's just which two images came from the same
% exam. Get the actual grades separately:
%   https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades
% (This is the same Krause et al. adjudication cited by essentially every
% paper that reports numbers on Messidor-2 - it's not a shortcut, it's
% the standard label source for this exact dataset.)
%
% DO NOT use this test set for anything except final reporting: no
% threshold selection, no temperature fitting, no hyperparameter tuning.
% The whole point of holding it out is that nothing about the pipeline
% was chosen by looking at it. calibrateTemperature.m should be run on a
% SEPARATE validation split from APTOS/IDRiD, never on Messidor-2.
%
% INPUTS:
%   imagesDir     - folder containing the extracted Messidor-2 .png/.jpg files
%                   (after merging however many zip parts you downloaded
%                   into one folder - see the note below)
%   gradesCsvPath - path to the downloaded grades CSV (adjust the column
%                   names below once you've actually opened your copy -
%                   Kaggle dataset exports do vary, and I can't fetch this
%                   one myself to confirm the exact header names)
%   pairsCsvPath  - (optional) your messidor-2.csv (left;right columns) -
%                   used only to assign a patientId so a downstream
%                   patient-level grouping tool (buildPatientLevelSplit.m)
%                   has something to key on if you ever need it. Not
%                   required for compareModels.m, which just needs image+grade.
%
% OUTPUT:
%   testSet - struct array, one entry per matched image, with fields:
%     .imagePath  - full path to the image file
%     .icdrGrade  - 0-4 (double)
%     .gradable   - logical (Messidor-2's adjudication marked 4 of 1748
%                   images ungradable - those are EXCLUDED automatically,
%                   not silently kept with a fabricated grade)
%     .patientId  - string (only populated if pairsCsvPath given)
%
% ON THE 4-PART ZIP: dataset hosts split large downloads two different
% ways. If unzipping part 1 alone just works and gives you a quarter of
% the images, they're independent zips - unzip all 4 into the SAME
% destination folder and you're done. If part 1 alone errors about a
% missing "next volume" / "part 2", they're a true split archive - you
% need every part present before unzipping, e.g. via 7-Zip, or on
% Linux/macOS: `zip -s 0 part1.zip --out combined.zip && unzip combined.zip`.
%
% Requires: nothing beyond base MATLAB (readtable).

if nargin < 3
    pairsCsvPath = '';
end

% --- Adjust these to match your actual downloaded grades CSV's headers ---
IMAGE_COL = "image_id";      % or "id_code" / "filename" - check your CSV
GRADE_COL = "adjudicated_dr_grade"; % or "diagnosis" / "dr_grade" - check your CSV
GRADABLE_COL = "adjudicated_gradable"; % set to '' below if your CSV has no such column

grades = readtable(gradesCsvPath, 'TextType', 'string');
if ~ismember(IMAGE_COL, string(grades.Properties.VariableNames))
    error(['buildMessidorTestSet:columnNotFound - "%s" is not a column in %s. ' ...
           'Open the CSV, check the real header names, and edit IMAGE_COL/GRADE_COL/GRADABLE_COL ' ...
           'at the top of this function to match. Found columns: %s'], ...
        IMAGE_COL, gradesCsvPath, strjoin(string(grades.Properties.VariableNames), ', '));
end

patientOf = containers.Map('KeyType','char','ValueType','char');
if ~isempty(pairsCsvPath) && isfile(pairsCsvPath)
    pairs = readtable(pairsCsvPath, 'Delimiter', ';', 'TextType', 'string');
    for i = 1:height(pairs)
        pid = sprintf('patient_%04d', i);
        patientOf(char(pairs.left(i)))  = pid;
        patientOf(char(pairs.right(i))) = pid;
    end
end

files = [dir(fullfile(imagesDir,'*.png')); dir(fullfile(imagesDir,'*.jpg')); dir(fullfile(imagesDir,'*.tif'))];
if isempty(files)
    error('buildMessidorTestSet:noImages', 'No .png/.jpg/.tif files found in %s - check the path and that the zip parts were actually extracted.', imagesDir);
end

testSet = struct('imagePath', {}, 'icdrGrade', {}, 'gradable', {}, 'patientId', {});
nMatched = 0; nSkippedUngradable = 0; nUnmatched = 0;
for i = 1:numel(files)
    fname = files(i).name;
    [~, stem, ~] = fileparts(fname);
    row = grades(strcmp(string(grades.(IMAGE_COL)), stem) | strcmp(string(grades.(IMAGE_COL)), string(fname)), :);
    if isempty(row)
        nUnmatched = nUnmatched + 1;
        continue;
    end
    isGradable = true;
    if ~isempty(GRADABLE_COL) && ismember(GRADABLE_COL, string(grades.Properties.VariableNames))
        isGradable = logical(row.(GRADABLE_COL)(1));
    end
    if ~isGradable
        nSkippedUngradable = nSkippedUngradable + 1;
        continue;
    end
    nMatched = nMatched + 1;
    pid = "";
    if isKey(patientOf, fname)
        pid = string(patientOf(fname));
    end
    testSet(nMatched) = struct('imagePath', fullfile(imagesDir, fname), ...
        'icdrGrade', double(row.(GRADE_COL)(1)), 'gradable', true, 'patientId', pid); %#ok<AGROW>
end

fprintf('Matched %d images to grades, skipped %d marked ungradable, %d image files had no matching grade row.\n', ...
    nMatched, nSkippedUngradable, nUnmatched);
if nMatched == 0
    error('buildMessidorTestSet:noMatches', ['Zero images matched - almost certainly a filename-format ' ...
        'mismatch between the image files and the grades CSV''s %s column. Print a few values from ' ...
        'each side and compare by hand before re-running.'], IMAGE_COL);
end
counts = histcounts([testSet.icdrGrade], -0.5:4.5);
fprintf('ICDR grade distribution in matched test set: Level 0=%d, 1=%d, 2=%d, 3=%d, 4=%d\n', counts);
end
