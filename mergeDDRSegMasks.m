function mergeDDRSegMasks(ddrSegDir)
% mergeDDRSegMasks: One-time helper that merges DDR_seg's four per-class
% mask sets (MA, HE, EX, SE) into the two binary targets
% train_UNet_Segmentation.m expects: masks_maho/ (MA|HE) and
% masks_exudate/ (EX|SE).
%
% WHY THIS FILE EXISTS: train_UNet_Segmentation.m hardcodes
% <source>/masks_maho + <source>/masks_exudate, but DDR_seg ships
% MA/HE/EX/SE as four SEPARATE classes (see datasetRegistry.m). The
% original header admitted "I can't script that merge sight-unseen".
% This scripts the merge WITHOUT assuming one fixed DDR layout: it
% searches common layout variants and merges whatever it finds.
%
% SUPPORTED INPUT LAYOUTS (searched in order, first hit per class wins):
%   <ddrSegDir>/MA/*.png, .../HE/*.png, .../EX/*.png, .../SE/*.png
%   <ddrSegDir>/ma/*.png (lowercase), etc.
%   <ddrSegDir>/*MA*.png flat (filename contains class tag)
%   <ddrSegDir>/masks/MA/*.png (nested masks/ variant)
% If none is found for a class, that class is skipped with a warning and
% the merge still proceeds for the classes that were found - but at least
% one MA/HE class (for maho) and one EX/SE class (for exudate) must be
% found or this errors.
%
% SE POLICY (documented choice): SE (soft exudates / cotton-wool spots)
% are merged into EXUDATES, not discarded, because both present as bright
% lesions and the exudate U-Net's clinical role is "bright-lesion" recall
% for the Level-2 rule (assignClinicalGrade exudatePresent). The merge is
% logged in the console AND in masks_maho/EXUDATE_MERGE_README.txt so the
% .meta.json + future readers know SE was included.
%
% USAGE: mergeDDRSegMasks('data/DDR/segmentation')
%
% Requires: Image Processing Toolbox (imread/imwrite).

if nargin < 1 || isempty(ddrSegDir), ddrSegDir = fullfile('data','DDR','segmentation'); end
assert(isfolder(ddrSegDir), 'mergeDDRSegMasks:missingDir - not found: %s', ddrSegDir);

maFiles = localFindClass(ddrSegDir, {'MA','ma','microaneurysm','Microaneurysm'});
heFiles = localFindClass(ddrSegDir, {'HE','he','hemorrhage','Hemorrhage','haemorrhage'});
exFiles = localFindClass(ddrSegDir, {'EX','ex','exudate','Exudate','hard'});
seFiles = localFindClass(ddrSegDir, {'SE','se','soft','cotton'});

fprintf('DDR merge search in %s:\n  MA: %d  HE: %d  EX: %d  SE: %d\n', ...
    ddrSegDir, numel(maFiles), numel(heFiles), numel(exFiles), numel(seFiles));

if isempty(maFiles) && isempty(heFiles)
    error(['mergeDDRSegMasks:noMaho - found neither MA nor HE masks under %s. ' ...
           'Open the extracted DDR folder, see the actual per-class layout, and extend localFindClass if needed.'], ddrSegDir);
end
if isempty(exFiles) && isempty(seFiles)
    error(['mergeDDRSegMasks:noExudate - found neither EX nor SE masks under %s. ' ...
           'Open the extracted DDR folder, see the actual per-class layout, and extend localFindClass if needed.'], ddrSegDir);
end

% Pair by basename stem across classes: DDR per-class masks share stems
% (e.g. 007_MA.png / 007_HE.png). Union of all stems is the image set.
allStems = containers.Map('KeyType','char','ValueType','any');
for f = [maFiles heFiles exFiles seFiles]
    [~, stem, ~] = fileparts(f);
    % Strip trailing _MA/_HE/_EX/_SE or -MA etc. to recover the image stem.
    base = regexprep(stem, '[_-](MA|HE|EX|SE|ma|he|ex|se)$', '');
    if ~isKey(allStems, base), allStems(base) = true; end
end
stems = keys(allStems);
fprintf('  Union of image stems: %d\n', numel(stems));

mahoDir = fullfile(ddrSegDir, 'masks_maho');
exDir   = fullfile(ddrSegDir, 'masks_exudate');
if ~isfolder(mahoDir), mkdir(mahoDir); end
if ~isfolder(exDir),   mkdir(exDir); end

% Index per-class files by image stem for fast lookup.
maByStem = localIndexByStem(maFiles); heByStem = localIndexByStem(heFiles);
exByStem = localIndexByStem(exFiles); seByStem = localIndexByStem(seFiles);

for i = 1:numel(stems)
    base = stems{i};
    maho = localOrMasks({localGet(maByStem, base), localGet(heByStem, base)});
    exud = localOrMasks({localGet(exByStem, base), localGet(seByStem, base)});
    if ~isempty(maho), imwrite(maho, fullfile(mahoDir, [base '.png'])); end
    if ~isempty(exud), imwrite(exud, fullfile(exDir,   [base '.png'])); end
end

readme = fullfile(ddrSegDir, 'MERGE_README.txt');
fid = fopen(readme, 'w');
fprintf(fid, ['DDR_seg merge performed %s\nMA=%d HE=%d EX=%d SE=%d files -> %d maho + %d exudate masks\n' ...
    'Policy: maho = MA|HE (binary OR); exudate = EX|SE (SE = cotton-wool/soft exudates included).\n'], ...
    datestr(now, 'yyyy-mm-ddTHH:MM:SS'), numel(maFiles), numel(heFiles), numel(exFiles), numel(seFiles), ...
    numel(dir(fullfile(mahoDir,'*.png'))), numel(dir(fullfile(exDir,'*.png'))));
fclose(fid);
fprintf('Wrote %d maho + %d exudate masks. See %s\n', ...
    numel(dir(fullfile(mahoDir,'*.png'))), numel(dir(fullfile(exDir,'*.png'))), readme);
end

% ------------------------------------------------------------------
function files = localFindClass(root, tags)
files = {};
% 1) <root>/<TAG>/* and <root>/masks/<TAG>/* subfolder variants
for t = 1:numel(tags)
    for sub = {fullfile(root, tags{t}), fullfile(root, 'masks', tags{t})}
        if isfolder(sub{1})
            d = [dir(fullfile(sub{1},'*.png')); dir(fullfile(sub{1},'*.tif')); dir(fullfile(sub{1},'*.jpg'))];
            for i = 1:numel(d), files{end+1} = fullfile(d(i).folder, d(i).name); %#ok<AGROW>
            end
        end
    end
end
if ~isempty(files), files = unique(files); return; end
% 2) flat: filename contains _TAG (e.g. 007_MA.png)
allImg = [dir(fullfile(root,'*.png')); dir(fullfile(root,'*.tif')); dir(fullfile(root,'*.jpg'))];
for i = 1:numel(allImg)
    for t = 1:numel(tags)
        if contains(allImg(i).name, ['_' tags{t}], 'IgnoreCase', false) || contains(allImg(i).name, ['-' tags{t}])
            files{end+1} = fullfile(allImg(i).folder, allImg(i).name); %#ok<AGROW>
            break;
        end
    end
end
files = unique(files);
end

% ------------------------------------------------------------------
function m = localIndexByStem(files)
m = containers.Map('KeyType','char','ValueType','char');
for i = 1:numel(files)
    [~, stem, ~] = fileparts(files{i});
    base = regexprep(stem, '[_-](MA|HE|EX|SE|ma|he|ex|se)$', '');
    if ~isKey(m, base), m(base) = files{i}; end
end
end

% ------------------------------------------------------------------
function f = localGet(mapObj, key)
if isKey(mapObj, key), f = mapObj(key); else, f = ''; end
end

% ------------------------------------------------------------------
function out = localOrMasks(fileList)
out = [];
for i = 1:numel(fileList)
    f = fileList{i};
    if isempty(f) || ~isfile(f), continue; end
    m = imread(f);
    if ndims(m) == 3, m = m(:,:,1); end
    m = double(m);
    % Midpoint threshold handles both 0/1 (thr 0.5) and 0/255 (thr 127.5)
    % binary masks. Flat masks (all one value) fall back to nonzero test.
    if max(m(:)) == min(m(:))
        b = logical(m ~= 0);
    else
        b = m > (max(m(:)) + min(m(:))) / 2;
    end
    if isempty(out), out = b;
    else
        if ~isequal(size(out), size(b))
            b = imresize(b, size(out), 'nearest'); % mismatched per-class sizes: nearest, categorical
        end
        out = out | b;
    end
end
if ~isempty(out), out = uint8(out) * 255; end
end
        out = out | m;
    end
end
if ~isempty(out), out = uint8(out) * 255; end
end
