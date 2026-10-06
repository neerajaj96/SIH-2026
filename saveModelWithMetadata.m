function saveModelWithMetadata(matFilePath, net, extraInfo)
% saveModelWithMetadata: Saves a trained network the same way `save()`
% always has, PLUS a metadata sidecar - because right now, retraining
% any network produces a .mat file with zero record of when it was
% trained, on what data, or with what result. There's no rollback, no
% "which version is actually deployed", no way to tell two
% trained_dr_grader.mat files apart without loading and testing both.
%
% Writes <matFilePath>.meta.json next to the .mat file, with:
%   trainedAt        - ISO 8601 timestamp
%   datasetSnapshot   - datasetRegistry() output at save time (which
%                       sources were FOUND, so you can tell a model
%                       trained with DDR from one trained without it)
%   codeHash          - a simple content hash of every .m file in the
%                       project directory at save time, NOT a real git
%                       commit hash (this project may or may not be under
%                       git) - good enough to tell "was this the exact
%                       same code" apart from "something changed",
%                       not a replacement for actual version control
%   extraInfo         - whatever the caller wants recorded (validation
%                       metrics, notes) - pass a struct, e.g. from
%                       compareModels.m's output
%
% USAGE (drop-in for a bare save() call at the end of a training script):
%   saveModelWithMetadata('trained_dr_grader.mat', net, struct('qwk', comparison.pipeline.qwk));
%
% Requires: nothing beyond base MATLAB.

if nargin < 3
    extraInfo = struct();
end

save(matFilePath, 'net');

meta = struct();
meta.trainedAt = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
meta.matFile = matFilePath;

try
    reg = datasetRegistry();
    snap = struct();
    for i = 1:numel(reg)
        snap.(reg(i).name) = reg(i).available;
    end
    meta.datasetSnapshot = snap;
catch
    meta.datasetSnapshot = struct('note', 'datasetRegistry.m not on path or failed - not recorded');
end

meta.codeHash = computeProjectCodeHash();
meta.extraInfo = extraInfo;

metaPath = strrep(matFilePath, '.mat', '.meta.json');
fid = fopen(metaPath, 'w');
if fid == -1
    warning('saveModelWithMetadata:writeFailed', 'Could not write %s - model saved, metadata was not.', metaPath);
    return;
end
fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true));
fclose(fid);
fprintf('Saved %s and %s\n', matFilePath, metaPath);
end

% ------------------------------------------------------------------
function h = computeProjectCodeHash()
% A simple, dependency-free "did the code change" fingerprint: hash the
% concatenated byte content of every .m file in the current directory,
% sorted by name for a stable order. NOT cryptographically meaningful,
% not a replacement for git - just enough to tell two model saves apart.
files = dir('*.m');
names = sort({files.name});
allBytes = uint8([]);
for i = 1:numel(names)
    fid = fopen(names{i}, 'r');
    if fid == -1, continue; end
    allBytes = [allBytes; uint8(fread(fid, Inf, 'uint8'))]; %#ok<AGROW>
    fclose(fid);
end
if isempty(allBytes)
    h = 'no-m-files-found';
    return;
end
try
    engine = java.security.MessageDigest.getInstance('SHA-256');
    engine.update(allBytes);
    digest = typecast(engine.digest(), 'uint8');
    h = lower(sprintf('%02x', digest));
catch
    h = sprintf('sum%d-len%d', sum(double(allBytes)), numel(allBytes)); % Java unavailable fallback - weaker, but still distinguishes most real changes
end
end
