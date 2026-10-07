function p = uniqueArtifactPath(outDir, stem, ext)
% uniqueArtifactPath: Collision-proof artifact filename for reports,
% CSVs and supporting images. Second-granularity timestamps alone let
% concurrent or repeated runs overwrite each other, so this appends a
% deterministic collision check: stem_timestamp, then _v2, _v3... until
% the path does not exist. Readability preserved (stem + timestamp stay
% the leading components); uniqueness does not rely on milliseconds.
%
% INPUTS: outDir (created if missing), stem (no timestamp needed -
%   added here), ext ('.csv', '.png', '.pdf', '.txt', with or w/o dot)
% OUTPUT: full path guaranteed not to exist at call time.
%
% Millisecond timestamp + _vN collision loop: two artifacts in the same
% second (batch + report, concurrent runs) no longer share a base name.
% No randomness is introduced (filenames stay sortable/deterministic).
%
% Requires: base MATLAB only.

if nargin < 3, ext = ''; end
if ~isempty(ext) && ext(1) ~= '.', ext = ['.' ext]; end
if ~isfolder(outDir), mkdir(outDir); end
try
    stamp = datestr(now, 'yyyymmdd_HHMMSS_FFF');
catch
    stamp = datestr(now, 'yyyymmdd_HHMMSS');
end
base = sprintf('%s_%s', stem, stamp);
p = fullfile(outDir, [base ext]);
k = 2;
while exist(p, 'file')
    p = fullfile(outDir, sprintf('%s_v%d%s', base, k, ext));
    k = k + 1;
end
end
