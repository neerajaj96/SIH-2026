function out = hashArtifacts(kind, payload)
% hashArtifacts: Computed (not attested) content hashes for the MLOps chain.
%
%   hashArtifacts('manifest', manifestStruct) -> 'sha256:...' | 'fnv1a:...'
%   hashArtifacts('file', pathToMatOrAnyFile) -> same (bytes of the file)
%   hashArtifacts('struct', anyStructOrCell)  -> same (canonical string)
%
% Convention matches freezeCandidate.localContentHash: SHA-256 via JVM when
% available, FNV-1a fallback otherwise; the method is part of the returned
% string (a verifier can always tell hash strength). Timestamps are NEVER
% hashed (callers pass content only). Missing files error loudly - callers
% that cannot compute record ATTESTED_NOT_COMPUTED explicitly instead of
% calling this.
%
% *** UNEXECUTED against real artifacts here (no MATLAB/data); contract
% tests pin the convention synthetically. ***
%
% Requires: base MATLAB only (+ JVM when present, never required).

if strcmp(kind, 'file')
    assert(ischar(payload) || isstring(payload), 'hashArtifacts:badPath - file kind needs a path.');
    assert(isfile(char(payload)), 'hashArtifacts:missing - %s does not exist.', char(payload));
    fid = fopen(char(payload), 'r');
    assert(fid >= 0, 'hashArtifacts:unreadable - %s.', char(payload));
    raw = fread(fid, Inf, '*uint8');
    fclose(fid);
    out = localDigest(char(raw'));
    return;
elseif strcmp(kind, 'manifest')
    assert(isstruct(payload) && isfield(payload, 'rows'), 'hashArtifacts:badManifest - needs evaluationManifest output.');
    rows = payload.rows;
    parts = cell(1, numel(rows));
    for i = 1:numel(rows)
        r = rows(i);
        ex = double(logical(r.exclusion));
        parts{i} = sprintf('%s|%s|%s|%s|%d|%d|%s|%s', char(r.dataset), char(r.datasetVersion), ...
            char(r.imageId), char(r.split), double(r.labelICDR), ex, ...
            char(r.preprocessingVersion), char(r.provenance));
    end
    out = localDigest(strjoin(sort(parts), char(10)));
    return;
elseif strcmp(kind, 'struct')
    out = localDigest(localCanon(payload));
    return;
else
    error('hashArtifacts:badKind - kind must be manifest|file|struct.');
end
end

function c = localCanon(v)
if isstruct(v)
    f = sort(fieldnames(v));
    parts = cell(1, numel(f));
    for k = 1:numel(f)
        parts{k} = [f{k} '=' localCanon(v.(f{k}))];
    end
    c = ['{' strjoin(parts, ';') '}'];
elseif iscell(v)
    parts = cell(1, numel(v));
    for k = 1:numel(v)
        parts{k} = localCanon(v{k});
    end
    c = ['[' strjoin(parts, ',') ']'];
elseif isnumeric(v) || islogical(v)
    c = mat2str(v(:), 17);
elseif ischar(v) || isstring(v)
    c = char(v);
else
    c = class(v);
end
end

function h = localDigest(txt)
try
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(uint8(txt));
    raw = typecast(md.digest(), 'uint8');
    h = ['sha256:' sprintf('%02x', raw)];
catch
    x = uint64(14695981039346656037);
    for k = 1:numel(uint8(txt))
        x = bitxor(x, uint64(uint8(txt(k))));
        x = x * uint64(1099511628211);
    end
    h = sprintf('fnv1a:%016x', x);
end
end
