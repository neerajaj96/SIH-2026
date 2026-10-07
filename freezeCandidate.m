function candidate = freezeCandidate(phase, inputs)
% freezeCandidate: Two-phase candidate freeze with content-hash identity.
%
% LIFECYCLE (enforced order):
%   TEST_SPLIT_FROZEN (manifest test hash recorded FIRST, before any
%     training/selection iteration) -> training/development ->
%   PRE_TRAIN_CANDIDATE -> training -> FINALIZED_CANDIDATE (Stage-14-ready)
%
% IDENTITY RULES (anti-tamper + anti-drift):
%  - candidateId/contentHash derive ONLY from immutable content inputs
%    (manifest/split hashes, config versions, code identity). Timestamps
%    (createdAt/finalizedAt) are provenance metadata and NEVER identity
%    material: identical inputs at different times yield identical IDs.
%  - gitCommitSha recorded when .git metadata is available (best effort,
%    marked 'unavailable' otherwise - never fabricated).
%  - FINALIZED requires exact checkpoint + VAL calibration artifact with
%    matching model/split/class-order provenance; mismatch INVALIDATES
%    (never silently attaches a fresh artifact to a frozen candidate).
%  - Any test-cohort change => new evaluation version (new freeze), never
%    a rewrite of the old record.
%
% *** UNEXECUTED against real training here (no MATLAB/data). Contract
% tests pin the state machine + hashing rules synthetically. ***
%
% INPUTS:
%   phase - 'PRE_TRAIN' | 'FINALIZE'
%   inputs - struct with phase-appropriate fields:
%     PRE_TRAIN: manifestHash, trainSplitHash, valSplitHash,
%       testSplitHash, testManifestVersion, configVersions (struct),
%       codeIdentity (struct with .gitCommitSha + .contentHash, optional -
%       derived via localCodeIdentity() when absent), preprocessingVersion,
%       OPTIONAL 'acceptVerdict' (acceptDataset output): overallStatus
%       REJECTED blocks PRE_TRAIN unless inputs.overrideRejected==true
%       (explicit, auditable override - never silent).
%     FINALIZE: all PRE_TRAIN fields (must match a recorded PRE_TRAIN
%       candidate - passed as .preTrainCandidate) + checkpointId,
%       checkpointHash, validationSplitId, calibrationArtifactId,
%       calibrationArtifactHash, calibrationProvenance, modelConfig,
%       trainingProvenance (seed, matlab/toolboxes/HW recorded, unknown
%       marked explicitly)
%
%   HASH PROVENANCE: hash inputs accepted as strings are recorded
%     ATTESTED_NOT_COMPUTED (caller-supplied, not verified by this
%     function). Prefer hashArtifacts('manifest'|'file') outputs, which
%     carry a sha256:/fnv1a: method prefix. The candidate records
%     hashMethod per contentHash so verifiers know the strength.
%
%   STAGE MAP (stored candidateState is the contract; stage is the
%     human-readable MLOps step):
%     DATA_VALIDATED (acceptDataset ACCEPTED_WITH_NOTES) ->
%     TEST_FROZEN (testSplitHash recorded first) ->
%     TRAINING_READY (PRE_TRAIN_CANDIDATE) -> CHECKPOINT_CREATED ->
%     CALIBRATED_ON_VALIDATION -> FINALIZED (FINALIZED_CANDIDATE) ->
%     EVALUATION_READY (Stage-14 held-out run; recorded downstream, not here).
%
% OUTPUT: candidate struct(candidateId, candidateState, gitCommitSha,
%   contentHash, manifestHash, trainSplitHash, valSplitHash,
%   testSplitHash, testManifestVersion, preprocessingVersion,
%   modelConfigVersion, checkpoint/checkpointHash, calibration*/...,
%   matlab/toolboxes/hardware/seed, createdAt, finalizedAt, limitations).

if nargin < 2, inputs = struct(); end
if strcmp(phase, 'PRE_TRAIN')
    for f = {'manifestHash','trainSplitHash','valSplitHash','testSplitHash','testManifestVersion'}
        assert(isfield(inputs, f{1}) && ~isempty(inputs.(f{1})), ...
            'freezeCandidate:missingInput - PRE_TRAIN requires %s (freeze TEST first).', f{1});
    end
    % REJECTED datasets never freeze silently: an explicit auditable
    % override is required (and recorded in limitations).
    rejectedOverride = false;
    if isfield(inputs,'acceptVerdict') && isstruct(inputs.acceptVerdict) && ...
            isfield(inputs.acceptVerdict,'overallStatus') && ...
            strcmp(inputs.acceptVerdict.overallStatus, 'REJECTED')
        if ~(isfield(inputs,'overrideRejected') && isequal(inputs.overrideRejected, true))
            error(['freezeCandidate:rejectedDataset - acceptDataset verdict is REJECTED; ' ...
                'refusing PRE_TRAIN. Pass inputs.overrideRejected=true to override explicitly (audited).']);
        end
        rejectedOverride = true;
    end
    if ~isfield(inputs,'codeIdentity') || isempty(inputs.codeIdentity)
        inputs.codeIdentity = localCodeIdentity();
    end
    if ~isfield(inputs,'configVersions'), inputs.configVersions = struct(); end
    if ~isfield(inputs,'preprocessingVersion'), inputs.preprocessingVersion = ''; end
    idMaterial = {'PRE_TRAIN', inputs.manifestHash, inputs.trainSplitHash, inputs.valSplitHash, ...
        inputs.testSplitHash, inputs.testManifestVersion, localCanon(inputs.configVersions), ...
        inputs.codeIdentity.gitCommitSha, inputs.codeIdentity.contentHash, inputs.preprocessingVersion};
    [candidateId, contentHash] = localContentHash(idMaterial);
    hashMethod = localHashMethod(contentHash);
    lims = {'PRE_TRAIN only - not Stage-14-ready until FINALIZED_CANDIDATE exists', ...
        'caller-supplied hashes recorded ATTESTED_NOT_COMPUTED - recompute via hashArtifacts for computed lineage'};
    if rejectedOverride
        lims{end+1} = 'OVERRIDE: frozen despite REJECTED acceptDataset verdict (explicit overrideRejected=true).';
    end
    candidate = struct('candidateId', candidateId, 'candidateState', 'PRE_TRAIN_CANDIDATE', ...
        'stage', 'TRAINING_READY', ...
        'gitCommitSha', inputs.codeIdentity.gitCommitSha, 'contentHash', contentHash, ...
        'hashMethod', hashMethod, ...
        'manifestHash', inputs.manifestHash, 'trainSplitHash', inputs.trainSplitHash, ...
        'valSplitHash', inputs.valSplitHash, 'testSplitHash', inputs.testSplitHash, ...
        'testManifestVersion', inputs.testManifestVersion, ...
        'preprocessingVersion', inputs.preprocessingVersion, ...
        'modelConfigVersion', localCanon(inputs.configVersions), ...
        'checkpointId', '', 'checkpointHash', '', ...
        'calibrationArtifactId', '', 'calibrationArtifactHash', '', ...
        'calibrationDataset', '', 'calibrationSplit', '', ...
        'matlabVersion', '', 'toolboxes', {{}}, 'hardware', '', 'seed', NaN, ...
        'createdAt', datestr(now, 30), 'finalizedAt', '', ...
        'parentCandidateId', '', ...
        'limitations', {lims});
elseif strcmp(phase, 'FINALIZE')
    assert(isfield(inputs,'preTrainCandidate'), ...
        'freezeCandidate:missingInput - FINALIZE requires the PRE_TRAIN candidate record.');
    pre = inputs.preTrainCandidate;
    assert(strcmp(pre.candidateState, 'PRE_TRAIN_CANDIDATE'), ...
        'freezeCandidate:badState - can only finalize a PRE_TRAIN_CANDIDATE.');
    for f = {'checkpointId','checkpointHash','validationSplitId','calibrationArtifactId', ...
             'calibrationArtifactHash','calibrationProvenance','modelConfig','trainingProvenance'}
        assert(isfield(inputs, f{1}) && ~isempty(inputs.(f{1})), ...
            'freezeCandidate:missingInput - FINALIZE requires %s.', f{1});
    end
    % Calibration provenance MUST match checkpoint + VAL split + classes.
    cp = inputs.calibrationProvenance;
    if ~isfield(cp,'checkpointId') || ~strcmp(cp.checkpointId, inputs.checkpointId)
        error('freezeCandidate:calibrationMismatch - calibration artifact checkpoint does not match candidate checkpoint.');
    end
    if isfield(cp,'splitId') && ~isempty(cp.splitId) && ~strcmp(cp.splitId, inputs.validationSplitId)
        error('freezeCandidate:calibrationMismatch - calibration fitted on %s, validation split is %s.', cp.splitId, inputs.validationSplitId);
    end
    if isfield(cp,'classOrdering') && (~isequal(cp.classOrdering(:), (0:4)'))
        error('freezeCandidate:calibrationMismatch - calibration class ordering is not ICDR 0-4.');
    end
    idMaterial = {'FINALIZE', pre.candidateId, inputs.checkpointId, inputs.checkpointHash, ...
        inputs.validationSplitId, inputs.calibrationArtifactId, inputs.calibrationArtifactHash, ...
        localCanon(inputs.modelConfig)};
    [candidateId, contentHash] = localContentHash(idMaterial);
    candidate = pre;
    candidate.candidateId = candidateId;
    candidate.candidateState = 'FINALIZED_CANDIDATE';
    candidate.stage = 'FINALIZED';
    candidate.contentHash = contentHash;
    candidate.hashMethod = localHashMethod(contentHash);
    candidate.parentCandidateId = pre.candidateId;
    candidate.checkpointId = inputs.checkpointId;
    candidate.checkpointHash = inputs.checkpointHash;
    candidate.calibrationArtifactId = inputs.calibrationArtifactId;
    candidate.calibrationArtifactHash = inputs.calibrationArtifactHash;
    candidate.calibrationDataset = localField(inputs.calibrationProvenance, 'datasetTag', 'unspecified');
    candidate.calibrationSplit = inputs.validationSplitId;
    candidate.modelConfigVersion = localCanon(inputs.modelConfig);
    tp = inputs.trainingProvenance;
    candidate.matlabVersion = localField(tp, 'matlabVersion', 'unknown');
    candidate.toolboxes = localField(tp, 'toolboxes', {});
    candidate.hardware = localField(tp, 'hardware', 'unknown');
    candidate.seed = localField(tp, 'seed', NaN);
    candidate.finalizedAt = datestr(now, 30);
    candidate.limitations = {'Stage-14-ready: evaluate frozen TEST cohort only; never tune on it.', ...
        'caller-supplied checkpoint/calibration hashes ATTESTED_NOT_COMPUTED - recompute via hashArtifacts for computed lineage', ...
        'EVALUATION_READY is recorded by the Stage-14 held-out run, not by this freeze.'};
else
    error('freezeCandidate:badPhase - phase must be PRE_TRAIN or FINALIZE.');
end
end

function s = localField(st, f, dflt)
if isfield(st, f) && ~isempty(st.(f))
    s = st.(f);
else
    s = dflt;
end
end

function c = localCanon(v)
% Canonical string for structs/cells/strings/numbers (field-order stable).
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

function [id, h] = localContentHash(material)
% Content hash with NO timestamps: identical inputs at different times
% yield identical IDs (verified by contract test). SHA-256 via JVM when
% available, FNV-1a fallback otherwise (method recorded in the hash).
txt = localCanon(material);
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
id = ['cand_' h(1:min(16, numel(h)))];
end

function m = localHashMethod(h)
% The contentHash string carries its method (sha256:/fnv1a:); surface it
% so verifiers know the strength without parsing.
if strncmp(h, 'sha256:', 7), m = 'SHA-256 (JVM)';
elseif strncmp(h, 'fnv1a:', 6), m = 'FNV-1a fallback (no JVM)';
else, m = 'unknown';
end
end

function ci = localCodeIdentity()
% Best-effort code identity: git SHA when .git metadata readable,
% else 'unavailable' (never fabricated); content hash always computed
% over the working .m set via saveModelWithMetadata's hasher when
% present, else the literal string 'unavailable'.
ci = struct('gitCommitSha', 'unavailable', 'contentHash', 'unavailable');
try
    [st, out] = system('git rev-parse HEAD');
    if st == 0
        ci.gitCommitSha = strtrim(out);
    end
catch
end
try
    ci.contentHash = computeProjectCodeHash();
catch
end
end
