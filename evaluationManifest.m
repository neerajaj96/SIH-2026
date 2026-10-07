function manifest = evaluationManifest(rows)
% evaluationManifest: Builds + validates the single evaluation manifest
% contract (one struct row per image). All Stage-12 evaluation reads
% cohort definition from here - never from ad-hoc dir scans.
%
% FIELDS (all required unless noted):
%   dataset, datasetVersion, datasetSource, imageId, patientId
%     ('' when unknown - see patientIdsAvailable), labelICDR (0-4),
%   referableLabel (derived ICDR>=2 unless separatelyAdjudicated=true),
%   separatelyAdjudicated (logical), segGT (struct with .vessel/.mahe/
%     .exudate logical presence flags; all-false = no seg GT),
%   segClass ('vessel,mahe,exudate' subset string), exclusion (logical),
%   exclusionReason ('' if included), provenance (free text, required
%   non-empty), split ('TRAIN'/'VAL'/'TEST'),
%   preprocessingVersion, checkpointId, calibrationArtifactId
%   OPTIONAL provenance detail (recorded when known, never fabricated):
%   graderCount (NaN unknown), adjudicationRule ('' unknown),
%   labelDate ('' unknown). Presence is validated; absence stays explicit.
%
% VALIDATION (errors loudly, never silently repairs):
%   - ICDR in {0..4}; split in {TRAIN,VAL,TEST}; provenance non-empty
%   - referable consistent with ICDR>=2 unless separatelyAdjudicated
%     (both definitions preserved then, never silently replaced)
%   - patientIdsAvailable=false recorded when all patientId empty, and
%     patient-level claims are DISABLED downstream (image-level split
%     permitted only with the limitation explicit)
%
% INPUT: rows - struct array (partially filled structs are completed
%   with defaults for optional fields before validation).
% OUTPUT: manifest struct(rows, patientIdsAvailable, nPatients, nImages).
%
% Requires: base MATLAB only.

req = {'dataset','datasetVersion','datasetSource','imageId','labelICDR','split','provenance'};
n = numel(rows);
assert(n > 0, 'evaluationManifest:empty - manifest requires at least one row.');
for i = 1:n
    for k = 1:numel(req)
        assert(isfield(rows(i), req{k}), ...
            'evaluationManifest:missingField - row %d lacks required field %s.', i, req{k});
    end
    if ~isfield(rows(i),'patientId') || isempty(rows(i).patientId)
        rows(i).patientId = '';
    end
    if ~isfield(rows(i),'separatelyAdjudicated') || isempty(rows(i).separatelyAdjudicated)
        rows(i).separatelyAdjudicated = false;
    end
    if ~isfield(rows(i),'referableLabel') || isempty(rows(i).referableLabel)
        rows(i).referableLabel = double(rows(i).labelICDR >= 2);
    end
    if ~isfield(rows(i),'exclusion') || isempty(rows(i).exclusion)
        rows(i).exclusion = false; rows(i).exclusionReason = '';
    end
    if ~isfield(rows(i),'segGT')
        rows(i).segGT = struct('vessel',false,'mahe',false,'exudate',false);
    end
    if ~isfield(rows(i),'segClass'), rows(i).segClass = ''; end
    if ~isfield(rows(i),'preprocessingVersion'), rows(i).preprocessingVersion = ''; end
    if ~isfield(rows(i),'checkpointId'), rows(i).checkpointId = ''; end
    if ~isfield(rows(i),'calibrationArtifactId'), rows(i).calibrationArtifactId = ''; end
    % Structured label provenance (Stage-12 protocol section 2): unknown
    % stays unknown (NaN/''), never a fabricated count or rule.
    if ~isfield(rows(i),'graderCount') || isempty(rows(i).graderCount)
        rows(i).graderCount = NaN;
    end
    if ~isfield(rows(i),'adjudicationRule') || isempty(rows(i).adjudicationRule)
        rows(i).adjudicationRule = '';
    end
    if ~isfield(rows(i),'labelDate') || isempty(rows(i).labelDate)
        rows(i).labelDate = '';
    end
    if ~isnan(rows(i).graderCount)
        assert(rows(i).graderCount >= 1 && rows(i).graderCount == floor(rows(i).graderCount), ...
            'evaluationManifest:badGraderCount - row %d graderCount must be a positive integer or NaN (unknown).', i);
    end
    assert(ismember(rows(i).labelICDR, 0:4), ...
        'evaluationManifest:badICDR - row %d ICDR=%g not in {0..4}.', i, rows(i).labelICDR);
    assert(ismember(rows(i).split, {'TRAIN','VAL','TEST'}), ...
        'evaluationManifest:badSplit - row %d split must be TRAIN/VAL/TEST.', i);
    if ~rows(i).separatelyAdjudicated
        assert(rows(i).referableLabel == double(rows(i).labelICDR >= 2), ...
            'evaluationManifest:referableDrift - row %d referable must equal ICDR>=2 unless separately adjudicated.', i);
    end
    assert(~isempty(rows(i).provenance), 'evaluationManifest:noProvenance - row %d needs provenance.', i);
end
pids = {rows.patientId};
patientIdsAvailable = any(~cellfun(@isempty, pids));
if patientIdsAvailable
    nPatients = numel(unique(pids(~cellfun(@isempty, pids))));
else
    nPatients = NaN; % unknown - never claim patient-level protection
end
manifest = struct('rows', rows, 'patientIdsAvailable', patientIdsAvailable, ...
    'nPatients', nPatients, 'nImages', n);
end
