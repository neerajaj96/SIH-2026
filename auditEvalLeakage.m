function audit = auditEvalLeakage(manifest, varargin)
% auditEvalLeakage: Leakage/role audit over an evaluationManifest.
% Returns explicit statuses + reasons per check (never bare booleans).
%
% CHECKS:
%  1. split roles present (TRAIN+VAL+TEST non-empty; warns otherwise)
%  2. image disjointness across splits (by imageId)
%  3. patient disjointness (only when patientIdsAvailable; else the
%     check records NOT_APPLICABLE with the limitation explicit)
%  4. EXACT duplicates (same imageId twice, or same file hash when
%     imageHash field present) -> EXACT_DUPLICATE
%  5. POSSIBLE duplicates (same basename stem across different imageIds)
%     -> POSSIBLE_DUPLICATE (collision to review, never auto-confirmed)
%  6. NEAR duplicates: NO content-similarity method exists in this repo,
%     so this check records UNSUPPORTED (not "clean") - do not claim a
%     near-duplicate screen that was never implemented.
%  7. calibration/threshold split != TEST (by calibrationArtifactId /
%     thresholdSource fields when present; else UNVERIFIABLE warning)
%  8. preprocessing version consistency across splits (mismatch flagged)
%  9. GT-separation attestation inputs (inferenceUsedGT logical, default
%     false must be explicitly passed true/false by the caller - absent
%     attestation => UNVERIFIABLE, never assumed clean)
%  10. DDR_seg subset overlap: flags when datasetSource mixes DDR
%      segmentation and DDR grading rows (memorization risk, warn only)
%
% INPUT: manifest (evaluationManifest output), 'InferenceUsedGT' (default
%   [] = unattested), 'CalibrationSplit','ThresholdSplit' (default '').
% OUTPUT: audit struct(checks(struct array name/status/reason),
%   overall ('CLEAN'|'FINDINGS'|'UNVERIFIABLE'), configVersion).

p = inputParser;
addParameter(p, 'InferenceUsedGT', [], @(x) isempty(x) || islogical(x));
addParameter(p, 'CalibrationSplit', '', @ischar);
addParameter(p, 'ThresholdSplit', '', @ischar);
parse(p, varargin{:});

rows = manifest.rows;
checks = struct('name', {}, 'status', {}, 'reason', {});
add = @(n,s,r) struct('name', n, 'status', s, 'reason', r);

splits = {rows.split};
for s = {'TRAIN','VAL','TEST'}
    if ~any(strcmp(splits, s{1}))
        checks(end+1) = add(['split-present-' s{1}], 'FINDINGS', 'split role empty in manifest'); %#ok<AGROW>
    else
        checks(end+1) = add(['split-present-' s{1}], 'CLEAN', 'role populated'); %#ok<AGROW>
    end
end

% Image disjointness.
ids = {rows.imageId};
ok = true; detail = '';
for a = {'TRAIN','VAL','TEST'}
    for b = {'TRAIN','VAL','TEST'}
        if strcmp(a{1}, b{1}), continue; end
        ia = ids(strcmp(splits, a{1})); ib = ids(strcmp(splits, b{1}));
        ov = intersect(ia, ib);
        if ~isempty(ov)
            ok = false; detail = sprintf('%s<->%s share %d ids', a{1}, b{1}, numel(ov));
        end
    end
end
if ok
    checks(end+1) = add('image-disjointness', 'CLEAN', 'no imageId crosses splits'); %#ok<AGROW>
else
    checks(end+1) = add('image-disjointness', 'FINDINGS', detail); %#ok<AGROW>
end

% Patient disjointness (only when IDs exist).
if manifest.patientIdsAvailable
    pids = {rows.patientId};
    ok = true; detail = 'no patient crosses splits';
    for a = {'TRAIN','VAL','TEST'}
        for b = {'TRAIN','VAL','TEST'}
            if strcmp(a{1}, b{1}), continue; end
            pa = pids(strcmp(splits, a{1})); pb = pids(strcmp(splits, b{1}));
            pa = pa(~cellfun(@isempty, pa)); pb = pb(~cellfun(@isempty, pb));
            ov = intersect(unique(pa), unique(pb));
            if ~isempty(ov)
                ok = false; detail = sprintf('%s<->%s share patients', a{1}, b{1});
            end
        end
    end
    if ok
        checks(end+1) = add('patient-disjointness', 'CLEAN', detail); %#ok<AGROW>
    else
        checks(end+1) = add('patient-disjointness', 'FINDINGS', detail); %#ok<AGROW>
    end
else
    checks(end+1) = add('patient-disjointness', 'NOT_APPLICABLE', ...
        'no patient IDs - image-level split only; patient-level protection NOT claimed'); %#ok<AGROW>
end

% EXACT duplicates.
[uniqIds, ~, ic] = unique(ids);
dupExact = uniqIds(histcounts(ic, 1:max(ic)+1) > 1);
if isempty(dupExact)
    checks(end+1) = add('exact-duplicates', 'CLEAN', 'no repeated imageId'); %#ok<AGROW>
else
    checks(end+1) = add('exact-duplicates', 'FINDINGS', ...
        sprintf('EXACT_DUPLICATE imageIds: %s', strjoin(dupExact(1:min(5,numel(dupExact))), ','))); %#ok<AGROW>
end
if isfield(rows, 'imageHash')
    h = {rows.imageHash}; h = h(~cellfun(@isempty, h));
    [uh, ~, ih] = unique(h);
    dupH = uh(histcounts(ih, 1:max(ih)+1) > 1);
    if ~isempty(dupH)
        checks(end+1) = add('exact-duplicates-hash', 'FINDINGS', 'EXACT_DUPLICATE by content hash'); %#ok<AGROW>
    end
end

% POSSIBLE duplicates (stem collisions only - review, don't confirm).
stems = cellfun(@localStem, ids, 'UniformOutput', false);
[us, ~, is_] = unique(stems);
dupS = us(histcounts(is_, 1:max(is_)+1) > 1);
if isempty(dupS)
    checks(end+1) = add('possible-duplicates', 'CLEAN', 'no stem collisions'); %#ok<AGROW>
else
    checks(end+1) = add('possible-duplicates', 'FINDINGS', ...
        sprintf('POSSIBLE_DUPLICATE stems (review, not confirmed): %s', strjoin(dupS(1:min(5,numel(dupS))), ','))); %#ok<AGROW>
end
checks(end+1) = add('near-duplicates', 'UNSUPPORTED', ...
    'no content-similarity method implemented - near-duplicate screening NOT performed, not claimed clean'); %#ok<AGROW>

% Calibration / threshold separation.
for f = {'CalibrationSplit', 'ThresholdSplit'}
    v = p.Results.(f{1});
    if isempty(v)
        checks(end+1) = add(lower(f{1}), 'UNVERIFIABLE', 'split identity not supplied - separation unproven'); %#ok<AGROW>
    elseif strcmp(v, 'TEST')
        checks(end+1) = add(lower(f{1}), 'FINDINGS', 'fitted on TEST - contamination'); %#ok<AGROW>
    else
        checks(end+1) = add(lower(f{1}), 'CLEAN', sprintf('fitted on %s, metrics on TEST', v)); %#ok<AGROW>
    end
end

% Preprocessing consistency.
pre = unique({rows.preprocessingVersion});
pre = pre(~cellfun(@isempty, pre));
if numel(pre) <= 1
    checks(end+1) = add('preprocessing-identity', 'CLEAN', 'single preprocessing version'); %#ok<AGROW>
else
    checks(end+1) = add('preprocessing-identity', 'FINDINGS', 'multiple preprocessing versions across splits'); %#ok<AGROW>
end

% GT-separation attestation.
if isempty(p.Results.InferenceUsedGT)
    checks(end+1) = add('gt-separation', 'UNVERIFIABLE', 'inference GT use unattested - confirm predicted-masks-only'); %#ok<AGROW>
elseif p.Results.InferenceUsedGT
    checks(end+1) = add('gt-separation', 'FINDINGS', 'GT masks supplied to inference path'); %#ok<AGROW>
else
    checks(end+1) = add('gt-separation', 'CLEAN', 'attested predicted-masks-only'); %#ok<AGROW>
end

% DDR_seg overlap.
srcs = unique({rows.datasetSource});
if any(contains(srcs, 'DDR')) && numel(srcs) > 1
    checks(end+1) = add('ddr-overlap', 'FINDINGS', 'DDR_seg subset overlap risk - dedupe before cross-population claims'); %#ok<AGROW>
else
    checks(end+1) = add('ddr-overlap', 'CLEAN', 'no mixed DDR sources'); %#ok<AGROW>
end

stats = {checks.status};
if any(strcmp(stats, 'FINDINGS'))
    overall = 'FINDINGS';
elseif any(strcmp(stats, 'UNVERIFIABLE')) || any(strcmp(stats, 'UNSUPPORTED'))
    overall = 'UNVERIFIABLE';
else
    overall = 'CLEAN';
end
audit = struct('checks', checks, 'overall', overall, 'configVersion', '1.0.0-stage12');
end

function s = localStem(id)
[~, n, ~] = fileparts(id);
s = lower(n);
end
