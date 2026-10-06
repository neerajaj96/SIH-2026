function [icdrGrade, evidence, ruleReport] = assignClinicalGrade(hemorrhageMask, quadrantMask, maskInfo)
% assignClinicalGrade: Status-aware ICDR 4-2-1 evidence engine.
%
% BACKWARD COMPATIBILITY: [grade, evidence] = assignClinicalGrade(...)
% keeps its historical shape. New code should use the third output
% ruleReport (status, evidenceStatus, provenance). Grades are returned
% ONLY when evidence is sufficient; otherwise grade is NaN with status
% INSUFFICIENT_EVIDENCE (never a fabricated 0-4).
%
% EVIDENCE STATUS VOCABULARY (clinicalConfig.m, frozen strings):
%   VERIFIED / PROXY / NOT_DETECTED / UNAVAILABLE / INVALID, and rule
%   status SUFFICIENT / INSUFFICIENT_EVIDENCE / INVALID.
%
% MEDICAL-SAFETY RULES (conservative by design):
%  - VB and IRMA have NO detectors in this release: unless the caller
%    supplies an explicit VERIFIED/PROXY status (manual grading or a
%    future validated assessor), both are UNAVAILABLE - never zero.
%  - Vitreous hemorrhage is manual input only (default UNAVAILABLE).
%  - A definitive Level 0 claim requires having assessed everything;
%    with VB/IRMA/vitreous UNAVAILABLE, Level 0/1/2 cannot be asserted -
%    the engine returns INSUFFICIENT_EVIDENCE + NaN instead of 0/1/2.
%  - NV comes ONLY from the detectNeovascularization SCREENING PROXY:
%    a positive yields grade 4 with status PROXY (never "diagnosis").
%  - Hemorrhage counts derive from the MERGED MA/HE channel (no true
%    MA-vs-hemorrhage separation exists); provenance always says so.
%  - Lesion speckles are removed by the canonical filterLesionComponents
%    policy (clinicalConfig, bwconncomp 8) on EVERY path before counting,
%    so raw U-Net noise cannot fabricate a Severe-NPDR "4".
%
% INPUTS:
%   hemorrhageMask - logical MA/HE-combined mask (trained U-Net output or
%                    placeholder); speckle-filtered here canonically
%   quadrantMask   - uint8 label image (1-4) from partitionQuadrants.m
%   maskInfo       - struct with fields:
%       .maPresent, .exudatePresent (logical, observed findings)
%       .venousBeadingQuadrants / .venousBeadingStatus
%       .irmaQuadrants / .irmaStatus
%       .neovascularization / .neovascularizationStatus
%         ('PROXY_POSITIVE' | 'NOT_DETECTED' | 'UNAVAILABLE' | 'INVALID';
%         default derived: true=>PROXY_POSITIVE, false=>NOT_DETECTED
%         by the proxy - i.e. a proxy-negative, never a verified absence)
%       .vitreousHemorrhage / .vitreousStatus ('VERIFIED' if manual true
%         else UNAVAILABLE)
%     Missing *Status fields default conservatively: counts>0 with no
%     status => VERIFIED (explicit manual input honored); counts==0 with
%     no status => UNAVAILABLE (never assumed absent).
%
% OUTPUTS:
%   icdrGrade  - 0-4 when SUFFICIENT/PROXY-justified, else NaN
%   evidence   - cellstr: fired rules + assessed findings + missing-evidence
%                notices (Level-0 text describes ONLY examined evidence)
%   ruleReport - struct(status, grade, evidenceStatus, provenance, configVersion)
%
% Requires: Image Processing Toolbox (via filterLesionComponents).

ccfg = clinicalConfig();
evidence = {};

% --- Canonical speckle guard (both placeholder and trained paths) ---
hemMask = filterLesionComponents(logical(hemorrhageMask), ccfg.minBlobAreaPx, ccfg.connectivity);

% --- Resolve evidence statuses (absent status => conservative default) ---
[vbCount, vbStatus] = localResolve(maskInfo, 'venousBeadingQuadrants', 'venousBeadingStatus', ccfg);
[irmaCount, irmaStatus] = localResolve(maskInfo, 'irmaQuadrants', 'irmaStatus', ccfg);
[nvFlag, nvStatus] = localResolveNV(maskInfo, ccfg);
[vitFlag, vitStatus] = localResolveVit(maskInfo, ccfg);

% --- Per-quadrant hemorrhage COUNT (connected components, pinned conn. 8) ---
quadrantHemCounts = zeros(1, ccfg.nQuadrants);
for q = 1:ccfg.nQuadrants
    qMask = hemMask & (quadrantMask == q);
    cc = bwconncomp(qMask, ccfg.connectivity);
    quadrantHemCounts(q) = cc.NumObjects;
end

provenance = struct( ...
    'hemSource', 'merged MA/HE evidence channel (no MA-vs-hemorrhage separation; NOT validated hemorrhage detection)', ...
    'hemCounts', quadrantHemCounts, ...
    'hemThreshold', ccfg.severeHemPerQuadrant, ...
    'vbStatus', vbStatus, 'vbCount', vbCount, ...
    'irmaStatus', irmaStatus, 'irmaCount', irmaCount, ...
    'nvStatus', nvStatus, 'vitreousStatus', vitStatus, ...
    'specklePolicy', sprintf('filterLesionComponents>=%dpx,conn=%d', ccfg.minBlobAreaPx, ccfg.connectivity), ...
    'configVersion', ccfg.version);

statusOf = @(s) s; %#ok<NASGU>

% --- Level 4: vitreous (manual VERIFIED) or NV PROXY positive ---
if strcmp(vitStatus, ccfg.statusVerified) && vitFlag
    icdrGrade = 4;
    evidence{end+1} = 'Vitreous/preretinal hemorrhage present (manual input, VERIFIED) - Proliferative DR criterion.';
    ruleReport = localReport(ccfg.statusSufficient, 4, evidence, provenance, 'vitreous-VERIFIED', vbStatus, irmaStatus, nvStatus, vitStatus);
    return;
end
if strcmp(nvStatus, 'PROXY_POSITIVE') && nvFlag
    icdrGrade = 4;
    evidence{end+1} = 'NV SCREENING PROXY positive (PROXY - screening flag only, NOT a diagnosis): dense tortuous peridiscal vasculature flagged for manual review.';
    ruleReport = localReport(ccfg.statusProxy, 4, evidence, provenance, 'nv-proxy', vbStatus, irmaStatus, nvStatus, vitStatus);
    return;
end

% --- Severe NPDR triggers (only from assessable evidence) ---
fired = {};
if all(quadrantHemCounts > ccfg.severeHemPerQuadrant)
    fired{end+1} = sprintf(['Severe MA/HE-lesion burden (>%d per-quadrant components, MERGED MA/HE channel - NOT validated hemorrhage detection) ' ...
        'in all 4 quadrants (counts: %s) - the "4" of the 4-2-1 rule.'], ccfg.severeHemPerQuadrant, mat2str(quadrantHemCounts));
end
if (strcmp(vbStatus, ccfg.statusVerified) || strcmp(vbStatus, ccfg.statusProxy)) && vbCount >= ccfg.vbSevereQuadrants
    fired{end+1} = sprintf('Definite venous beading in %d quadrants (%s) - the "2" of the 4-2-1 rule.', vbCount, vbStatus);
end
if (strcmp(irmaStatus, ccfg.statusVerified) || strcmp(irmaStatus, ccfg.statusProxy)) && irmaCount >= ccfg.irmaSevereQuadrants
    fired{end+1} = sprintf('Prominent IRMA in %d quadrant(s) (%s) - the "1" of the 4-2-1 rule.', irmaCount, irmaStatus);
end
if ~isempty(fired)
    icdrGrade = 3;
    evidence = [evidence, fired];
    ruleReport = localReport(ccfg.statusSufficient, 3, evidence, provenance, 'severe-trigger', vbStatus, irmaStatus, nvStatus, vitStatus);
    return;
end

% --- No severe trigger fired: can we assert a definitive lower grade? ---
% Definitive 0/1/2 requires every trigger evaluable. VB, IRMA, and
% vitreous are UNAVAILABLE in this release (no detectors), so a lower
% grade cannot be asserted - return INSUFFICIENT_EVIDENCE + NaN with an
% honest account of what WAS examined.
missing = {};
if strcmp(vbStatus, ccfg.statusUnavailable) || strcmp(vbStatus, ccfg.statusInvalid)
    missing{end+1} = sprintf('venous beading (%s)', vbStatus);
end
if strcmp(irmaStatus, ccfg.statusUnavailable) || strcmp(irmaStatus, ccfg.statusInvalid)
    missing{end+1} = sprintf('IRMA (%s)', irmaStatus);
end
if strcmp(vitStatus, ccfg.statusUnavailable) || strcmp(vitStatus, ccfg.statusInvalid)
    missing{end+1} = sprintf('vitreous hemorrhage (%s)', vitStatus);
end
if strcmp(nvStatus, ccfg.statusInvalid)
    missing{end+1} = 'neovascularization screen (INVALID vessel input - PDR cannot be excluded)';
end
if ~isempty(missing)
    icdrGrade = NaN;
    assessed = sprintf(['Assessed findings - MA/HE lesion components by quadrant: %s (MERGED channel); ' ...
        'exudates present: %d; NV screening proxy: %s.'], mat2str(quadrantHemCounts), ...
        localHas(maskInfo, 'exudatePresent'), nvStatus);
    evidence{end+1} = [assessed ' Clinical rule determination INCOMPLETE: ' ...
        strjoin(missing, ', ') ' evidence unavailable - no definitive 0-4 rule grade asserted.'];
    ruleReport = localReport(ccfg.statusInsufficient, NaN, evidence, provenance, 'missing-evidence', vbStatus, irmaStatus, nvStatus, vitStatus);
    return;
end

% --- All evidence assessable (future detectors/manual review supplied
%     every status): definitive lower grades permitted ---
anyOtherLesion = localHas(maskInfo, 'exudatePresent') || any(quadrantHemCounts > 0);
if anyOtherLesion
    icdrGrade = 2;
    evidence{end+1} = sprintf(['Lesions present beyond microaneurysms (MA/HE-component counts by quadrant: %s (MERGED channel); ' ...
        'exudates present: %d), below Severe NPDR criteria; all 4-2-1 triggers assessable and negative.'], ...
        mat2str(quadrantHemCounts), localHas(maskInfo, 'exudatePresent'));
    ruleReport = localReport(ccfg.statusSufficient, 2, evidence, provenance, 'moderate', vbStatus, irmaStatus, nvStatus, vitStatus);
    return;
end
if localHas(maskInfo, 'maPresent')
    icdrGrade = 1;
    evidence{end+1} = 'Microaneurysms only (all 4-2-1 triggers assessable and negative).';
    ruleReport = localReport(ccfg.statusSufficient, 1, evidence, provenance, 'mild', vbStatus, irmaStatus, nvStatus, vitStatus);
    return;
end
icdrGrade = 0;
evidence{end+1} = 'No DR abnormalities in any assessed evidence channel (MA/HE, exudates, NV proxy, VB, IRMA, vitreous all assessable and negative).';
ruleReport = localReport(ccfg.statusSufficient, 0, evidence, provenance, 'none', vbStatus, irmaStatus, nvStatus, vitStatus);
end

% ------------------------------------------------------------------
function [count, status] = localResolve(info, countField, statusField, ccfg)
count = 0;
if isfield(info, countField), count = info.(countField); end
if isfield(info, statusField)
    status = info.(statusField);
else
    % Conservative default: an explicit positive count is honored as
    % manual VERIFIED input; zero with no status is UNAVAILABLE, never
    % a verified absence.
    if count > 0
        status = ccfg.statusVerified;
    else
        status = ccfg.statusUnavailable;
    end
end
end

function [flag, status] = localResolveNV(info, ccfg)
flag = false;
if isfield(info, 'neovascularization'), flag = logical(info.neovascularization); end
if isfield(info, 'neovascularizationStatus')
    status = info.neovascularizationStatus;
else
    if flag
        status = 'PROXY_POSITIVE';
    else
        % Proxy-negative: the screen did not flag - a proxy negative,
        % never a verified absence of neovascularization.
        status = ccfg.statusProxy;
    end
end
end

function [flag, status] = localResolveVit(info, ccfg)
flag = false;
if isfield(info, 'vitreousHemorrhage'), flag = logical(info.vitreousHemorrhage); end
if isfield(info, 'vitreousStatus')
    status = info.vitreousStatus;
else
    if flag
        status = ccfg.statusVerified; % manual examiner input honored
    else
        status = ccfg.statusUnavailable;
    end
end
end

function v = localHas(info, field)
v = isfield(info, field) && any(info.(field)(:));
end

function rep = localReport(status, grade, evidence, provenance, trigger, vbS, irmaS, nvS, vitS)
rep = struct('status', status, 'grade', grade, 'evidence', {evidence}, ...
    'trigger', trigger, 'provenance', provenance, ...
    'evidenceStatus', struct('vb', vbS, 'irma', irmaS, 'nv', nvS, 'vitreous', vitS));
end
