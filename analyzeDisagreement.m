function dis = analyzeDisagreement(dlGrade, confidence, confStatus, ruleGrade, ruleStatus, nvStatus, qualityDecision, explainStatus, varargin)
% analyzeDisagreement: Structured DL-vs-clinical disagreement model.
%
% Orthogonal dimensions (never one mutually-exclusive enum - uncertainty
% in one dimension must not collapse another):
%   clinicalEvidenceStatus - ruleStatus passthrough (SUFFICIENT /
%     INSUFFICIENT_EVIDENCE / PROXY / INVALID)
%   gradeRelationship      - AGREE | NUMERIC_DISAGREE | NOT_COMPARABLE
%     (NOT_COMPARABLE when either grade is NaN - missing evidence is
%     NEVER manufactured into disagreement)
%   confidenceStatus       - CALIBRATED | UNCALIBRATED | LOW_CONFIDENCE
%     (LOW only a review heuristic per explainabilityConfig; DATA-GATED
%     as any referral rule)
%   explanationStatus      - gradCAM reliability passthrough
%   escalate               - logical review recommendation (conservative:
%     any disagreement, insufficiency, proxy-only-with-flags, low
%     confidence, or degraded explanation escalates)
%   reasons{}              - one string per raised dimension
%
% INPUTS: dlGrade (NaN if simulated), confidence (NaN if simulated),
%   confStatus ('CALIBRATED'/'UNCALIBRATED' from temperature state),
%   ruleGrade (NaN if INSUFFICIENT), ruleStatus, nvStatus (informational),
%   qualityDecision ('PASS'/'BORDERLINE'/'FAIL'), explainStatus,
%   'Config' (explainabilityConfig), 'Confidence' threshold override.
%
% Requires: base MATLAB only (pure logic).

p = inputParser;
addParameter(p, 'Config', explainabilityConfig(), @isstruct);
parse(p, varargin{:});
cfg = p.Results.Config;

reasons = {};

% --- gradeRelationship (NaN-safe: missing evidence is not disagreement) ---
if isnan(dlGrade) || isnan(ruleGrade)
    gradeRelationship = 'NOT_COMPARABLE';
    if isnan(ruleGrade)
        reasons{end+1} = 'Clinical rule grade unavailable (INSUFFICIENT_EVIDENCE or INVALID) - not counted as disagreement.';
    end
    if isnan(dlGrade)
        reasons{end+1} = 'DL grade unavailable (simulated mode) - not counted as disagreement.';
    end
elseif dlGrade == ruleGrade
    gradeRelationship = 'AGREE';
else
    gradeRelationship = 'NUMERIC_DISAGREE';
    reasons{end+1} = sprintf('Numeric grade disagreement: DL=%d vs rule=%d - recommend manual review.', dlGrade, ruleGrade);
end

% --- confidenceStatus (fallback T is never "calibrated") ---
if strcmp(confStatus, cfg.confCalibrated) && ~isnan(confidence) && confidence >= cfg.lowConfidenceReview
    confidenceStatus = cfg.confCalibrated;
elseif strcmp(confStatus, cfg.confCalibrated)
    confidenceStatus = 'LOW_CONFIDENCE';
    reasons{end+1} = sprintf('DL confidence %.2f below review heuristic %.2f (DATA-GATED heuristic, not a referral rule).', confidence, cfg.lowConfidenceReview);
else
    confidenceStatus = cfg.confUncalibrated;
end

% --- escalation (conservative OR over dimensions) ---
escalate = strcmp(gradeRelationship, 'NUMERIC_DISAGREE') ...
    || strcmp(ruleStatus, 'INSUFFICIENT_EVIDENCE') || strcmp(ruleStatus, 'INVALID') ...
    || strcmp(confidenceStatus, 'LOW_CONFIDENCE') ...
    || strcmp(explainStatus, cfg.explainDegraded) || strcmp(explainStatus, cfg.explainInvalid);
if strcmp(ruleStatus, 'INSUFFICIENT_EVIDENCE')
    reasons{end+1} = 'Clinical evidence insufficient - DL grade stands alone; human review required.';
end
if strcmp(explainStatus, cfg.explainDegraded) || strcmp(explainStatus, cfg.explainInvalid)
    reasons{end+1} = sprintf('Explanation reliability %s - do not rely on heatmap.', explainStatus);
end

dis = struct('clinicalEvidenceStatus', ruleStatus, 'gradeRelationship', gradeRelationship, ...
    'confidenceStatus', confidenceStatus, 'explanationStatus', explainStatus, ...
    'escalate', escalate, 'reasons', {reasons}, 'configVersion', cfg.version);
end
