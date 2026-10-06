function cfg = explainabilityConfig()
% explainabilityConfig: SINGLE versioned source of truth for every
% Stage-5 explainability/calibration/disagreement threshold. gradingConfig.m
% is FROZEN - nothing here duplicates or overrides it.
%
% OUTPUT: cfg struct, versioned. Add fields only.

cfg = struct();
cfg.version = '1.0.0-stage5';

% --- Grad-CAM target selection ---
% ReductionLayer stays "prob" (validated: per-class probabilities exist).
cfg.reductionLayer = 'prob';
% FeatureLayer: "" = framework auto-select (the only safe default without
% a MATLAB session: the exact DenseNet-121 last-conv name was never
% recorded in this repo - train_DR_Grader logs the STEM conv at train
% time, which is not the explanation layer). To pin a layer, confirm it
% via net.Layers in MATLAB and set the name here; explainGradCAM then
% validates it against the actual graph and returns UNAVAILABLE/INVALID
% for unknown names - never silently substitutes another layer.
% MATLAB-GATED: confirming + pinning the exact name.
cfg.featureLayer = '';

% --- Activation-region mathematics (deterministic, documented) ---
% Activation mass = score values above highFrac * max(map) (not just the
% single peak pixel: a peak-only test is blind to diffuse mislocalization).
cfg.highFrac = 0.5;            % "active region" threshold as fraction of map max
cfg.borderBandFrac = 0.06;     % outer 6% frame band = border zone
cfg.borderMassFrac = 0.5;      % active mass in border zone above this => border-dominated
cfg.discZoneRadii = 1.5;       % disc zone radius in OD radii (matches gradCamOnDisc rule)
cfg.discMassFrac = 0.5;        % active mass in disc zone above this (with lesions elsewhere) => disc-dominated

% --- Confidence review heuristic (DATA-GATED, not a referral rule) ---
% Any threshold here lacks real validation: it may only trigger human
% review, never a clinical decision. Tune on labeled data only.
cfg.lowConfidenceReview = 0.6; % below this => confidenceStatus LOW (review heuristic)

% --- Disagreement escalation policy ---
cfg.escalateOnDisagree = true;   % numeric DL-vs-rule disagreement escalates
cfg.escalateOnInsufficient = true; % INSUFFICIENT_EVIDENCE always escalates (with reasons)
cfg.escalateOnProxyOnly = false; % PROXY-only rule result: report, escalate only with other flags

% --- Evidence status vocabulary (mirrors clinicalConfig; frozen strings) ---
cfg.explainValid = 'VALID';
cfg.explainDegraded = 'DEGRADED';
cfg.explainUnavailable = 'UNAVAILABLE';
cfg.explainInvalid = 'INVALID';
cfg.confCalibrated = 'CALIBRATED';
cfg.confUncalibrated = 'UNCALIBRATED';
end
