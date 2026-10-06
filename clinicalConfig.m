function cfg = clinicalConfig()
% clinicalConfig: SINGLE versioned source of truth for every Stage-4
% clinical-reasoning operational threshold. No Stage-4 file may hardcode
% the constants below - read them from here. Geometric facts (e.g. 4
% quadrants) are documented where used, not duplicated here.
%
% MEDICAL-SAFETY DESIGN (conservative by default):
%   - Venous beading (VB) and IRMA have NO validated detectors in this
%     release: both statuses are UNAVAILABLE, and the engine treats them
%     as unevaluable - never as zero/absent. Extension points accept a
%     future VERIFIED or PROXY assessor without changing the 4-2-1
%     contract (see assessVenousBeading.m / assessIRMA.m extension-point
%     stubs - NOT detectors).
%   - Neovascularization is a SCREENING PROXY (never a diagnosis); every
%     consumer surface must say PROXY.
%   - Hemorrhage counts derive from the MERGED MA/HE channel (no true
%     MA-vs-hemorrhage separation exists); provenance always says so.
%
% OUTPUT: cfg struct, versioned. Add fields only; never rename/remove
%   without updating every Stage-4 consumer + contract tests.

cfg = struct();
cfg.version = '1.0.0-stage4';

% --- Lesion-component filtering (canonical speckle guard) ---
% bwconncomp connectivity is pinned explicitly (8) - never rely on the
% ambiguous/deprecated bwlabel default. Components below minBlobAreaPx
% are speckle/noise, not lesions, on BOTH the placeholder and trained
% paths (the old code filtered only the placeholder path, letting raw
% U-Net speckles fabricate severe-NPDR counts).
cfg.connectivity = 8;
cfg.minBlobAreaPx = 6;   % matches the historical placeholder floor

% --- Hemorrhage / MA-HE rule parameters (ICDR 4-2-1 "4") ---
cfg.severeHemPerQuadrant = 20;  % ">20 hemorrhages in EACH of 4 quadrants"
cfg.nQuadrants = 4;
% Evidence source tag for the "4" count (machine-readable provenance):
% the count derives from the MERGED MA/HE channel - there is NO true
% MA-vs-hemorrhage separation in this release. A future dedicated
% hemorrhage detector replaces this tag (extension point) without
% changing the 4-2-1 contract.
cfg.maHeSource = 'MA_HE_COMBINED';

% --- VB / IRMA (UNAVAILABLE this release - extension points only) ---
cfg.vbStatusDefault   = 'UNAVAILABLE';
cfg.irmaStatusDefault = 'UNAVAILABLE';
cfg.vbSevereQuadrants = 2;   % ICDR "2": beading in >=2 quadrants (for future assessors)
cfg.irmaSevereQuadrants = 1; % ICDR "1": IRMA in >=1 quadrant (for future assessors)

% --- Neovascularization screening proxy ---
cfg.nvTortuosityCutoff  = 1.2;  % peridiscal mean arc/chord above this is abnormal
cfg.nvDensityRatio      = 2.0;  % peridiscal density above 2x whole-image density
cfg.nvSearchRadii       = 2.5;  % peridiscal zone radii in OD radii (NVD search zone)
cfg.nvMinVesselFrac     = 0.005;% vessel-pixel fraction below this => mask unusable => INVALID
cfg.nvMinSegments       = 3;    % fewer measurable peridiscal segments => INVALID

% --- Landmark validity bands (fractions of ROI diameter unless noted) ---
cfg.odRadiusFallbackFrac = 0.08; % fallback radius = 8% of min(frame) - explicit FALLBACK state
cfg.foveaSearchInnerRadii = 4;   % fovea band inner edge, in OD radii
cfg.foveaSearchOuterRadii = 6;   % fovea band outer edge, in OD radii
cfg.foveaBandHalfWidthRadii = 1.5; % horizontal meridian half-width, in OD radii
cfg.odFoveaDistMinDiam = 2.0;    % sanity band for OD-fovea distance (disc diameters)
cfg.odFoveaDistMaxDiam = 2.5;

% --- Quadrant fallback policy (conservative default) ---
cfg.allowFallbackQuadrants = false; % FALLBACK landmarks usable ONLY if explicitly true
% UNRELIABLE / INVALID landmarks always block quadrant reasoning.

% --- Evidence status vocabulary (frozen strings) ---
cfg.statusVerified    = 'VERIFIED';
cfg.statusProxy       = 'PROXY';
cfg.statusUnavailable = 'UNAVAILABLE';
cfg.statusInvalid     = 'INVALID';
cfg.statusNotDetected = 'NOT_DETECTED';
cfg.statusSufficient  = 'SUFFICIENT';
cfg.statusInsufficient = 'INSUFFICIENT_EVIDENCE';
end
