function cfg = qualityConfig()
% qualityConfig: SINGLE canonical source for every Image Quality Assessment
% + Enhancement constant. Stage-1.
%
% WHY: focus/entropy thresholds (8 / 3.5) were triplicated across
%   assessAndEnhanceImage.m defaults, runScreeningPipeline.m:55 literals,
%   and production_inference.m:62-63 display-only copies. ROI/enhancement
%   constants (seed 12, erode disk 8, bg fraction 0.05 floor 15, CLAHE
%   0.01 [8 8], laplacian alpha 0.2, entropy bins 256) were buried inline.
%   Every caller must read this file instead of hardcoding.
%
% CALIBRATION OVERRIDE: calibrateQualityThresholds.m may write
%   qualityThresholds.mat with focusThresh/entropyThresh. Callers that need
%   calibrated values should call qualityConfig() then override with
%   qualityLoadCalibration(cfg) - see that function. This file itself never
%   reads disk, so it stays deterministic and testable.
%
% OUTPUT: cfg struct, versioned. Add fields only (never rename/remove
%   without updating all 4 call sites + wrapper).

cfg = struct();
cfg.version = '1.1.0-stage1-config';

% --- Gate thresholds (PLACEHOLDERS until calibrated - see
%     calibrateQualityThresholds.m). Canonical values; do not duplicate. ---
cfg.focusThresh   = 8;
cfg.entropyThresh = 3.5;

% --- BORDERLINE band: within this relative margin above threshold the
%     decision is BORDERLINE (gradeable-with-warning), not PASS. ---
cfg.borderlineFocusMargin   = 0.15;  % 15% above focusThresh
cfg.borderlineEntropyMargin = 0.10;  % 10% above entropyThresh

% --- ROI detection ---
cfg.roiSeedThresh       = 12;   % grayFull > 12 seed (was inline)
cfg.roiErodeDisk        = 8;    % floor for edge-shave radius (px)
cfg.roiErodeRelFrac     = 0.007;% + 0.7% of ROI diameter, so 4288px IDRiD gets ~20px shave not 8px
cfg.roiMinCoverageFrac  = 0.05; % ROI must cover >=5% of frame else degenerate
cfg.roiCircularityMin   = 0.55; % 4*pi*Area/Perim^2 lower bound for fundus FOV

% --- Focus / entropy math ---
cfg.laplacianAlpha = 0.2;   % fspecial('laplacian', alpha)
cfg.entropyBins    = 256;   % masked Shannon histogram bins over [0,1]

% --- Flat-field / enhancement ---
cfg.bgFraction = 0.05;      % structuring element = 5% of ROI diameter
cfg.bgFloorPx  = 15;        % floor so tiny ROIs do not collapse
cfg.claheClipLimit = 0.01;
cfg.claheNumTiles  = [8 8];
cfg.enhanceFlatThresh = 0.12;   % gray ROI P95-P5 below this => flat/noisy: halve CLAHE to avoid noise amplification
cfg.enhanceFlatClip   = 0.005;  % reduced ClipLimit for flat images
cfg.denoiseMinDiameter = 256;   % skip imnlmfilt below this ROI diameter (oversmooths tiny thumbs)

% --- Extended quality metrics (assessFundusQuality.m) ---
cfg.illumLowPctl   = 5;     % V-channel low percentile for illumination
cfg.illumHighPctl  = 95;    % V-channel high percentile
cfg.underexposedV  = 0.15;  % fraction of ROI with V < 0.15 => underexposed
cfg.overexposedV   = 0.92;  % fraction of ROI with V > 0.92 => overexposed
cfg.specularV      = 0.97;  % V > 0.97 counted as specular artefact
cfg.contrastPctlLo = 5;
cfg.contrastPctlHi = 95;

% --- Calibration artifact names ---
cfg.calibratedMatName  = 'qualityThresholds.mat';
cfg.calibratedMetaName = 'qualityThresholds.meta.json';
end
