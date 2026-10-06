function [focusThresh, entropyThresh, calibInfo] = qualityLoadCalibration(cfg, scriptDir)
% qualityLoadCalibration: Override canonical thresholds with calibrated
% qualityThresholds.mat when present. Stage-1.
%
% INPUTS:
%   cfg       - struct from qualityConfig()
%   scriptDir - folder to look for qualityThresholds.mat (default: pwd)
%
% OUTPUTS:
%   focusThresh, entropyThresh - calibrated or canonical fallback
%   calibInfo - struct(isCalibrated, source, temperatureNote)
%
% Never errors on missing/corrupt file - falls back LOUDLY.

if nargin < 1 || isempty(cfg), cfg = qualityConfig(); end
if nargin < 2 || isempty(scriptDir), scriptDir = pwd; end

calibInfo = struct('isCalibrated', false, 'source', 'canonical qualityConfig.m', 'note', 'UNCALIBRATED PLACEHOLDER');
focusThresh = cfg.focusThresh;
entropyThresh = cfg.entropyThresh;

matPath = fullfile(scriptDir, cfg.calibratedMatName);
if exist(matPath, 'file')
    try
        S = load(matPath, 'focusThresh', 'entropyThresh', 'calibrationReport');
        focusThresh = S.focusThresh;
        entropyThresh = S.entropyThresh;
        calibInfo.isCalibrated = true;
        calibInfo.source = matPath;
        if isfield(S, 'calibrationReport')
            calibInfo.note = 'calibrated';
            calibInfo.report = S.calibrationReport;
        end
    catch ME
        warning('qualityLoadCalibration:loadFailed', ...
            'Found %s but could not load it (%s) - using canonical thresholds.', matPath, ME.message);
    end
end
end
