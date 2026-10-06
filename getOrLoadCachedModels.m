function models = getOrLoadCachedModels()
% getOrLoadCachedModels: Loads the four trained networks ONCE per
% MATLAB session and reuses them on every later call, via a persistent
% variable.
%
% WHY THIS EXISTS: bridge_server.py keeps one MATLAB engine process
% alive across every web request (starting one per request would be far
% too slow - engine startup alone takes several seconds). But
% screenOneImage.m originally called loadModelsIfPresent() directly
% whenever no models were passed in, and bridge_server.py never passes
% any - so every single /screen request was calling load() on all four
% .mat files from disk, all over again, inside what's supposed to be a
% fast per-request path. Caught during a scalability review, not before.
%
% MATLAB's `persistent` keyword makes cachedModels survive across
% multiple calls to this function WITHIN one MATLAB session - exactly
% the lifetime of one bridge_server.py process - without needing any
% change on the Python side.
%
% Requires: nothing beyond base MATLAB (calls loadModelsIfPresent.m,
% which needs whatever that function's own model files need).

persistent cachedModels cachedTempSerial
if isempty(cachedModels)
    cachedModels = loadModelsIfPresent();
    cachedTempSerial = localTempSerial();
end
models = cachedModels;
% Temperature hot-reload: calibration is a tiny scalar .mat, so re-check
% its timestamp on every call and refresh ONLY the scalar when it
% changes - networks are never reloaded (no repeated model loading).
% Without this, recalibration required a bridge restart to take effect.
if models.haveTrainedModels
    cur = localTempSerial();
    if ~isequal(cur, cachedTempSerial)
        if isfile('calibrated_temperature.mat')
            try
                S = load('calibrated_temperature.mat','temperatureT');
                models.temperatureT = S.temperatureT;
                cachedModels.temperatureT = S.temperatureT;
            catch
            end
        else
            models.temperatureT = 1.5;
            cachedModels.temperatureT = 1.5;
        end
        cachedTempSerial = cur;
    else
        models.temperatureT = cachedModels.temperatureT;
    end
end
end

function s = localTempSerial()
% Comparable serial for the temperature artifact: datenum + byte size when
% present, '' when absent. dir() is one cheap stat call - no model I/O.
% Bytes are included so two calibrations within one datenum tick still
% invalidate (datenum granularity alone could miss sub-second rewrites).
if isfile('calibrated_temperature.mat')
    d = dir('calibrated_temperature.mat');
    s = sprintf('%.6f_%d', d.datenum, d.bytes);
else
    s = '';
end
end
