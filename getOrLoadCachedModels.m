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
% Forward-compat: caches created before calibrationState existed gain it
% here (revalidated, not assumed).
if models.haveTrainedModels && ~isfield(models, 'calibrationState')
    [t0, s0] = localValidateTemp();
    models.temperatureT = t0; models.calibrationState = s0;
    cachedModels.temperatureT = t0; cachedModels.calibrationState = s0;
end
% Temperature hot-reload: calibration is a tiny scalar .mat, so re-check
% its serial on every call and refresh ONLY the scalar + match state when
% it changes - networks are never reloaded (no repeated model loading).
% Without this, recalibration required a bridge restart to take effect.
% Validation mirrors loadModelsIfPresent (finite T>0, class-order match).
if models.haveTrainedModels
    cur = localTempSerial();
    if ~isequal(cur, cachedTempSerial)
        [tNew, stateNew] = localValidateTemp();
        models.temperatureT = tNew;
        models.calibrationState = stateNew;
        cachedModels.temperatureT = tNew;
        cachedModels.calibrationState = stateNew;
        cachedTempSerial = cur;
    else
        models.temperatureT = cachedModels.temperatureT;
        models.calibrationState = cachedModels.calibrationState;
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

function [t, state] = localValidateTemp()
% Same acceptance rules as loadModelsIfPresent (kept in sync; contract
% tests pin both spellings): finite T>0 + class-order match =>
% CALIBRATED_VALID; present-but-bad => CALIBRATED_MISMATCH + fallback;
% absent => UNCALIBRATED_FALLBACK.
t = 1.5; state = 'UNCALIBRATED_FALLBACK';
if ~isfile('calibrated_temperature.mat')
    return;
end
try
    S = load('calibrated_temperature.mat');
    okT = isfield(S,'temperatureT') && isscalar(S.temperatureT) && isfinite(S.temperatureT) && S.temperatureT > 0;
    okCls = ~isfield(S,'calibrationProvenance') || ...
        (isequal(S.calibrationProvenance.numClasses, 5) && isequal(S.calibrationProvenance.classOrdering(:), (0:4)'));
    if okT && okCls
        t = S.temperatureT;
        state = 'CALIBRATED_VALID';
    else
        state = 'CALIBRATED_MISMATCH';
    end
catch
    state = 'CALIBRATED_MISMATCH';
end
end
