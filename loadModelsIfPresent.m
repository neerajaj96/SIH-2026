function models = loadModelsIfPresent()
% loadModelsIfPresent: Loads the four trained networks from disk if all
% four .mat files exist next to the current working directory, else
% returns haveTrainedModels=false so callers fall back to SIMULATED mode.
%
% Extracted out of screenOneImage.m into its own file so
% getOrLoadCachedModels.m can wrap it in a persistent cache without
% duplicating the loading logic - this was a local function living
% inside screenOneImage.m only, which is exactly the kind of small
% duplication that's easy to let drift once a second caller needs the
% same thing.
%
% Requires: nothing beyond base MATLAB.

modelDir = localModelDir();
models = struct('haveTrainedModels', false, 'calibrationState', 'UNAVAILABLE');
if isfile(fullfile(modelDir, 'unet_Vessels.mat')) && isfile(fullfile(modelDir, 'unet_MicroaneurysmsHemorrhages.mat')) && ...
   isfile(fullfile(modelDir, 'unet_Exudates.mat')) && isfile(fullfile(modelDir, 'trained_dr_grader.mat'))
    S = load(fullfile(modelDir, 'unet_Vessels.mat'),'net'); models.vesselNet = S.net;
    S = load(fullfile(modelDir, 'unet_MicroaneurysmsHemorrhages.mat'),'net'); models.maheNet = S.net;
    S = load(fullfile(modelDir, 'unet_Exudates.mat'),'net'); models.exudateNet = S.net;
    S = load(fullfile(modelDir, 'trained_dr_grader.mat'),'net'); models.drNet = S.net;
    % Temperature artifact validation (isfile alone never proves the
    % artifact belongs to THIS model): requires a finite scalar T plus
    % matching class ordering/count when provenance is present. Legacy
    % artifacts without provenance are accepted as CALIBRATED_VALID with
    % provenance 'legacy (no provenance block)' - they predate this check
    % but carry the same scalar contract.
    models.temperatureT = 1.5;
    models.calibrationState = 'UNCALIBRATED_FALLBACK';
    if isfile(fullfile(modelDir, 'calibrated_temperature.mat'))
        try
            S = load(fullfile(modelDir, 'calibrated_temperature.mat'));
            okT = isfield(S,'temperatureT') && isscalar(S.temperatureT) && isfinite(S.temperatureT) && S.temperatureT > 0;
            okCls = ~isfield(S,'calibrationProvenance') || ...
                (isequal(S.calibrationProvenance.numClasses, 5) && isequal(S.calibrationProvenance.classOrdering(:), (0:4)'));
            if okT && okCls
                models.temperatureT = S.temperatureT;
                if isfield(S,'calibrationProvenance')
                    models.calibrationState = 'CALIBRATED_VALID';
                else
                    models.calibrationState = 'CALIBRATED_VALID';
                    warning(['loadModelsIfPresent:legacyArtifact - calibrated_temperature.mat has no provenance block; ' ...
                             'accepted on scalar contract (finite T>0). Re-run calibrateTemperature.m to stamp provenance.']);
                end
            else
                warning(['loadModelsIfPresent:artifactMismatch - calibrated_temperature.mat failed validation ' ...
                         '(non-finite T or class-ordering mismatch); falling back to T=1.5 UNCALIBRATED.']);
                models.calibrationState = 'CALIBRATED_MISMATCH';
            end
        catch ME
            warning('loadModelsIfPresent:artifactUnreadable - calibrated_temperature.mat unreadable (%s); T=1.5 UNCALIBRATED.', ME.message);
            models.calibrationState = 'CALIBRATED_MISMATCH';
        end
    end
    models.haveTrainedModels = true;
    models.modelDir = modelDir;
end
end

function d = localModelDir()
% Canonical artifact resolution: SIH_PROJECT_DIR (bridge/production) ->
% this file's folder (MATLAB path) -> pwd (legacy fallback, LOUD).
% Never silently loads weights from an unrelated working directory: the
% chosen directory is recorded on the output (models.modelDir) and a
% warning names the fallback whenever pwd is used while a project dir
% was available.
cands = {};
env = getenv('SIH_PROJECT_DIR');
if ~isempty(env), cands{end+1} = env; end
try, cands{end+1} = fileparts(mfilename('fullpath')); catch, end
cands{end+1} = pwd;
names = {'unet_Vessels.mat','unet_MicroaneurysmsHemorrhages.mat', ...
    'unet_Exudates.mat','trained_dr_grader.mat','calibrated_temperature.mat'};
for k = 1:numel(cands)
    dk = cands{k};
    if isempty(dk) || ~isfolder(dk), continue; end
    for n = 1:numel(names)
        if isfile(fullfile(dk, names{n}))
            if strcmp(dk, pwd) && numel(cands) > 1 && ~isempty(env) && isfolder(env) && ~strcmp(env, pwd)
                warning(['loadModelsIfPresent:cwdFallback - SIH_PROJECT_DIR is set (%s) but holds no artifacts; ' ...
                    'falling back to pwd (%s). Serving from the wrong CWD silently runs SIMULATED/UNCALIBRATED.'], env, pwd);
            end
            d = dk;
            return;
        end
    end
end
if ~isempty(env) && isfolder(env), d = env; else, d = pwd; end
end
