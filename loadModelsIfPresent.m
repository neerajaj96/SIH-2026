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

models = struct('haveTrainedModels', false, 'calibrationState', 'UNAVAILABLE');
if isfile('unet_Vessels.mat') && isfile('unet_MicroaneurysmsHemorrhages.mat') && ...
   isfile('unet_Exudates.mat') && isfile('trained_dr_grader.mat')
    S = load('unet_Vessels.mat','net'); models.vesselNet = S.net;
    S = load('unet_MicroaneurysmsHemorrhages.mat','net'); models.maheNet = S.net;
    S = load('unet_Exudates.mat','net'); models.exudateNet = S.net;
    S = load('trained_dr_grader.mat','net'); models.drNet = S.net;
    % Temperature artifact validation (isfile alone never proves the
    % artifact belongs to THIS model): requires a finite scalar T plus
    % matching class ordering/count when provenance is present. Legacy
    % artifacts without provenance are accepted as CALIBRATED_VALID with
    % provenance 'legacy (no provenance block)' - they predate this check
    % but carry the same scalar contract.
    models.temperatureT = 1.5;
    models.calibrationState = 'UNCALIBRATED_FALLBACK';
    if isfile('calibrated_temperature.mat')
        try
            S = load('calibrated_temperature.mat');
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
end
end
