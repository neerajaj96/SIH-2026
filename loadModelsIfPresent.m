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

models = struct('haveTrainedModels', false);
if isfile('unet_Vessels.mat') && isfile('unet_MicroaneurysmsHemorrhages.mat') && ...
   isfile('unet_Exudates.mat') && isfile('trained_dr_grader.mat')
    S = load('unet_Vessels.mat','net'); models.vesselNet = S.net;
    S = load('unet_MicroaneurysmsHemorrhages.mat','net'); models.maheNet = S.net;
    S = load('unet_Exudates.mat','net'); models.exudateNet = S.net;
    S = load('trained_dr_grader.mat','net'); models.drNet = S.net;
    models.temperatureT = 1.5;
    if isfile('calibrated_temperature.mat')
        S = load('calibrated_temperature.mat','temperatureT'); models.temperatureT = S.temperatureT;
    end
    models.haveTrainedModels = true;
end
end
