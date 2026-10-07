function env = matlab_env_report(outDir)
% matlab_env_report: Records the exact MATLAB runtime environment into a
% caller-controlled evidence directory (default: tempdir - NEVER the
% repository, so evidence artifacts cannot pollute the tree).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB here) ***
%
% OUTPUT: env struct(matlabVersion, toolboxList, platform, arch,
%   gpuAvailable, pythonVersion, projectDir, assetInventory, reportPath).
% Asset inventory lists required .mat names with present true/false
% (presence only - loading is a separate gated check).

if nargin < 1 || isempty(outDir), outDir = tempdir; end
if ~isfolder(outDir), mkdir(outDir); end

env = struct();
env.matlabVersion = version();
env.platform = computer();
env.arch = computer('arch');
try
    v = ver;
    env.toolboxList = {v.Name};
catch
    env.toolboxList = {};
end
try
    g = gpuDeviceCount();
    env.gpuAvailable = g > 0;
catch
    env.gpuAvailable = 'unknown';
end
try
    env.pythonVersion = pyenv().Version;
catch
    env.pythonVersion = 'unavailable';
end
env.projectDir = pwd();
assets = {'unet_Vessels.mat', 'unet_MicroaneurysmsHemorrhages.mat', ...
    'unet_Exudates.mat', 'trained_dr_grader.mat', ...
    'calibrated_temperature.mat', 'qualityThresholds.mat'};
env.assetInventory = struct('name', {}, 'present', {});
for k = 1:numel(assets)
    env.assetInventory(k).name = assets{k};
    env.assetInventory(k).present = logical(isfile(fullfile(pwd(), assets{k})));
end
env.reportPath = fullfile(outDir, sprintf('matlab_env_%s.mat', datestr(now,'yyyymmdd_HHMMSS')));
save(env.reportPath, 'env');
fprintf('Environment recorded: MATLAB %s on %s; GPU available: %s; assets present %d/%d. See %s\n', ...
    env.matlabVersion, env.platform, mat2str(env.gpuAvailable), ...
    sum([env.assetInventory.present]), numel(assets), env.reportPath);
end
