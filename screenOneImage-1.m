function result = screenOneImage(imagePath, models)
% screenOneImage: Thin wrapper around runScreeningPipeline.m - reads the
% image from disk and returns just the lean, JSON-friendly subset the
% web/mobile bridge (bridge_server.py) needs, discarding the intermediate
% masks/images a REST response shouldn't carry.
%
% REFACTORED (was the full pipeline copy-pasted here directly): now
% delegates the actual computation to runScreeningPipeline.m, the one
% shared core every entry point uses. This file's only job is the
% path-in/lean-struct-out shape the bridge expects.
%
% INPUTS:
%   imagePath - path to a fundus photo
%   models    - (optional) pre-loaded network struct, passed straight
%               through to runScreeningPipeline.m
%
% OUTPUT: result - struct with dl, conf, rule, evidence, nv,
%   gradCamOnDisc, status, errorMessage, focus, entropy, roiPassed -
%   unchanged shape from before, so bridge_server.py needs no changes.
%
% Requires: same toolboxes as runScreeningPipeline.m.

if nargin < 2
    models = [];
end

rawImage = imread(imagePath);
r = runScreeningPipeline(rawImage, models);

result = struct('status', r.status, 'errorMessage', r.errorMessage, ...
    'focus', r.focus, 'entropy', r.entropy, 'roiPassed', r.roiPassed, ...
    'dl', r.dlGrade, 'conf', r.confidence, 'rule', r.ruleGrade, ...
    'evidence', {r.evidence}, 'nv', r.nvFlagged, 'gradCamOnDisc', r.gradCamOnDisc);
end
