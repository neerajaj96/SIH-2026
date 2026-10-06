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

persistent cachedModels
if isempty(cachedModels)
    cachedModels = loadModelsIfPresent();
end
models = cachedModels;
end
