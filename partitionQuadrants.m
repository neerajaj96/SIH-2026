function [quadrantMask, qValid] = partitionQuadrants(imageSize, foveaCenter, odCenter, varargin)
% partitionQuadrants: Divides the image into the 4 quadrants ICDR grading
% actually uses - centered on the FOVEA/macula, not the optic disc, per
% standard clinical grading grids. Divided by horizontal/vertical lines
% through the fovea; nasal/temporal sides are determined from which side
% of the fovea the optic disc falls on (works for either eye without
% needing separate left/right-eye metadata).
%
% Quadrant labels: 1 = Superotemporal, 2 = Superonasal,
%                  3 = Inferonasal,    4 = Inferotemporal
%
% VALIDITY POLICY (conservative): an invalid/unavailable fovea must NOT
% silently generate whole-frame quadrants. qValid is one of
% VALID | FALLBACK | INVALID. Optional landmark-validity inputs (from
% localizeOpticDiscFovea.m lmStatus) drive the policy; when omitted the
% legacy whole-frame behavior is preserved with qValid='VALID' for
% backward compatibility (callers that know validity must pass it).
%   - fovea/OD CONFIDENT -> VALID
%   - either FALLBACK -> FALLBACK, usable ONLY if clinicalConfig
%     .allowFallbackQuadrants is true (default false)
%   - either UNRELIABLE/INVALID, NaN geometry, or unusable under policy ->
%     INVALID: quadrantMask is zeros (NOT whole-frame quadrants) and the
%     caller must block 4-2-1 reasoning.
%
% INPUTS:
%   imageSize   - [H W]
%   foveaCenter - [x y], from localizeOpticDiscFovea.m
%   odCenter    - [x y], from localizeOpticDiscFovea.m
%   odValidity, foveaValidity - optional 'CONFIDENT'|'FALLBACK'|
%     'UNRELIABLE'|'INVALID' (default 'CONFIDENT' = legacy behavior)
%
% OUTPUTS:
%   quadrantMask - uint8 [H W] label image, values 1-4 (zeros if INVALID)
%   qValid       - 'VALID' | 'FALLBACK' | 'INVALID'
%
% Requires: nothing beyond base MATLAB.

odValidity = 'CONFIDENT'; foveaValidity = 'CONFIDENT';
if numel(varargin) >= 1 && ~isempty(varargin{1}), odValidity = varargin{1}; end
if numel(varargin) >= 2 && ~isempty(varargin{2}), foveaValidity = varargin{2}; end

H = imageSize(1); W = imageSize(2);

if any(isnan(foveaCenter(:))) || any(isnan(odCenter(:)))
    % NaN geometry (e.g. UNRELIABLE landmarks) can never define quadrants.
    quadrantMask = zeros(H, W, 'uint8');
    qValid = 'INVALID';
    return;
end
usable = localUsable(odValidity, foveaValidity);
if strcmp(usable, 'INVALID')
    quadrantMask = zeros(H, W, 'uint8');
    qValid = 'INVALID';
    return;
end
qValid = usable; % VALID or FALLBACK (FALLBACK usable only per config)

[xx, yy] = meshgrid(1:W, 1:H);

isSuperior = yy < foveaCenter(2); % image y grows downward; "superior" = upper half of frame

if (odCenter(1) - foveaCenter(1)) >= 0
    isTemporal = xx < foveaCenter(1); % OD to the right of fovea -> nasal is right -> temporal is left
else
    isTemporal = xx > foveaCenter(1); % OD to the left of fovea -> nasal is left -> temporal is right
end

quadrantMask = zeros(H, W, 'uint8');
quadrantMask(isSuperior & isTemporal)   = 1; % Superotemporal
quadrantMask(isSuperior & ~isTemporal)  = 2; % Superonasal
quadrantMask(~isSuperior & ~isTemporal) = 3; % Inferonasal
quadrantMask(~isSuperior & isTemporal)  = 4; % Inferotemporal
end

% ------------------------------------------------------------------
function usable = localUsable(odV, fovV)
% NaN geometry or UNRELIABLE/INVALID landmarks always block. FALLBACK
% needs explicit permission (conservative default: blocked).
ccfg = clinicalConfig();
bad = @(v) strcmp(v,'UNRELIABLE') || strcmp(v,'INVALID');
if bad(odV) || bad(fovV)
    usable = 'INVALID';
    return;
end
if strcmp(odV,'FALLBACK') || strcmp(fovV,'FALLBACK')
    if ccfg.allowFallbackQuadrants
        usable = 'FALLBACK';
    else
        usable = 'INVALID';
    end
    return;
end
usable = 'VALID';
end
