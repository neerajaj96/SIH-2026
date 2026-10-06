function quadrantMask = partitionQuadrants(imageSize, foveaCenter, odCenter)
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
% INPUTS:
%   imageSize   - [H W]
%   foveaCenter - [x y], from localizeOpticDiscFovea.m
%   odCenter    - [x y], from localizeOpticDiscFovea.m
%
% OUTPUT:
%   quadrantMask - uint8 [H W] label image, values 1-4
%
% Requires: nothing beyond base MATLAB.

H = imageSize(1); W = imageSize(2);
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
