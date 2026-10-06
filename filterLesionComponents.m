function clean = filterLesionComponents(mask, minAreaPx, connectivity)
% filterLesionComponents: CANONICAL lesion-component speckle guard.
% Removes connected components smaller than minAreaPx so segmentation
% speckles (single-pixel noise, placeholder top-hat dust, raw U-Net
% false-positive dots) cannot fabricate hemorrhage counts that drive a
% Severe-NPDR ("4") call. MUST be applied on BOTH the placeholder and
% trained-mask paths before any counting (the old pipeline filtered only
% the placeholder path).
%
% Connectivity is explicit (default 8 per clinicalConfig) - never rely on
% bwlabel's ambiguous/deprecated default.
%
% INPUTS: mask (logical), minAreaPx, connectivity (4/8)
% OUTPUT: clean (logical, same size)
%
% Requires: Image Processing Toolbox (bwconncomp).

if nargin < 3 || isempty(connectivity), connectivity = 8; end
if nargin < 2 || isempty(minAreaPx)
    c = clinicalConfig();
    minAreaPx = c.minBlobAreaPx; connectivity = c.connectivity;
end
mask = logical(mask);
if ~any(mask(:))
    clean = mask;
    return;
end
cc = bwconncomp(mask, connectivity);
if cc.NumObjects == 0
    clean = mask;
    return;
end
sizes = cellfun(@numel, cc.PixelIdxList);
keep = sizes >= minAreaPx;
clean = false(size(mask));
for k = find(keep)
    clean(cc.PixelIdxList{k}) = true;
end
end
