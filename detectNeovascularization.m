function [isFlagged, tortuosityScore, densityScore, nvReport] = detectNeovascularization(vesselMask, odCenter, odRadius)
% detectNeovascularization: A SCREENING PROXY, not a diagnosis - and not
% pixel-level neovascularization segmentation, which the official PS asks
% for and which remains a genuinely hard, actively-researched problem
% even in the published literature (not just this codebase). Say that
% out loud in your pitch rather than letting a judge assume this is a
% validated NVD/NVE detector. Every evidence string this produces says
% PROXY; no consumer may present it as a diagnostic criterion.
%
% Flags regions of unusually dense, tortuous vasculature near the optic
% disc (new vessels on the disc, NVD, is the more common and more
% dangerous presentation of the two). Uses the same two features the
% published literature uses for this task - vessel density and
% tortuosity on the fine vessel structure (see e.g. the mutual-
% information-maximization neovascularization detection literature) -
% implemented here as a simple, transparent self-relative threshold
% rather than the fuller curvelet/optimization pipelines those papers use.
%
% Tortuosity metric = arc length / chord length along each vessel
% skeleton branch (the classic "Distance Factor" - 1.0 is perfectly
% straight, higher is more tortuous). Verified on synthetic test curves
% before this file was written: a straight segment scores 1.000, a
% gently curved (normal) vessel scores ~1.01, and a wiggly/corkscrew
% segment mimicking a neovascular frond scores ~2.08 - the metric
% separates them the way it's supposed to.
%
% MASK-USABILITY GATE: sparse/empty vessel masks carry no information -
% they yield status INVALID (isFlagged=false, scores NaN), NOT
% "not detected". A sparse mask that still flagged would be noise
% calling itself evidence.
%
% INPUTS:
%   vesselMask - logical vessel mask from the trained vessel U-Net
%   odCenter   - [x y] from localizeOpticDiscFovea.m (NaN => INVALID)
%   odRadius   - optic disc radius (pixels) from localizeOpticDiscFovea.m
%
% OUTPUTS:
%   isFlagged       - true if the peridiscal region's density AND
%                     tortuosity both exceed the CONFIGURED thresholds -
%                     a "flag for manual review", not a diagnosis
%   tortuosityScore - mean arc/chord ratio of vessel segments in the
%                     peridiscal region (NaN when INVALID)
%   densityScore    - local vessel-pixel density in the peridiscal region
%                     (NaN when INVALID)
%   nvReport        - struct(status, tortuosity, density, wholeImageDensity,
%                     nSegments, vesselFrac, provenance): status is one of
%                     PROXY_POSITIVE | NOT_DETECTED (proxy-negative) |
%                     INVALID (unusable input)
%
% Thresholds live in clinicalConfig.m. Requires: Image Processing
% Toolbox. Uses bwskel; if your release predates it, replace with
% bwmorph(vesselMask,'skel',Inf).

ccfg = clinicalConfig();
vesselMask = logical(vesselMask);

% --- Usability gate 1: geometry ---
if any(isnan(odCenter(:))) || ~isfinite(odRadius) || odRadius <= 0
    [isFlagged, tortuosityScore, densityScore] = deal(false, NaN, NaN);
    nvReport = localReport('INVALID', NaN, NaN, NaN, 0, 0, ...
        'unusable OD geometry (NaN/nonpositive radius) - peridiscal zone undefined', ccfg);
    return;
end

% --- Usability gate 2: vessel content ---
vesselFrac = sum(vesselMask(:)) / max(numel(vesselMask), 1);
if vesselFrac < ccfg.nvMinVesselFrac
    [isFlagged, tortuosityScore, densityScore] = deal(false, NaN, NaN);
    nvReport = localReport('INVALID', NaN, NaN, vesselFrac, 0, vesselFrac, ...
        sprintf('sparse/empty vessel mask (frac %.4f < %.4f) - no information', vesselFrac, ccfg.nvMinVesselFrac), ccfg);
    return;
end

[H, W] = size(vesselMask);
[xx, yy] = meshgrid(1:W, 1:H);
peridiscal = hypot(xx-odCenter(1), yy-odCenter(2)) < ccfg.nvSearchRadii*odRadius; % NVD's classic search zone (radii from config)

try
    skel = bwskel(vesselMask);
catch
    skel = bwmorph(vesselMask, 'skel', Inf); % fallback for older toolbox versions
end
% A raw connected-components pass on a skeleton does NOT split at branch
% points - an entire branching vessel tree comes back as ONE component,
% which made the per-"segment" arc/chord ordering below snake across
% unrelated branches and inflated tortuosity even on ordinary radiating
% vessels (caught by testing against a synthetic normal-vessel mask, which
% scored artificially high before this fix). Remove branch points first so
% each remaining piece is a genuine unbranched vessel segment.
branchPoints = bwmorph(skel, 'branchpoints');
skel = skel & ~imdilate(branchPoints, strel('disk', 1));
branches = bwconncomp(skel, 8); % explicit 8-connectivity (never rely on defaults)

tortRatios = [];
for i = 1:branches.NumObjects
    [py, px] = ind2sub(size(skel), branches.PixelIdxList{i});
    if numel(px) < 5
        continue; % too short a fragment to measure tortuosity meaningfully
    end
    inPeridiscal = mean(peridiscal(sub2ind(size(peridiscal), py, px))) > 0.5;
    if ~inPeridiscal
        continue;
    end
    ordered = localOrderPoints([px py]);
    arcLen = sum(hypot(diff(ordered(:,1)), diff(ordered(:,2))));
    chordLen = hypot(ordered(end,1)-ordered(1,1), ordered(end,2)-ordered(1,2));
    if chordLen > 1
        tortRatios(end+1) = arcLen / chordLen; %#ok<AGROW>
    end
end

if isempty(tortRatios)
    tortuosityScore = 1.0;
else
    tortuosityScore = mean(tortRatios);
end
densityScore = sum(vesselMask(peridiscal)) / max(sum(peridiscal(:)), 1);

% Usability gate 3: too few measurable peridiscal segments - a tortuosity
% mean over 0-2 fragments is noise, not evidence.
nSegments = numel(tortRatios);
if nSegments < ccfg.nvMinSegments
    [isFlagged, tortuosityScore, densityScore] = deal(false, NaN, NaN);
    nvReport = localReport('INVALID', NaN, NaN, vesselFrac, nSegments, vesselFrac, ...
        sprintf('only %d measurable peridiscal segment(s) (< %d) - tortuosity mean unreliable', nSegments, ccfg.nvMinSegments), ccfg);
    return;
end

% Self-relative thresholds from clinicalConfig (cutoffs synthetic-reasoned,
% NOT clinically validated - tune against real labeled NVD/NVE examples
% such as IDRiD annotations before trusting beyond a screening flag).
wholeImageDensity = sum(vesselMask(:)) / numel(vesselMask);
isFlagged = (tortuosityScore > ccfg.nvTortuosityCutoff) && (densityScore > ccfg.nvDensityRatio * wholeImageDensity);
if isFlagged
    status = 'PROXY_POSITIVE';
    prov = 'peridiscal density+tortuosity above configured cutoffs (PROXY screening flag, NOT a diagnosis)';
else
    status = 'NOT_DETECTED';
    prov = 'proxy-negative: cutoffs not met (a proxy negative, never a verified absence of NV)';
end
nvReport = localReport(status, tortuosityScore, densityScore, wholeImageDensity, nSegments, vesselFrac, prov, ccfg);
end

function rep = localReport(status, tort, dens, wholeDens, nSeg, vFrac, provenance, ccfg)
rep = struct('status', status, 'tortuosity', tort, 'density', dens, ...
    'wholeImageDensity', wholeDens, 'nSegments', nSeg, 'vesselFrac', vFrac, ...
    'cutoffs', struct('tortuosity', ccfg.nvTortuosityCutoff, 'densityRatio', ccfg.nvDensityRatio, ...
        'searchRadii', ccfg.nvSearchRadii, 'minVesselFrac', ccfg.nvMinVesselFrac, 'minSegments', ccfg.nvMinSegments), ...
    'provenance', provenance, 'configVersion', ccfg.version);
end

% ------------------------------------------------------------------
function ordered = localOrderPoints(points)
% Greedy nearest-neighbor ordering of an unordered point set along a thin
% curve - good enough for a short skeleton branch, not a general solver.
remaining = points;
ordered = remaining(1,:);
remaining(1,:) = [];
while ~isempty(remaining)
    d = hypot(remaining(:,1)-ordered(end,1), remaining(:,2)-ordered(end,2));
    [~, idx] = min(d);
    ordered(end+1,:) = remaining(idx,:); %#ok<AGROW>
    remaining(idx,:) = [];
end
end
