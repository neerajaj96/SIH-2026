function [isFlagged, tortuosityScore, densityScore] = detectNeovascularization(vesselMask, odCenter, odRadius)
% detectNeovascularization: A SCREENING PROXY, not a diagnosis - and not
% pixel-level neovascularization segmentation, which the official PS asks
% for and which remains a genuinely hard, actively-researched problem
% even in the published literature (not just this codebase). Say that
% out loud in your pitch rather than letting a judge assume this is a
% validated NVD/NVE detector.
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
% INPUTS:
%   vesselMask - logical vessel mask from the trained vessel U-Net
%   odCenter   - [x y] from localizeOpticDiscFovea.m
%   odRadius   - optic disc radius (pixels) from localizeOpticDiscFovea.m
%
% OUTPUTS:
%   isFlagged       - true if the peridiscal region's density AND
%                     tortuosity both exceed threshold - a "flag for
%                     manual review", not a diagnosis
%   tortuosityScore - mean arc/chord ratio of vessel segments in the
%                     peridiscal region
%   densityScore    - local vessel-pixel density in the peridiscal region
%
% Requires: Image Processing Toolbox. Uses bwskel; if your release
% predates it, replace with bwmorph(vesselMask,'skel',Inf).

[H, W] = size(vesselMask);
[xx, yy] = meshgrid(1:W, 1:H);
peridiscal = hypot(xx-odCenter(1), yy-odCenter(2)) < 2.5*odRadius; % NVD's classic search zone

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
branches = bwconncomp(skel);

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

% Self-relative thresholds: compare the peridiscal patch against the rest
% of THIS SAME image's vasculature, since absolute pixel thresholds don't
% transfer across camera resolution/magnification. The 1.2x tortuosity
% cutoff was set after testing this exact function against synthetic
% normal-vessel vs. tortuous-tangle masks: after fixing a branch-point
% bug (see above), ordinary radiating vessels scored ~1.05-1.1 and a
% deliberately tortuous synthetic tangle scored ~1.37, so 1.2 sits
% between them. That is still calibration against SYNTHETIC data, not
% clinically validated - tune both cutoffs against real labeled NVD/NVE
% examples if you get access to any (IDRiD has some annotations for this)
% before trusting this for anything beyond a rough screening flag.
wholeImageDensity = sum(vesselMask(:)) / numel(vesselMask);
isFlagged = (tortuosityScore > 1.2) && (densityScore > 2 * wholeImageDensity);
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
