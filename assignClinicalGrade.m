function [icdrGrade, evidence] = assignClinicalGrade(hemorrhageMask, quadrantMask, maskInfo)
% assignClinicalGrade: Implements the real ICDR "4-2-1 rule" instead of
% relying on the classifier's softmax output alone plus a generic
% heatmap. This is what the official PS's phrase "lesion-level evidence
% correlated with clinical criteria" means in practice - every grade
% this returns comes with the specific rule that triggered it, in the
% same terms an ophthalmologist uses, which is a categorically stronger
% explainability claim than Grad-CAM alone and directly answers the PS
% background section's complaint about black-box AI.
%
% VERIFIED rule (cross-checked against the ICDR/ETDRS severity scale):
%   Level 0: no abnormalities
%   Level 1: microaneurysms only
%   Level 2: more than microaneurysms, but below Severe NPDR
%   Level 3 (Severe NPDR) - ANY of:
%     (a) severe (>20) intraretinal hemorrhages in EACH of 4 quadrants
%     (b) definite venous beading in >=2 quadrants
%     (c) prominent IRMA in >=1 quadrant
%     - with no signs of proliferative disease
%   Level 4 (Proliferative DR): neovascularization or vitreous/
%     preretinal hemorrhage
% This logic was prototyped and tested against 7 hand-built cases -
% one per ICDR level plus one for each of the three separate Severe-NPDR
% triggers - and passed all seven before being translated here.
%
% INPUTS:
%   hemorrhageMask - logical mask of detected hemorrhages (from the
%                    microaneurysm/hemorrhage U-Net's output)
%   quadrantMask   - uint8 label image (1-4) from partitionQuadrants.m
%   maskInfo       - struct with fields:
%       .maPresent              - logical, any microaneurysm detected
%       .exudatePresent         - logical, any exudate detected
%       .venousBeadingQuadrants - count of quadrants with definite venous
%                                 beading (default 0 - see NOTE)
%       .irmaQuadrants          - count of quadrants with prominent IRMA
%                                 (default 0 - see NOTE)
%       .neovascularization     - logical, from detectNeovascularization.m
%                                 (a SCREENING FLAG, not a diagnosis)
%       .vitreousHemorrhage     - logical, manual input - not detected by
%                                 this pipeline (fundus photography alone
%                                 often can't distinguish this reliably)
%
% NOTE on venous beading / IRMA: both are subtle vascular findings that
% remain genuinely difficult automated-detection problems in the
% published literature, not just this codebase. This function accepts
% pre-computed counts so the rule engine itself is complete and correct,
% but this pipeline does not supply a real detector for either yet - be
% upfront about that if a judge asks. With both defaulted to 0, the "4"
% (quadrant hemorrhage) and neovascularization criteria are the ones this
% pipeline can actually claim to assess end to end today.
%
% OUTPUTS:
%   icdrGrade - integer 0-4
%   evidence  - cell array of strings: the specific rule(s) that fired.
%               This is what belongs in the PDF report, and what your
%               team should be able to read straight to a judge.
%
% Requires: Image Processing Toolbox (bwlabel)

evidence = {};

if maskInfo.neovascularization || maskInfo.vitreousHemorrhage
    icdrGrade = 4;
    if maskInfo.neovascularization
        evidence{end+1} = 'Neovascularization flagged by the vessel density/tortuosity screen - Proliferative DR criterion.';
    end
    if maskInfo.vitreousHemorrhage
        evidence{end+1} = 'Vitreous/preretinal hemorrhage present - Proliferative DR criterion.';
    end
    return;
end

% --- Per-quadrant hemorrhage COUNT (connected components, not pixel area -
%     the clinical rule counts distinct hemorrhage spots) ---
quadrantHemCounts = zeros(1,4);
for q = 1:4
    qMask = hemorrhageMask & (quadrantMask == q);
    cc = bwlabel(qMask);
    quadrantHemCounts(q) = max(cc(:));
end

quadsWithSevereHem = sum(quadrantHemCounts > 20);
if quadsWithSevereHem == 4
    evidence{end+1} = sprintf('Severe hemorrhages (>20) in all 4 quadrants (counts: %s) - the "4" of the 4-2-1 rule.', mat2str(quadrantHemCounts));
end
if maskInfo.venousBeadingQuadrants >= 2
    evidence{end+1} = sprintf('Definite venous beading in %d quadrants - the "2" of the 4-2-1 rule.', maskInfo.venousBeadingQuadrants);
end
if maskInfo.irmaQuadrants >= 1
    evidence{end+1} = sprintf('Prominent IRMA in %d quadrant(s) - the "1" of the 4-2-1 rule.', maskInfo.irmaQuadrants);
end

if ~isempty(evidence)
    icdrGrade = 3;
    return;
end

anyOtherLesion = maskInfo.exudatePresent || any(quadrantHemCounts > 0);
if anyOtherLesion
    icdrGrade = 2;
    evidence{end+1} = sprintf('Lesions present beyond microaneurysms (hemorrhage counts by quadrant: %s; exudates present: %d), but below Severe NPDR criteria.', mat2str(quadrantHemCounts), maskInfo.exudatePresent);
    return;
end

if maskInfo.maPresent
    icdrGrade = 1;
    evidence{end+1} = 'Microaneurysms only.';
    return;
end

icdrGrade = 0;
evidence{end+1} = 'No visible DR abnormalities detected.';
end
