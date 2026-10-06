function r = runScreeningPipeline(rawImage, models)
% runScreeningPipeline: THE single computation core for one fundus image
% - quality gate through clinical grading. Every other entry point
% (screenOneImage.m for the web/mobile bridge, runBatchScreening.m for
% folder batches, production_inference.m for the detailed single-case
% report) now calls THIS, instead of each having its own copy of the
% same ~40 lines. Caught during a scalability/upgradability review: three
% independently-maintained copies of the same pipeline steps is exactly
% how these quietly drift out of sync over time - this collapses them
% back to one.
%
% Returns the FULL intermediate state (masks, landmarks, the Grad-CAM
% score map, the enhanced image) rather than just a lean summary,
% because production_inference.m's detailed report needs the image data
% a JSON API response never would. Callers that only need the lean
% subset (screenOneImage.m) just pick the fields they want back out.
%
% INPUTS:
%   rawImage - fundus photo, as read by imread (not a file path - callers
%              that have a path should imread() it themselves, so this
%              function works identically whether the image came from
%              disk, an upload buffer, or anywhere else)
%   models   - (optional) pre-loaded network struct; defaults to
%              getOrLoadCachedModels() if omitted, which is itself cheap
%              to call repeatedly (persistent-cached, see that file)
%
% OUTPUT: r - struct with fields:
%   status, errorMessage          - 'ok' | 'ungradeable' | 'error'
%   focus, entropy, roiPassed     - quality gate
%   enhancedRGB, enhancedGray, roiMask
%   vesselMask, maheMask, exudateMask, lesionMask
%   odCenter, odRadius, foveaCenter, quadrantMask
%   nvFlagged, nvTortuosity, nvDensity
%   ruleGrade, evidence           - clinical rule engine (always computed,
%                                   even in simulated mode - see assignClinicalGrade.m)
%   haveTrainedModels
%   dlGrade, confidence, scoreMap, gradCamOnDisc  - NaN/empty if simulated
%
% Requires: same toolboxes as production_inference.m.

r = struct('status','ok','errorMessage','', 'focus',NaN,'entropy',NaN,'roiPassed',false, ...
    'enhancedRGB',[],'enhancedGray',[],'roiMask',[], ...
    'vesselMask',[],'maheMask',[],'exudateMask',[],'lesionMask',[], ...
    'odCenter',[],'odRadius',NaN,'foveaCenter',[],'quadrantMask',[], ...
    'nvFlagged',false,'nvTortuosity',NaN,'nvDensity',NaN, ...
    'ruleGrade',NaN,'evidence',{{}}, 'haveTrainedModels',false, ...
    'dlGrade',NaN,'confidence',NaN,'scoreMap',[],'gradCamOnDisc',false);

if nargin < 2 || isempty(models)
    models = getOrLoadCachedModels();
end
r.haveTrainedModels = models.haveTrainedModels;

try
    qcfg = qualityConfig();
    [isGradeable, enhancedRGB, enhancedGray, focus, ent, roiMask] = assessAndEnhanceImage(rawImage, qcfg.focusThresh, qcfg.entropyThresh);
    r.focus = focus; r.entropy = ent; r.roiPassed = isGradeable;
    r.enhancedRGB = enhancedRGB; r.enhancedGray = enhancedGray; r.roiMask = roiMask;
    if ~isGradeable
        r.status = 'ungradeable';
        return;
    end

    if r.haveTrainedModels
        vesselMask = runSegmentationNet(models.vesselNet, enhancedGray, roiMask);
        maheMask   = runSegmentationNet(models.maheNet, enhancedGray, roiMask);
        exudateMask = runSegmentationNet(models.exudateNet, enhancedGray, roiMask);
    else
        % Classical-image-processing placeholders so the plumbing runs end
        % to end without trained networks. NOT a substitute for the real
        % nets - and a real test run on a real photo found even the
        % placeholder was missing something basic: assessAndEnhanceImage.m
        % already ran CLAHE, which locally maximizes contrast, and running
        % a naive top-hat/threshold straight on top of THAT amplifies
        % ordinary pixel noise into thousands of single-pixel-scale
        % "blobs", which the clinical rule engine then dutifully counts as
        % thousands of "hemorrhages" per quadrant. A minimum-blob-size
        % filter is applied before these masks go anywhere downstream -
        % this does NOT make the placeholder a real detector, it just
        % stops single-pixel noise specks from being counted as phantom
        % lesions. The real fix is still a trained U-Net, not more
        % threshold tuning.
        minBlobAreaPx = 6;
        vesselMask = imbinarize(enhancedGray, 'adaptive') & roiMask;
        maheMask = bwareaopen((imtophat(imcomplement(enhancedGray), strel('disk', 8)) > 30) & roiMask, minBlobAreaPx);
        exudateMask = bwareaopen((enhancedGray > prctile(double(enhancedGray(roiMask)), 98)) & roiMask, minBlobAreaPx);
    end
    lesionMask = maheMask | exudateMask;
    r.vesselMask = vesselMask; r.maheMask = maheMask; r.exudateMask = exudateMask; r.lesionMask = lesionMask;

    [odCenter, odRadius, foveaCenter] = localizeOpticDiscFovea(rawImage, roiMask, vesselMask);
    quadrantMask = partitionQuadrants(size(roiMask), foveaCenter, odCenter);
    r.odCenter = odCenter; r.odRadius = odRadius; r.foveaCenter = foveaCenter; r.quadrantMask = quadrantMask;

    [nvFlagged, nvTort, nvDens] = detectNeovascularization(vesselMask, odCenter, odRadius);
    r.nvFlagged = nvFlagged; r.nvTortuosity = nvTort; r.nvDensity = nvDens;

    clinInfo = struct('maPresent', any(maheMask(:)), 'exudatePresent', any(exudateMask(:)), ...
        'venousBeadingQuadrants', 0, 'irmaQuadrants', 0, ...
        'neovascularization', nvFlagged, 'vitreousHemorrhage', false);
    [ruleGrade, evidence] = assignClinicalGrade(maheMask, quadrantMask, clinInfo);
    r.ruleGrade = ruleGrade; r.evidence = evidence;

    if r.haveTrainedModels
        % Stage-2 fix: masks are categorical - nearest-neighbor only.
        % The previous default (bicubic) invented fractional 0-255 values
        % the grader never saw as binary. Photos stay bilinear.
        fusionTensor = single(cat(3, imresize(enhancedRGB, [224 224], 'bilinear'), ...
            imresize(uint8(vesselMask)*255, [224 224], 'nearest'), imresize(uint8(lesionMask)*255, [224 224], 'nearest')));
        dlX = dlarray(fusionTensor, 'SSC');
        logits = predict(models.drNet, dlX, Outputs="dr_fc");
        probs = extractdata(softmax(logits ./ models.temperatureT));
        [conf, idx] = max(probs);
        r.dlGrade = idx - 1;
        r.confidence = conf;

        scoreMap = extractdata(gradCAM(models.drNet, dlX, idx, ReductionLayer="prob"));
        r.scoreMap = scoreMap;
        scoreMapFull = imresize(scoreMap, [size(enhancedRGB,1) size(enhancedRGB,2)]);
        [~, peakIdx] = max(scoreMapFull(:));
        [peakY, peakX] = ind2sub(size(scoreMapFull), peakIdx);
        r.gradCamOnDisc = hypot(peakX-odCenter(1), peakY-odCenter(2)) < 1.5*odRadius && r.dlGrade ~= 4;
    end
catch ME
    r.status = 'error';
    r.errorMessage = ME.message;
end
end
