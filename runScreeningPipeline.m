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
%   focus, entropy, roiPassed     - quality gate (roiPassed=false on FAIL)
%   qualityDecision               - 'PASS' | 'BORDERLINE' | 'FAIL' (never
%                                   silently coerced; BORDERLINE stays visible)
%   qualityReasons, qualityGuidance - cellstr from assessFundusQuality
%   focusThresh, entropyThresh     - EFFECTIVE thresholds used by the gate
%   qualityCalibrated, qualityCalibSource - calibration traceability
%   enhancedRGB, enhancedGray, roiMask
%   vesselMask, maheMask, exudateMask, lesionMask
%   odCenter, odRadius, foveaCenter, quadrantMask
%   nvFlagged, nvTortuosity, nvDensity
%   ruleGrade, evidence           - clinical rule engine (NaN + status
%                                   INSUFFICIENT_EVIDENCE when VB/IRMA/
%                                   vitreous evidence is UNAVAILABLE; see
%                                   assignClinicalGrade.m - never a
%                                   fabricated 0-4)
%   ruleStatus, ruleTrigger      - evidence sufficiency + fired trigger
%   haveTrainedModels
%   dlGrade, confidence, scoreMap, gradCamOnDisc  - NaN/empty if simulated
%   temperatureT, temperatureCalibrated, temperatureSource - the APPLIED
%     softmax temperature (from the model cache, hot-reloaded on artifact
%     mtime); NaN/false when simulated (no confidence computed)
%
% Requires: same toolboxes as production_inference.m.

r = struct('status','ok','errorMessage','', 'focus',NaN,'entropy',NaN,'roiPassed',false, ...
    'qualityDecision','FAIL','qualityReasons',{{}},'qualityGuidance',{{}}, ...
    'focusThresh',NaN,'entropyThresh',NaN,'qualityCalibrated',false,'qualityCalibSource','', ...
    'enhancedRGB',[],'enhancedGray',[],'roiMask',[], ...
    'vesselMask',[],'maheMask',[],'exudateMask',[],'lesionMask',[], ...
    'odCenter',[],'odRadius',NaN,'foveaCenter',[],'quadrantMask',[], ...
    'odValidity','UNRELIABLE','foveaValidity','UNRELIABLE','quadrantValid','INVALID','landmarkMethod','', ...
    'nvFlagged',false,'nvTortuosity',NaN,'nvDensity',NaN,'nvStatus','UNAVAILABLE', ...
    'ruleGrade',NaN,'evidence',{{}}, 'ruleStatus','INSUFFICIENT_EVIDENCE','ruleTrigger','unevaluated', ...
    'haveTrainedModels',false, ...
    'dlGrade',NaN,'confidence',NaN,'scoreMap',[],'gradCamOnDisc',false, ...
    'explainStatus','UNAVAILABLE','explainReasons',{{}},'explainProvenance','no trained models (simulated)', ...
    'explainChannelsCaveat','', ...
    'temperatureT',NaN,'temperatureCalibrated',false,'temperatureState','UNAVAILABLE','temperatureSource','simulated (no trained models)', ...
    'disagreement',struct('clinicalEvidenceStatus','INSUFFICIENT_EVIDENCE','gradeRelationship','NOT_COMPARABLE', ...
        'confidenceStatus','UNCALIBRATED','explanationStatus','UNAVAILABLE','escalate',true,'reasons',{{'simulated mode'}}));

if nargin < 2 || isempty(models)
    models = getOrLoadCachedModels();
end
r.haveTrainedModels = models.haveTrainedModels;

try
    % Canonical quality call (single core compute): calibration resolved
    % inside via pwd convention (same as loadModelsIfPresent.m); the
    % EFFECTIVE thresholds are reported back and reused for display, so
    % the gate and reporting can never disagree. FAIL hard-rejects;
    % BORDERLINE proceeds as gradeable-with-warning (decision + reasons
    % preserved, never coerced to PASS).
    qrep = assessFundusQuality(rawImage);
    r.focus = qrep.focusScore; r.entropy = qrep.entropyScore;
    r.roiPassed = qrep.isGradeable;
    r.qualityDecision = qrep.decision;
    r.qualityReasons = qrep.reasons;
    r.qualityGuidance = qrep.recaptureGuidance;
    r.focusThresh = qrep.focusThresh; r.entropyThresh = qrep.entropyThresh;
    r.qualityCalibrated = qrep.isCalibrated; r.qualityCalibSource = qrep.calibrationSource;
    r.enhancedRGB = qrep.enhancedRGB; r.enhancedGray = qrep.enhancedGray; r.roiMask = qrep.roiMask;
    enhancedRGB = qrep.enhancedRGB; enhancedGray = qrep.enhancedGray; roiMask = qrep.roiMask;
    if ~qrep.isGradeable
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

    [odCenter, odRadius, foveaCenter, lmStatus] = localizeOpticDiscFovea(rawImage, roiMask, vesselMask);
    [quadrantMask, qValid] = partitionQuadrants(size(roiMask), foveaCenter, odCenter, lmStatus.odValidity, lmStatus.foveaValidity);
    r.odCenter = odCenter; r.odRadius = odRadius; r.foveaCenter = foveaCenter; r.quadrantMask = quadrantMask;
    r.odValidity = lmStatus.odValidity; r.foveaValidity = lmStatus.foveaValidity;
    r.quadrantValid = qValid; r.landmarkMethod = [lmStatus.odMethod ' // ' lmStatus.foveaMethod];

    [nvFlagged, nvTort, nvDens, nvReport] = detectNeovascularization(vesselMask, odCenter, odRadius);
    r.nvFlagged = nvFlagged; r.nvTortuosity = nvTort; r.nvDensity = nvDens;
    r.nvStatus = nvReport.status;

    clinInfo = struct('maPresent', any(maheMask(:)), 'exudatePresent', any(exudateMask(:)), ...
        'venousBeadingQuadrants', 0, 'venousBeadingStatus', 'UNAVAILABLE', ...
        'irmaQuadrants', 0, 'irmaStatus', 'UNAVAILABLE', ...
        'neovascularization', nvFlagged, 'neovascularizationStatus', nvReport.status, ...
        'vitreousHemorrhage', false, 'vitreousStatus', 'UNAVAILABLE');
    % Provenance: the engine cannot verify mask origin pixel-wise, but the
    % pipeline knows which path produced the masks.
    if r.haveTrainedModels
        clinInfo.maskSource = 'trained';
    else
        clinInfo.maskSource = 'placeholder';
    end
    % VB/IRMA/vitreous carry NO status fields here: no detectors exist, so
    % the engine marks them UNAVAILABLE (never zero) and returns
    % INSUFFICIENT_EVIDENCE + NaN unless assessable evidence fires.
    % INVALID quadrant geometry blocks 4-2-1 reasoning (never silently
    % use frame-center geometry). NV screening still runs (it degrades to
    % INVALID itself on unusable geometry rather than false-negative).
    if strcmp(qValid, 'INVALID')
        r.ruleGrade = NaN;
        r.evidence = {'Quadrant geometry INVALID (landmarks unreliable and fallback not permitted) - 4-2-1 reasoning blocked.'};
        r.ruleStatus = 'INSUFFICIENT_EVIDENCE'; r.ruleTrigger = 'invalid-quadrants';
    else
        [ruleGrade, evidence, ruleReport] = assignClinicalGrade(maheMask, quadrantMask, clinInfo);
        r.ruleGrade = ruleGrade; r.evidence = evidence;
        r.ruleStatus = ruleReport.status; r.ruleTrigger = ruleReport.trigger;
    end

    if r.haveTrainedModels
        % Single canonical fusion implementation (same builder training
        % uses): photos bilinear, masks nearest, lesion = mahe|exudate,
        % loud validation on NaN/Inf, size mismatch, fractional masks.
        fusionTensor = buildGradingFusionTensor(enhancedRGB, vesselMask, maheMask, exudateMask);
        dlX = dlarray(fusionTensor, 'SSC');
        % Applied temperature comes from the model cache (hot-reloaded on
        % artifact mtime by getOrLoadCachedModels) and is reported back so
        % display can never show a different T than softmax used.
        % Calibrated-ness = artifact presence in the model dir (same pwd
        % convention as loadModelsIfPresent); a fitted T==1.5 still counts.
        r.temperatureT = models.temperatureT;
        r.temperatureState = models.calibrationState;
        r.temperatureCalibrated = strcmp(models.calibrationState, 'CALIBRATED_VALID');
        r.temperatureSource = 'model cache (getOrLoadCachedModels, hot-reloaded)';
        logits = predict(models.drNet, dlX, Outputs="dr_fc");
        probs = extractdata(softmax(logits ./ models.temperatureT));
        [conf, idx] = max(probs);
        r.dlGrade = idx - 1;
        r.confidence = conf;

        % Canonical explanation (ONLY Grad-CAM path): ROI-masked,
        % reliability-gated, lesion-contextualized. Explanation failure
        % never blocks the grade above.
        [gradMap, gradRep] = explainGradCAM(models.drNet, dlX, idx, roiMask, lesionMask, odCenter, odRadius);
        r.scoreMap = gradMap;
        r.explainStatus = gradRep.status;
        r.explainReasons = gradRep.reasons;
        r.explainProvenance = gradRep.provenance;
        r.explainChannelsCaveat = gradRep.channelsCaveat;
        % Legacy peak-on-disc flag preserved from the canonical peak
        % (post-ROI-mask, so border peaks cannot trigger it).
        if ~isempty(gradMap)
            r.gradCamOnDisc = gradRep.peakInDiscZone && r.dlGrade ~= 4;
        else
            r.gradCamOnDisc = false;
        end
    end
    % Structured disagreement (orthogonal dimensions; missing evidence is
    % NOT_COMPARABLE, never manufactured disagreement). Fallback T is
    % never "calibrated": only CALIBRATED_VALID counts.
    if strcmp(r.temperatureState, 'CALIBRATED_VALID')
        confState = 'CALIBRATED';
    else
        confState = 'UNCALIBRATED';
    end
    r.disagreement = analyzeDisagreement(r.dlGrade, r.confidence, confState, ...
        r.ruleGrade, r.ruleStatus, r.explainStatus);
catch ME
    r.status = 'error';
    r.errorMessage = ME.message;
end
end
