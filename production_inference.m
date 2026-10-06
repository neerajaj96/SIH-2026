% =========================================================================
% MODULE 4 (fixed + extended): Inference + Grad-CAM + calibrated
% confidence + rule-based clinical evidence + report
% Requires: Image Processing Toolbox, Deep Learning Toolbox.
% MATLAB Report Generator is used IF AVAILABLE for a PDF report - see the
% "FIXED THIS ROUND" note below for what happens if it isn't. Report
% Generator is NOT in the PS's own listed toolbox set, so don't assume a
% judge's or teammate's machine has it.
%
% FIXES from the first round:
%   - fusionTensor now uses the ENHANCED image from Module 1, not raw.
%   - Filename matches the actual sample.jpg, derived from this script's
%     own folder.
%   - Temperature scaling is real: pulls PRE-softmax logits by name and
%     divides by T, rather than a hardcoded calibratedConfidence.
%   - gradCAM call updated to the dlnetwork signature.
%   - "Simulated vs. real" is one explicit boolean, and the report is
%     visibly stamped with which mode produced it.
%   - The report actually embeds the enhanced image, lesion mask, and
%     Grad-CAM overlay.
%
% ADDED a round ago, closing gaps against the OFFICIAL problem statement
% text (not just the earlier paraphrased blueprint):
%   - Optic disc/fovea localization (localizeOpticDiscFovea.m) - one of
%     six Module-2 structures the PS names that nothing built so far touched.
%   - A quadrant partition (partitionQuadrants.m) needed for real
%     clinical-criteria evidence, not just a heatmap.
%   - A neovascularization SCREENING PROXY (detectNeovascularization.m) -
%     explicitly not a validated detector; say so if asked.
%   - assignClinicalGrade.m: the actual ICDR "4-2-1 rule" engine. This
%     runs as an INDEPENDENT SECOND OPINION alongside the deep-learning
%     classifier - both grades are reported, and if they disagree, that's
%     surfaced rather than hidden.
%
% FIXED LAST ROUND:
%   - Segmentation now goes through runSegmentationNet.m instead of the
%     old predictBinaryMask() local function - see that file's header.
%   - temperatureT loads a real fitted value from
%     calibrated_temperature.mat if calibrateTemperature.m has been run,
%     falling back to the old 1.5 placeholder otherwise - and the report
%     itself visibly says which one happened.
%
% FIXED THIS ROUND - the exact failure this project just hit for real:
%   MATLAB Report Generator isn't installed in every MATLAB environment
%   (it's an add-on, and it isn't in the PS's own toolbox list), so
%   add(rpt, ...) used to hard-error the WHOLE script after everything
%   upstream (quality gate, segmentation, landmarks, quadrants, clinical
%   grading) had already run correctly - throwing away real, computed
%   results because of a missing report-formatting add-on. The PDF path
%   is now wrapped so that if Report Generator isn't available (checked
%   up front AND caught if the build itself fails for any other reason),
%   this writes the exact same diagnostic content to a plain-text file
%   instead, plus the same supporting PNG images - so a missing add-on
%   degrades the OUTPUT FORMAT, not the whole pipeline. The images also
%   now save next to this script instead of into the OS temp folder,
%   since that's the actual deliverable when there's no PDF to embed them in.
% =========================================================================

% --- 0. Config ---
% NOTE: quality thresholds are NOT loaded here. The gate inside
% runScreeningPipeline.m resolves calibration (pwd convention) and reports
% the effective values back in r - display below reads r, so gate and
% report can never disagree. A second independent load here previously
% used scriptDir while the gate used pwd (two .mat files could disagree).
scriptDir = fileparts(mfilename('fullpath'));
imagePath = fullfile(scriptDir, 'sample.jpg');

% NOTE: temperature is NOT loaded here either. The applied T comes back
% in r (r.temperatureT / r.temperatureCalibrated, resolved by the model
% cache from the same pwd-convention artifact softmax actually used), so
% displayed T can never differ from applied T. A separate scriptDir load
% here previously risked showing one file while softmax used another.
scriptDir = fileparts(mfilename('fullpath'));
imagePath = fullfile(scriptDir, 'sample.jpg');

haveTrainedModels = exist('unet_Vessels.mat','file') && ...
                     exist('unet_MicroaneurysmsHemorrhages.mat','file') && ...
                     exist('unet_Exudates.mat','file') && ...
                     exist('trained_dr_grader.mat','file');

% Checked up front so the run can tell you what kind of report to expect
% BEFORE it gets all the way to the end - and checked again around the
% actual build (see Section 5) in case the license test passes but the
% checkout still fails for some environment-specific reason.
haveReportGen = license('test', 'MATLAB_Report_Gen');
if ~haveReportGen
    fprintf(['NOTE: MATLAB Report Generator not detected - this run will produce a plain-text ' ...
             'report plus PNG images next to this script instead of a PDF. Install Report Generator ' ...
             '(Add-On Explorer) if you specifically need a PDF, but nothing else in this pipeline needs it.\n']);
end

disp('Initializing Full Clinical Pipeline...');
if ~haveTrainedModels
    warning(['No trained model files found next to this script. The DEEP-LEARNING ' ...
             'grade below is SIMULATED - a placeholder, not a real prediction. The ' ...
             'rule-based clinical-evidence grade further down is a genuine computation ' ...
             'even in this mode, but only as good as the placeholder segmentation masks ' ...
             'feeding it - it is not a clinically valid answer either until real trained ' ...
             'segmentation networks are in place. Both are labeled accordingly in the report.']);
end

% --- 1-4. Quality gate, segmentation, landmarks, clinical rule grade,
% and deep-learning grading - all delegated to runScreeningPipeline.m,
% the one shared computation core every entry point now uses (was ~115
% lines duplicated here directly; see that file's header for why). ---
rawImage = imread(imagePath);
r = runScreeningPipeline(rawImage);

% Single source: effective gate thresholds come from the pipeline report
% itself (which resolved calibration), never from a second independent
% load that could disagree with the gate.
focus = r.focus; ent = r.entropy;
focusThresh = r.focusThresh; entropyThresh = r.entropyThresh;
if strcmp(r.status, 'ungradeable')
    error(['Image rejected by quality check (decision=%s, focus=%.1f < %.1f, or entropy=%.2f < %.2f). ' ...
           'Recapture required, or recalibrate thresholds with calibrateQualityThresholds.m.'], ...
           r.qualityDecision, focus, focusThresh, ent, entropyThresh);
elseif strcmp(r.status, 'error')
    error('runScreeningPipeline:failed', 'Pipeline error: %s', r.errorMessage);
end

enhancedRGB = r.enhancedRGB; enhancedGray = r.enhancedGray; roiMask = r.roiMask;
vesselMask = r.vesselMask; maheMask = r.maheMask; exudateMask = r.exudateMask; lesionMask = r.lesionMask;
odCenter = r.odCenter; odRadius = r.odRadius; foveaCenter = r.foveaCenter; quadrantMask = r.quadrantMask;
nvFlagged = r.nvFlagged; nvTortuosity = r.nvTortuosity; nvDensity = r.nvDensity;
clinicalGrade = r.ruleGrade; clinicalEvidence = r.evidence;
haveTrainedModels = r.haveTrainedModels;
% Applied temperature: the SAME value softmax divided by (reported by the
% pipeline, hot-reloaded by the model cache). Displayed T below is this
% value - never a second independent file read.
temperatureT = r.temperatureT;
haveCalibratedTemperature = r.temperatureCalibrated;

classNames = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"];
if haveTrainedModels
    predictedClass = r.dlGrade;
    predictedClassIdx = r.dlGrade + 1;
    calibratedConfidence = r.confidence;
    scoreMap = r.scoreMap;
    gradCamOnDisc = r.gradCamOnDisc;

    % Grad-CAM sanity check and DL/rule-engine disagreement warnings -
    % computed inside runScreeningPipeline.m, surfaced here at the
    % console the same way this script always has.
    if gradCamOnDisc
        warning(['Grad-CAM''s peak attention falls on/near the optic disc for a Level %d call. The optic ' ...
                 'disc is a normal bright structure - a model keying on it rather than actual lesions is a ' ...
                 'warning sign the prediction may be spurious, not evidence-based. Treat this case with extra scrutiny.'], ...
                predictedClass);
    end
    if predictedClass ~= clinicalGrade
        warning(['Deep-learning grade (Level %d) and rule-based clinical grade (Level %d) DISAGREE. ' ...
                 'This is exactly the kind of case the original design doc''s "flag as spurious" idea ' ...
                 'was meant to catch - have a human look at this one first.'], predictedClass, clinicalGrade);
    end
else
    predictedClass = NaN;
    predictedClassIdx = 1;
    calibratedConfidence = NaN;
    scoreMap = zeros(size(enhancedGray));
    gradCamOnDisc = false;
end

% --- 5. Generate supporting images (need Image Processing Toolbox / base
% graphics only - NOT Report Generator), then a report in whichever
% format the environment actually supports. ---
disp('Rendering supporting images...');
landmarkPath = fullfile(scriptDir, 'dr_report_landmarks.png');
fig = figure('Visible','off');
imshow(enhancedRGB); hold on;
plot(odCenter(1), odCenter(2), 'yo', 'MarkerSize', 15, 'LineWidth', 2);
plot(foveaCenter(1), foveaCenter(2), 'co', 'MarkerSize', 15, 'LineWidth', 2);
legend('Optic disc', 'Fovea', 'TextColor', 'white', 'Location', 'southoutside');
hold off;
exportgraphics(fig, landmarkPath);
close(fig);

lesionPath = fullfile(scriptDir, 'dr_report_lesion_mask.png');
imwrite(lesionMask, lesionPath);

gradcamPath = '';
if haveTrainedModels
    gradcamPath = fullfile(scriptDir, 'dr_report_gradcam.png');
    fig = figure('Visible','off');
    imshow(enhancedRGB); hold on;
    imagesc(imresize(scoreMap, [size(enhancedRGB,1) size(enhancedRGB,2)]), 'AlphaData', 0.5);
    colormap(fig, 'jet'); hold off;
    exportgraphics(fig, gradcamPath);
    close(fig);
end

reportProducedAsPdf = false;
if haveReportGen
    try
        disp('Generating PDF Report...');
        import mlreportgen.report.*
        import mlreportgen.dom.*

        rpt = Report(fullfile(scriptDir, 'DR_Clinical_Report'), 'pdf');

        titlePg = TitlePage;
        titlePg.Title = 'Automated Diabetic Retinopathy Assessment';
        titlePg.Author = 'AI Screening Node #42';
        if ~haveTrainedModels
            titlePg.Subtitle = 'DEEP-LEARNING GRADE IS SIMULATED - no trained model found. Rule-based grade below uses placeholder segmentation.';
        end
        add(rpt, titlePg);

        add(rpt, Chapter('Title', 'Diagnostic Summary'));
        if haveTrainedModels
            add(rpt, Paragraph(sprintf('Deep-learning ICDR grade: Level %d (%s)', predictedClass, classNames(predictedClassIdx))));
            add(rpt, Paragraph(sprintf('Calibrated confidence: %.1f%%  (temperature T = %.2f%s)', calibratedConfidence * 100, temperatureT, ...
                ternary(haveCalibratedTemperature, '', ' - UNCALIBRATED PLACEHOLDER, see calibrateTemperature.m'))));
        else
            add(rpt, Paragraph('Deep-learning grade: SIMULATED (no trained model found) - not a real prediction.'));
        end
        add(rpt, Paragraph(sprintf('Quality gate: %s - focus=%.1f (threshold %.1f), entropy=%.2f (threshold %.2f)%s', ...
            r.qualityDecision, focus, focusThresh, ent, entropyThresh, ternary(r.qualityCalibrated, '', ' - UNCALIBRATED PLACEHOLDER, see calibrateQualityThresholds.m'))));

        add(rpt, Chapter('Title', 'Rule-Based Clinical Grade (ICDR 4-2-1 criteria)'));
        add(rpt, Paragraph(sprintf('Clinical-criteria grade: Level %d (%s)', clinicalGrade, classNames(clinicalGrade+1))));
        for i = 1:numel(clinicalEvidence)
            add(rpt, Paragraph(['- ' clinicalEvidence{i}]));
        end
        if ~haveTrainedModels
            add(rpt, Paragraph('(Based on placeholder classical-image-processing masks, not trained segmentation networks - treat as a pipeline test, not a real grade.)'));
        end
        add(rpt, Paragraph(sprintf('Neovascularization screen: %s (tortuosity=%.2f, density=%.4f - see detectNeovascularization.m; this is a screening flag, not a diagnosis)', ...
            ternary(nvFlagged, 'FLAGGED for review', 'not flagged'), nvTortuosity, nvDensity)));
        if haveTrainedModels && predictedClass ~= clinicalGrade
            add(rpt, Paragraph(sprintf('NOTE: deep-learning grade (Level %d) and rule-based grade (Level %d) disagree - recommend manual review.', predictedClass, clinicalGrade)));
        end
        if haveTrainedModels && gradCamOnDisc
            add(rpt, Paragraph('NOTE: Grad-CAM attention falls on/near the optic disc rather than a lesion site - possible spurious prediction, recommend manual review.'));
        end

        add(rpt, Chapter('Title','Enhanced Image (fused into the classifier - not the raw upload) with Landmarks'));
        add(rpt, Image(landmarkPath));

        add(rpt, Chapter('Title','Detected Lesion Mask'));
        add(rpt, Image(lesionPath));

        if haveTrainedModels
            add(rpt, Chapter('Title','Grad-CAM Overlay'));
            add(rpt, Image(gradcamPath));
        end

        close(rpt);
        rptview(rpt);
        reportProducedAsPdf = true;
    catch reportErr
        warning(['PDF report generation failed even though a Report Generator license test passed ' ...
                 '(%s). Falling back to a plain-text report - the diagnostic results themselves are ' ...
                 'unaffected by this.'], reportErr.message);
    end
end

if ~reportProducedAsPdf
    disp('Generating plain-text report (no PDF available)...');
    txtPath = fullfile(scriptDir, 'DR_Clinical_Report_summary.txt');
    fid = fopen(txtPath, 'w');
    fprintf(fid, 'AUTOMATED DIABETIC RETINOPATHY ASSESSMENT\n');
    fprintf(fid, 'AI Screening Node #42 - generated %s\n', string(datetime('now')));
    fprintf(fid, '==========================================\n\n');
    if ~haveTrainedModels
        fprintf(fid, '*** DEEP-LEARNING GRADE IS SIMULATED - no trained model found. ***\n');
        fprintf(fid, '*** Rule-based grade below uses placeholder segmentation. ***\n\n');
    end
    fprintf(fid, '-- DIAGNOSTIC SUMMARY --\n');
    if haveTrainedModels
        fprintf(fid, 'Deep-learning ICDR grade: Level %d (%s)\n', predictedClass, classNames(predictedClassIdx));
        fprintf(fid, 'Calibrated confidence: %.1f%%  (temperature T = %.2f%s)\n', calibratedConfidence * 100, temperatureT, ...
            ternary(haveCalibratedTemperature, '', ' - UNCALIBRATED PLACEHOLDER, see calibrateTemperature.m'));
    else
        fprintf(fid, 'Deep-learning grade: SIMULATED (no trained model found) - not a real prediction.\n');
    end
    fprintf(fid, 'Quality gate: %s - focus=%.1f (threshold %.1f), entropy=%.2f (threshold %.2f)%s\n\n', ...
        r.qualityDecision, focus, focusThresh, ent, entropyThresh, ternary(r.qualityCalibrated, '', ' - UNCALIBRATED PLACEHOLDER, see calibrateQualityThresholds.m'));

    fprintf(fid, '-- RULE-BASED CLINICAL GRADE (ICDR 4-2-1 criteria) --\n');
    fprintf(fid, 'Clinical-criteria grade: Level %d (%s)\n', clinicalGrade, classNames(clinicalGrade+1));
    for i = 1:numel(clinicalEvidence)
        fprintf(fid, '  - %s\n', clinicalEvidence{i});
    end
    if ~haveTrainedModels
        fprintf(fid, '(Based on placeholder classical-image-processing masks, not trained segmentation networks - treat as a pipeline test, not a real grade.)\n');
    end
    fprintf(fid, 'Neovascularization screen: %s (tortuosity=%.2f, density=%.4f - screening flag, not a diagnosis)\n', ...
        ternary(nvFlagged, 'FLAGGED for review', 'not flagged'), nvTortuosity, nvDensity);
    if haveTrainedModels && predictedClass ~= clinicalGrade
        fprintf(fid, 'NOTE: deep-learning grade (Level %d) and rule-based grade (Level %d) disagree - recommend manual review.\n', predictedClass, clinicalGrade);
    end
    if haveTrainedModels && gradCamOnDisc
        fprintf(fid, 'NOTE: Grad-CAM attention falls on/near the optic disc rather than a lesion site - possible spurious prediction, recommend manual review.\n');
    end

    fprintf(fid, '\n-- SUPPORTING IMAGES (saved alongside this file) --\n');
    fprintf(fid, 'Enhanced image with optic disc / fovea landmarks: %s\n', landmarkPath);
    fprintf(fid, 'Detected lesion mask: %s\n', lesionPath);
    if haveTrainedModels
        fprintf(fid, 'Grad-CAM overlay: %s\n', gradcamPath);
    end
    fclose(fid);
    fprintf('Wrote %s\n', txtPath);
end

disp('Pipeline complete.');
if ~haveTrainedModels
    disp(['Reminder: the deep-learning grade above was SIMULATED. Train the networks in ' ...
          'train_UNet_Segmentation.m and train_DR_Grader.m, save them as unet_Vessels.mat / ' ...
          'unet_MicroaneurysmsHemorrhages.mat / unet_Exudates.mat / trained_dr_grader.mat next ' ...
          'to this script, and re-run for real output on both grading pathways.']);
end
if haveTrainedModels && ~haveCalibratedTemperature
    disp(['Reminder: temperature T above was the UNCALIBRATED placeholder (1.5). Run ' ...
          'calibrateTemperature.m on held-out validation logits/labels and re-run this script ' ...
          'so it picks up calibrated_temperature.mat automatically.']);
end
if ~reportProducedAsPdf
    disp('Reminder: report was written as plain text, not PDF - install MATLAB Report Generator (Add-On Explorer) if you specifically need a PDF.');
end

% ------------------------------------------------------------------
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end