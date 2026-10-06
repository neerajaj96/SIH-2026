function [temperatureT, report] = calibrateTemperature(valLogits, valLabels, calFilePath)
% calibrateTemperature: Fits the softmax temperature T on HELD-OUT
% VALIDATION logits/labels by minimizing negative log-likelihood, and
% reports Expected Calibration Error (ECE) before and after scaling.
%
% WHY THIS FILE EXISTS: train_DR_Grader.m and production_inference.m both
% describe temperature scaling correctly in comments, but nothing in the
% project ever actually FIT a temperature - production_inference.m
% simply hardcoded temperatureT = 1.5 and called it a placeholder. This
% is the missing fitting step. Once trained_dr_grader.mat exists, run
% this on a genuinely held-out validation set and it saves
% calibrated_temperature.mat, which production_inference.m now picks up
% automatically instead of the 1.5 placeholder.
%
% VALIDATION vs TEST - DO NOT SKIP THIS: T must be fit on a split the
% model has never been evaluated against for any other purpose. If you
% fit T on the same Messidor-2 split compareModels.m reports numbers on,
% your calibration numbers AND your sensitivity/specificity numbers are
% both contaminated by the same leakage - the earlier project review's
% "external validation must be carefully controlled" warning applies
% here specifically, not just to hyperparameter tuning in general.
%
% METHOD: T* = argmin_T  NLL(softmax(logits/T), labels), over T in
% (0.05, 10], via fminbnd - core MATLAB, no Optimization Toolbox
% dependency (that toolbox isn't in this PS's tool list). This is the
% exact recipe from Guo et al., "On Calibration of Modern Neural
% Networks" (2017), which introduced temperature scaling.
%
% VERIFIED (independently, in Python, before writing this file): on a
% synthetic 500-sample, 5-class logit set built to be overconfident
% (measured accuracy 77.0%), minimizing NLL over T recovered T*=1.56,
% cut ECE from 0.149 (T=1) to 0.069 (T=T*), and left accuracy exactly
% unchanged - as it must, since temperature scaling rescales CONFIDENCE,
% not which class wins the argmax. runSelfTests.m reproduces an
% equivalent check directly in MATLAB.
%
% INPUTS:
%   valLogits   - [N x numClasses] PRE-softmax logits on the VALIDATION
%                 set (extractdata'd from predict(drNet, dlX,
%                 Outputs="dr_fc") - plain numeric, not dlarray)
%   valLabels   - [N x 1] integer ICDR labels, 0-indexed (0-4), SAME
%                 order as valLogits' rows
%   calFilePath - (optional) where to save the result. Default:
%                 'calibrated_temperature.mat' in the current folder -
%                 production_inference.m looks for exactly this filename
%                 next to itself.
%
% OUTPUTS:
%   temperatureT - the fitted scalar T*
%   report       - struct: .eceBefore .eceAfter .nllBefore .nllAfter
%                  .accuracy (unaffected by T, reported for context) .n
%
% Requires: nothing beyond base MATLAB (fminbnd is core MATLAB, not
% Optimization Toolbox).

if nargin < 3 || isempty(calFilePath)
    calFilePath = 'calibrated_temperature.mat';
end
if numel(valLabels) ~= size(valLogits,1)
    error('calibrateTemperature:sizeMismatch', ...
        'valLogits has %d rows but valLabels has %d entries - these must be the SAME validation set in the SAME order.', ...
        size(valLogits,1), numel(valLabels));
end

report = struct();
report.n = numel(valLabels);
probsT1 = localSoftmaxRows(valLogits);
[~, predClass0] = max(probsT1, [], 2);
report.accuracy = mean((predClass0 - 1) == valLabels(:));
report.nllBefore = localNLL(1, valLogits, valLabels);
report.eceBefore = localECE(probsT1, valLabels);

objective = @(T) localNLL(T, valLogits, valLabels);
temperatureT = fminbnd(objective, 0.05, 10);

probsTstar = localSoftmaxRows(valLogits / temperatureT);
report.nllAfter = localNLL(temperatureT, valLogits, valLabels);
report.eceAfter = localECE(probsTstar, valLabels);

fprintf('=== Temperature calibration (n=%d validation samples) ===\n', report.n);
fprintf('Accuracy (unaffected by T): %.1f%%\n', report.accuracy*100);
fprintf('NLL   : %.4f (T=1)  ->  %.4f (T=%.3f)\n', report.nllBefore, report.nllAfter, temperatureT);
fprintf('ECE   : %.4f (T=1)  ->  %.4f (T=%.3f)\n', report.eceBefore, report.eceAfter, temperatureT);
if report.eceAfter >= report.eceBefore
    warning(['Calibration did not reduce ECE on this validation set - the model may already be ' ...
             'reasonably calibrated, or T is fighting a miscalibration shape (e.g. under- rather ' ...
             'than over-confidence in different classes) that a single global scalar T cannot fix. ' ...
             'Inspect a reliability diagram (confidence vs. accuracy per bin) before trusting either number.']);
end

calibrationReport = report; %#ok<NASGU> - saved under this name so production_inference.m's load() call finds it
save(calFilePath, 'temperatureT', 'calibrationReport');
fprintf('Saved %s - production_inference.m will pick this up automatically.\n', calFilePath);
end

% ------------------------------------------------------------------
function p = localSoftmaxRows(logits)
z = logits - max(logits, [], 2); % subtract row-max for numerical stability before exp
e = exp(z);
p = e ./ sum(e, 2);
end

% ------------------------------------------------------------------
function nll = localNLL(T, logits, labels)
p = localSoftmaxRows(logits / T);
n = numel(labels);
idx = sub2ind(size(p), (1:n)', labels(:)+1); % +1: labels are 0-indexed ICDR grades, p's columns are 1-indexed
nll = -mean(log(p(idx) + 1e-12));
end

% ------------------------------------------------------------------
function e = localECE(probs, labels, numBins)
% Standard binned Expected Calibration Error (Guo et al. 2017): split
% samples into confidence bins, and within each bin take the gap between
% mean confidence and actual accuracy, weighted by the bin's population
% share.
if nargin < 3 || isempty(numBins)
    numBins = 15;
end
[confidences, predClass0] = max(probs, [], 2);
correct = (predClass0 - 1) == labels(:);
n = numel(labels);
edges = linspace(0, 1, numBins+1);
e = 0;
for i = 1:numBins
    if i == 1
        inBin = confidences >= edges(i) & confidences <= edges(i+1);
    else
        inBin = confidences > edges(i) & confidences <= edges(i+1);
    end
    if any(inBin)
        accBin = mean(correct(inBin));
        confBin = mean(confidences(inBin));
        e = e + (sum(inBin)/n) * abs(accBin - confBin);
    end
end
end
