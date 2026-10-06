function bench = benchmarkSegmentationInference(netOrMat, varargin)
% benchmarkSegmentationInference: Runtime + memory benchmark for the
% segmentation inference path (what Stage-3 and deployment actually pay).
%
% Measures runSegmentationNet wall time at the configured inputSize on
% synthetic enhancedGray/roiMask inputs (no data dependency), plus the
% resize-back cost at representative original resolutions (DRIVE 584x565,
% portable 1280x1280, IDRiD 4288x2848). Reports mean/std/min over repeats.
% Memory is reported as input/output tensor bytes (exact) plus a peak-hint
% via whos on the largest intermediate (documented estimate, not a
% profiler measurement - MATLAB memory profiler hooks differ by release).
%
% INPUTS:
%   netOrMat - trained dlnetwork OR path to unet_*.mat. If omitted or
%              missing, benchmarks the resize/normalize plumbing with a
%              stub predict (reports RESIZE-ONLY timing, clearly labeled -
%              never presented as network latency).
%   Name/Value:
%     'Repeats'    - default 20
%     'NetInputSize' - default segmentationConfig.inputSize
%
% OUTPUT: bench struct with .inputSize, .repeats, .haveNet, .timingsMs
%         (per orig-size), .tensorBytes
%
% Requires: Image Processing Toolbox; Deep Learning Toolbox if net given.

p = inputParser();
p.addParameter('Repeats', 20, @isnumeric);
p.addParameter('NetInputSize', [], @isnumeric);
p.parse(varargin{:});
opt = p.Results;
cfg = segmentationConfig();
if isempty(opt.NetInputSize), opt.NetInputSize = cfg.inputSize; end

haveNet = false; net = [];
if nargin >= 1 && ~isempty(netOrMat)
    try
        if ischar(netOrMat) || isstring(netOrMat)
            if isfile(char(netOrMat))
                S = load(char(netOrMat), 'net'); net = S.net; haveNet = true;
            end
        else
            net = netOrMat; haveNet = true;
        end
    catch
        haveNet = false;
    end
end

origSizes = {[584 565], [1280 1280], [4288 2848]};
origNames = {'DRIVE-like', 'portable-like', 'IDRiD-like'};
bench = struct('inputSize', opt.NetInputSize, 'repeats', opt.Repeats, ...
    'haveNet', haveNet, 'timingsMs', struct(), 'tensorBytes', struct());

% Tensor bytes (exact, no run needed).
H = opt.NetInputSize(1); W = opt.NetInputSize(2);
bench.tensorBytes.inputSingle = H * W * 4;       % single HxW
bench.tensorBytes.logitSingle = H * W * 2 * 4;   % 2-class float map
bench.tensorBytes.maskLogical = H * W * 1;       % logical HxW

for s = 1:numel(origSizes)
    sz = origSizes{s};
    % Synthetic fundus-like input: circle ROI on black (matches
    % assessAndEnhanceImage output layout: uint8 gray + logical ROI).
    enh = uint8(zeros(sz)); roi = false(sz);
    [xx, yy] = meshgrid(1:sz(2), 1:sz(1));
    roi = hypot(xx - sz(2)/2, yy - sz(1)/2) < min(sz)/2 * 0.95;
    enh(roi) = uint8(100 + 80 * rand(nnz(roi), 1));
    times = zeros(opt.Repeats, 1);
    for r = 1:opt.Repeats
        t0 = tic;
        imgResized = imresize(enh, opt.NetInputSize, 'bilinear');
        dlIn = single(imgResized) / 255; %#ok<NASGU> - normalize cost included
        if haveNet
            dlInA = dlarray(dlIn, 'SSC');
            dlOut = extractdata(predict(net, dlInA));
            [~, cI] = max(dlOut, [], 3);
            maskNet = squeeze(cI) == 2;
            maskOrig = imresize(maskNet, sz, 'nearest') & roi; %#ok<NASGU>
        else
            maskNet = imresize(roi, opt.NetInputSize, 'nearest'); %#ok<NASGU> - plumbing only
            maskOrig = imresize(maskNet, sz, 'nearest') & roi; %#ok<NASGU>
        end
        times(r) = toc(t0) * 1000;
    end
    bench.timingsMs.(matlab.lang.makeValidName(origNames{s})) = struct( ...
        'mean', mean(times), 'std', std(times), 'min', min(times), 'raw', times);
    fprintf('[%s %dx%d -> [%d %d]%s] mean %.1f ms std %.1f min %.1f (n=%d)\n', ...
        origNames{s}, sz(1), sz(2), H, W, ternary(haveNet, '', ' RESIZE-ONLY, no net'), ...
        mean(times), std(times), min(times), opt.Repeats);
end
if ~haveNet
    fprintf(['NOTE: no trained net found - timings above are resize/normalize plumbing only, NOT network latency. ' ...
             'Re-run with a unet_*.mat path once train_UNet_Segmentation.m has produced weights.\n']);
end
end

% ------------------------------------------------------------------
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
