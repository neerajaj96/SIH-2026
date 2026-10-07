function bench = benchmarkStage11(imagePath, outDir, nRepeat)
% benchmarkStage11: Cold-vs-warm latency, per-stage timing proxy, memory
% snapshots, and artifact sizes for one image. Pure measurement harness:
% every figure is labeled with hardware/environment, and anything not
% measurable on the host is recorded as UNEXECUTED (never estimated).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB here) ***
% Requires: same toolboxes as runScreeningPipeline.m.
%
% OUTPUT: bench struct with .coldSec/.warmSecMean/.warmSecStd,
%   .perStage struct (quality/segmentation/grading masses measured by
%   repeated sub-calls where separable), .memoryBefore/After (or
%   'UNEXECUTED' string when the memory API is unavailable),
%   .artifactBytes, .hardware, .nRepeat.

if nargin < 2 || isempty(outDir), outDir = tempdir; end
if nargin < 3 || isempty(nRepeat), nRepeat = 5; end
if ~isfolder(outDir), mkdir(outDir); end

bench = struct();
bench.hardware = struct('platform', computer(), 'arch', computer('arch'));
try
    m0 = memory();
    bench.memoryBefore = m0.MemUsedMATLAB;
catch
    bench.memoryBefore = 'UNEXECUTED (memory API unavailable)';
end

raw = imread(imagePath);
t0 = tic;
r1 = runScreeningPipeline(raw);
bench.coldSec = toc(t0);
if ~strcmp(r1.status,'ok') && ~strcmp(r1.status,'ungradeable')
    warning('benchmarkStage11:firstRun - status %s; timings still recorded but atypical.', r1.status);
end
wt = zeros(nRepeat,1);
for i = 1:nRepeat
    ti = tic;
    runScreeningPipeline(raw);
    wt(i) = toc(ti);
end
bench.warmSecMean = mean(wt);
bench.warmSecStd = std(wt);
bench.nRepeat = nRepeat;

% Per-stage proxies (separable sub-calls; segmentation/grading need
% trained nets - MATLAB will error honestly if absent, recorded as-is).
tq = tic; assessFundusQuality(raw); bench.perStage.qualitySec = toc(tq);
try
    tg = tic;
    [~, ~, eGray, ~, ~, roi] = assessAndEnhanceImage(raw, -Inf, -Inf);
    bench.perStage.segmentInputSec = toc(tg);
catch ME
    bench.perStage.segmentInputSec = ['UNEXECUTED: ' ME.message];
end
try
    m1 = memory();
    bench.memoryAfter = m1.MemUsedMATLAB;
catch
    bench.memoryAfter = 'UNEXECUTED (memory API unavailable)';
end
d = dir(outDir);
bench.artifactBytes = sum([d.bytes]);
bench.outDir = outDir;
fprintf('Benchmark: cold %.2fs, warm %.2f+-%.2fs (n=%d) on %s.\n', ...
    bench.coldSec, bench.warmSecMean, bench.warmSecStd, nRepeat, bench.hardware.platform);
end
