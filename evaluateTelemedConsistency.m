function results = evaluateTelemedConsistency(cfg, reviewers, varargin)
% evaluateTelemedConsistency: Analytic-vs-simulation consistency check
% for the reviewer queue (the M/M/c-plausible stage).
%
% METHODOLOGY (no fabricated agreement): analytic Erlang-C mean review
% wait (same arrival/service assumptions as the sim) vs the mean of
% per-replication sim means (warm-up excluded), with a 95% t-interval
% over replications. Verdict PASS iff |sim-analytic|/analytic <=
% cfg.consistencyTolerance. Stochastic equality is NOT expected; the
% tolerance + interval ARE the claim.
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB/SimEvents here) ***
% Requires: Simulink, SimEvents.
%
% INPUTS:
%   cfg       - telemedConfig struct (arrival gap, review seconds,
%               replications, warmup, seed, tolerance)
%   reviewers - ophthalmologist count for this check (analytic M/M/c;
%               NOTE the SimEvents review block is a SINGLE server - set
%               reviewers=1 here unless the topology gains parallel
%               review servers, else the comparison is meaningless)
%   'SimMinutes' - horizon override (default cfg.simDurationMinutes)
%
% OUTPUT: results struct(scenario, parameters, analyticWaitMin,
%   simMeanMin, simCI, relGap, tolerance, verdict, stability,
%   bottleneck, assumptionStatus, simulationMetadata).
%
% Bottleneck rule: compare utilization proxies across TRANSMISSION
% (txSeconds vs gap), AI (aiTime vs gap), REVIEW (analytic rho) and
% report the max as dominant - evidence-based, never a single snapshot.

p = inputParser;
addParameter(p, 'SimMinutes', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
parse(p, varargin{:});
if nargin < 1 || isempty(cfg), cfg = telemedConfig(); end
if nargin < 2 || isempty(reviewers), reviewers = 1; end
simMinutes = p.Results.SimMinutes;
if isempty(simMinutes), simMinutes = cfg.simDurationMinutes; end

% --- Analytic expectation (M/M/c on the REVIEW stage) ---
lambdaPerHour = 60 / cfg.meanArrivalGapMinutes * sum(cfg.severityMix(3:5));
muPerHour = 3600 / cfg.reviewSecondsOptimistic;
rho = (lambdaPerHour / muPerHour) / reviewers;
analyticMin = erlangCWaitHours(lambdaPerHour, muPerHour, reviewers) * 60;
stable = isfinite(analyticMin) && rho < 1;

% --- Bottleneck evidence (utilization proxies, same assumptions) ---
txSec = telemedTransmissionSeconds(cfg);
uTx = (txSec/60) / cfg.meanArrivalGapMinutes;
uAI = (cfg.aiServiceTimeSeconds/60) / cfg.meanArrivalGapMinutes;
[~, bottleneck] = max([uTx, uAI, rho]);
bottleneck = {'TRANSMISSION', 'AI', 'OPHTHALMOLOGIST_REVIEW'}{bottleneck};

% --- Simulation replications (seeded, warm-up excluded) ---
repMeans = NaN(cfg.replications, 1);
for r = 1:cfg.replications
    rng(cfg.seed + r);
    mdl = buildTelemedModel(sprintf('DR_Consistency_R%d', r), ...
        cfg.meanArrivalGapMinutes, cfg.aiServiceTimeSeconds, cfg.reviewSecondsOptimistic, cfg);
    sim(mdl, simMinutes);
    w = evalin('base', 'reviewWaitLog');
    t = w.Time; v = w.Data;
    keep = t >= cfg.warmupMinutes;
    repMeans(r) = mean(v(keep));
    close_system(mdl, 0);
end
simMean = mean(repMeans);
simStd = std(repMeans);
simCI = simStd / sqrt(cfg.replications) * tinv(0.975, cfg.replications-1); % 95% t-interval over replications
if stable
    relGap = abs(simMean - analyticMin) / max(analyticMin, eps);
    if relGap <= cfg.consistencyTolerance
        verdict = 'PASS';
    else
        verdict = 'DIVERGED';
    end
else
    relGap = NaN;
    verdict = 'UNSTABLE (analytic Inf - sim comparison meaningless)';
end

results = struct('scenario', 'reviewer-queue consistency', ...
    'parameters', cfg, 'arrivalRate', lambdaPerHour, ...
    'serviceRates', struct('reviewPerHour', muPerHour), ...
    'servers', reviewers, 'utilization', rho, ...
    'queueMetrics', struct('analyticWaitMin', analyticMin, 'simMeanMin', simMean, 'simCI', simCI), ...
    'waitingTimes', struct('analyticMin', analyticMin, 'simMin', simMean), ...
    'throughput', lambdaPerHour, 'stability', stable, ...
    'bottleneck', bottleneck, ...
    'optimizerResult', struct(), ...
    'simulationMetadata', struct('seed', cfg.seed, 'replications', cfg.replications, ...
        'horizonMin', simMinutes, 'warmupMin', cfg.warmupMinutes), ...
    'analyticComparison', struct('relGap', relGap, 'tolerance', cfg.consistencyTolerance, 'verdict', verdict), ...
    'assumptionStatus', 'SCENARIO inputs; methodology executable only in MATLAB/SimEvents');
fprintf('Analytic review wait: %.2f min | sim mean: %.2f +- %.2f min (n=%d) | gap %.1f%% (tol %.0f%%) => %s\n', ...
    analyticMin, simMean, simCI, cfg.replications, 100*relGap, 100*cfg.consistencyTolerance, verdict);
end
