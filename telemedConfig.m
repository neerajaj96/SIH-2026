function cfg = telemedConfig()
% telemedConfig: SINGLE versioned source of truth for every tele-
% medicine simulation/queueing parameter. Replaces literals scattered
% across buildTelemedModel.m, SimEvents_Telemed_Model.m and
% optimizeResourceAllocation.m (which keep working defaults and read
% this file for anything new).
%
% PROVENANCE TAGS (every parameter carries one - never present a
% SCENARIO value as a field measurement):
%   'MEASURED'    - timed on this project's own stack
%   'ASSUMED'     - engineering placeholder, explicitly provisional
%   'SCENARIO'    - illustrative program-planning value (incl. ARDA)
%   'DATA-GATED'  - requires field measurement before use as evidence
%
% Units are stated per field and converted EXPLICITLY at use sites
% (bits vs bytes vs MB vs Mbps mismatches are a classic silent killer).
% Time base throughout: rates per HOUR, service times in SECONDS unless
% a field says otherwise; conversions use 3600 s/hr, 8 bits/byte.

cfg = struct();
cfg.version = '1.0.0-stage8';

% --- Demand (SCENARIO: district-program planning values) ---
cfg.annualPatients        = 100000;  % SCENARIO (PS target)
cfg.operatingHoursPerYear = 2000;    % SCENARIO (~250 days x 8h)
cfg.targetWaitMinutes     = 30;      % SCENARIO (review-wait SLA)

% --- Severity mix (SCENARIO, ARDA-sourced - NOT local measured
%     prevalence). Order: [NoDR Mild Moderate Severe PDR]; must sum to 1.
cfg.severityLabels = ["NoDR", "Mild", "Moderate", "Severe", "PDR"];
cfg.severityMix    = [0.827 0.032 0.104 0.009 0.028];
cfg.severityProvenance = 'ARDA-sourced scenario (~3,941 gradable images, India community screening); NOT locally measured prevalence';
cfg.referableFraction = 0.141;       % SCENARIO (Moderate+Severe+PDR of the mix above)
cfg.referableProvenance = 'derived from cfg.severityMix (0.104+0.009+0.028)';

% --- Image transmission (ASSUMED/DATA-GATED until field-measured) ---
cfg.imageMB            = 2.0;    % ASSUMED compressed fundus photo payload
cfg.bandwidthMbps      = 4.0;    % ASSUMED rural-link effective bandwidth
cfg.protocolOverheadFraction = 0.10; % ASSUMED TCP/TLS/retry overhead
% transmissionTimeSeconds = imageMB*8*(1+overhead)/bandwidthMbps
% (derived at use sites via telemedTransmissionSeconds, never inline).

% --- AI service (ASSUMED until Stage-6 runtime is benchmarked) ---
cfg.aiServiceTimeSeconds = 2.5;  % ASSUMED fixed pipeline time
cfg.aiWorkers            = 1;    % ASSUMED single AI worker
cfg.aiProvenance = 'ASSUMED/DATA-GATED: replace with measured Stage-6 runtime when benchmarked';

% --- Review service (SCENARIO pair: pitch claim vs conservative) ---
cfg.reviewSecondsOptimistic   = 30;   % SCENARIO (pitch claim)
cfg.reviewSecondsConservative = 120;  % SCENARIO (conservative planning)

% --- Queues ---
cfg.bandwidthQueueCapacity = 100;  % ASSUMED finite buffer (overflow blocks arrivals)

% --- Simulation control ---
cfg.meanArrivalGapMinutes = 6;     % SCENARIO (~10/hour camp pace)
cfg.simDurationMinutes    = 3000;  % SCENARIO (~500-patient surge horizon)
cfg.warmupMinutes         = 300;   % ASSUMED warm-up excluded from metrics
cfg.replications          = 10;    % ASSUMED replication count
cfg.seed                  = 42;    % seed for arrival/service RNG
cfg.consistencyTolerance  = 0.35;  % |sim-analytic|/analytic allowed (stochastic, not equality)
end
