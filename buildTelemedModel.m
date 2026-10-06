function modelName = buildTelemedModel(modelName, meanArrivalGapMinutes, aiServiceTimeSeconds, reviewServiceTimeSeconds, cfg)
% buildTelemedModel: Builds the SimEvents tele-ophthalmology queuing model.
% Extracted out of SimEvents_Telemed_Model.m into its own function so
% optimizeResourceAllocation.m can rebuild the same validated topology
% with different parameters for a sweep, instead of a second copy-pasted
% model-construction block drifting out of sync with this one.
%
% Requires: Simulink, SimEvents (see the CAUTION note in
% SimEvents_Telemed_Model.m about version-sensitive block paths/parameter
% names - it applies here identically, since this is the same topology).
%
% INPUTS (positional legacy form preserved; pass a telemedConfig struct
% as 5th arg or [] for defaults - config values fill any [] input):
%   modelName             - string, a fresh model name (closed/overwritten if already loaded)
%   meanArrivalGapMinutes - mean minutes between patient arrivals
%   aiServiceTimeSeconds  - fixed AI pipeline processing time, seconds
%   reviewServiceTimeSeconds - mean ophthalmologist review time, seconds
%                              (std dev fixed at reviewServiceTimeSeconds/6,
%                               i.e. roughly a 5-second spread around a 30s mean)
%   cfg                   - (optional) telemedConfig struct: severity mix
%                           (+provenance validation), transmission
%                           (imageMB/bandwidthMbps/overhead -> derived
%                           service delay on a dedicated Transmission
%                           server), queue capacity, seed.
%
% OUTPUT:
%   modelName - echoed back, for chaining into sim(modelName, ...)
%
% SEED CONTRACT: this builder intentionally does NOT call rng() (no hidden
% global side effects). Replication drivers (see evaluateTelemedConsistency.m)
% must rng(cfg.seed + replicationIndex) before each sim() for reproducible
% replications.

if nargin < 5 || isempty(cfg), cfg = telemedConfig(); end
if isempty(meanArrivalGapMinutes), meanArrivalGapMinutes = cfg.meanArrivalGapMinutes; end
if isempty(aiServiceTimeSeconds), aiServiceTimeSeconds = cfg.aiServiceTimeSeconds; end
if isempty(reviewServiceTimeSeconds), reviewServiceTimeSeconds = cfg.reviewSecondsOptimistic; end

% Severity-mix validation (length/order/nonneg/sum~1 + provenance tag).
% The mix is a SCENARIO (ARDA-sourced), never presented as measured here.
assert(isvector(cfg.severityMix) && numel(cfg.severityMix) == 5 && all(cfg.severityMix >= 0), ...
    'buildTelemedModel:badSeverityMix - severityMix must be 5 non-negative fractions [NoDR Mild Moderate Severe PDR].');
assert(abs(sum(cfg.severityMix) - 1) < 1e-9, ...
    'buildTelemedModel:badSeverityMix - severityMix must sum to 1 (got %.6f).', sum(cfg.severityMix));

% Transmission delay DERIVED from payload/bandwidth/overhead (single
% canonical helper - never inline bits/bytes arithmetic here).
txSeconds = telemedTransmissionSeconds(cfg);

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
new_system(modelName);
open_system(modelName);

add_block('simevents/Generators/Entity Generator', [modelName '/Patient_Arrivals']);
set_param([modelName '/Patient_Arrivals'], 'IntergenerationTimeAction', sprintf('exprnd(%g)', meanArrivalGapMinutes));
set_param([modelName '/Patient_Arrivals'], 'GenerateActionType', 'user-specified');
sev = cfg.severityMix;
set_param([modelName '/Patient_Arrivals'], 'GenerateAction', [ ...
    sprintf('r = rand; cumP = cumsum([%g %g %g %g %g]); ', sev(1), sev(2), sev(3), sev(4), sev(5)) ...
    'severity = find(r <= cumP, 1, ''first'') - 1; ' ...
    'TriagePort = (severity >= 2) + 1;']);
% Severity distribution defaults to the ARDA-sourced SCENARIO mix in
% cfg (order [NoDR Mild Moderate Severe PDR]); referable (Level 2+) is
% therefore sum(sev(3:5)) of screened patients. SCENARIO values, not
% locally measured prevalence - see cfg.severityProvenance.

add_block('simevents/Queues/Entity Queue', [modelName '/Bandwidth_Queue']);
set_param([modelName '/Bandwidth_Queue'], 'Capacity', sprintf('%d', cfg.bandwidthQueueCapacity));
set_param([modelName '/Bandwidth_Queue'], 'AverageWait', 'on');
set_param([modelName '/Bandwidth_Queue'], 'NumberEntitiesInBlock', 'on');

% Transmission server: the configured payload/bandwidth/overhead DERIVED
% delay (telemedTransmissionSeconds) as an actual service station, so
% bandwidth parameters AFFECT queueing instead of decorating comments.
add_block('simevents/Servers/Entity Server', [modelName '/Transmission_Server']);
set_param([modelName '/Transmission_Server'], 'ServiceTimeAction', sprintf('%g', txSeconds));

add_block('simevents/Servers/Entity Server', [modelName '/AI_Pipeline_Server']);
set_param([modelName '/AI_Pipeline_Server'], 'ServiceTimeAction', sprintf('%g', aiServiceTimeSeconds));

add_block('simevents/Routing/Entity Output Switch', [modelName '/Clinical_Triage']);
set_param([modelName '/Clinical_Triage'], 'NumOutputPorts', '2');
set_param([modelName '/Clinical_Triage'], 'SwitchCriterion', 'From attribute');
set_param([modelName '/Clinical_Triage'], 'SwitchAttributeName', 'TriagePort');

add_block('simevents/Sinks/Entity Terminator', [modelName '/Cleared_NonReferable']);

add_block('simevents/Servers/Entity Server', [modelName '/Ophthalmologist_Review']);
set_param([modelName '/Ophthalmologist_Review'], 'ServiceTimeAction', ...
    sprintf('normrnd(%g, %g)', reviewServiceTimeSeconds, max(reviewServiceTimeSeconds/6, 0.5)));
set_param([modelName '/Ophthalmologist_Review'], 'AverageWait', 'on');
set_param([modelName '/Ophthalmologist_Review'], 'Utilization', 'on');
add_block('simevents/Sinks/Entity Terminator', [modelName '/Reviewed_Referable']);

add_block('simulink/Sinks/Scope', [modelName '/Bandwidth_Wait_Monitor']);
add_block('simulink/Sinks/Scope', [modelName '/Ophthalmologist_Wait_Monitor']);

% Machine-readable logging (Scopes are demo-visual only): To Workspace
% blocks so sim() results carry extractable wait metrics, not just
% pictures. Variable names are returned in the run report.
add_block('simulink/Sinks/To Workspace', [modelName '/BW_Wait_Log']);
set_param([modelName '/BW_Wait_Log'], 'VariableName', 'bwWaitLog', 'SaveFormat', 'Timeseries');
add_block('simulink/Sinks/To Workspace', [modelName '/Review_Wait_Log']);
set_param([modelName '/Review_Wait_Log'], 'VariableName', 'reviewWaitLog', 'SaveFormat', 'Timeseries');

set_param([modelName '/Patient_Arrivals'],       'Position', [ 40 100  90 150]);
set_param([modelName '/Bandwidth_Queue'],        'Position', [160 100 210 150]);
set_param([modelName '/Transmission_Server'],    'Position', [280 100 330 150]);
set_param([modelName '/AI_Pipeline_Server'],     'Position', [400 100 450 150]);
set_param([modelName '/Clinical_Triage'],        'Position', [520 100 570 150]);
set_param([modelName '/Cleared_NonReferable'],   'Position', [700  30 750  60]);
set_param([modelName '/Ophthalmologist_Review'], 'Position', [700 170 750 220]);
set_param([modelName '/Reviewed_Referable'],     'Position', [820 170 870 220]);
set_param([modelName '/Bandwidth_Wait_Monitor'],      'Position', [160 260 280 320]);
set_param([modelName '/Ophthalmologist_Wait_Monitor'],'Position', [700 260 820 320]);
set_param([modelName '/BW_Wait_Log'],            'Position', [300 260 380 300]);
set_param([modelName '/Review_Wait_Log'],        'Position', [840 260 920 300]);

add_line(modelName, 'Patient_Arrivals/1', 'Bandwidth_Queue/1');
add_line(modelName, 'Bandwidth_Queue/1', 'Transmission_Server/1');
add_line(modelName, 'Transmission_Server/1', 'AI_Pipeline_Server/1');
add_line(modelName, 'AI_Pipeline_Server/1', 'Clinical_Triage/1');
add_line(modelName, 'Clinical_Triage/1', 'Cleared_NonReferable/1');
add_line(modelName, 'Clinical_Triage/2', 'Ophthalmologist_Review/1');
add_line(modelName, 'Ophthalmologist_Review/1', 'Reviewed_Referable/1');

add_line(modelName, 'Bandwidth_Queue/w', 'Bandwidth_Wait_Monitor/1');
add_line(modelName, 'Bandwidth_Queue/w', 'BW_Wait_Log/1');
add_line(modelName, 'Ophthalmologist_Review/w', 'Ophthalmologist_Wait_Monitor/1');
add_line(modelName, 'Ophthalmologist_Review/w', 'Review_Wait_Log/1');
end
