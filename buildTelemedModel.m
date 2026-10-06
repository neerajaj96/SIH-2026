function modelName = buildTelemedModel(modelName, meanArrivalGapMinutes, aiServiceTimeSeconds, reviewServiceTimeSeconds)
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
% INPUTS:
%   modelName             - string, a fresh model name (closed/overwritten if already loaded)
%   meanArrivalGapMinutes - mean minutes between patient arrivals
%   aiServiceTimeSeconds  - fixed AI pipeline processing time, seconds
%   reviewServiceTimeSeconds - mean ophthalmologist review time, seconds
%                              (std dev fixed at reviewServiceTimeSeconds/6,
%                               i.e. roughly a 5-second spread around a 30s mean)
%
% OUTPUT:
%   modelName - echoed back, for chaining into sim(modelName, ...)

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
new_system(modelName);
open_system(modelName);

add_block('simevents/Generators/Entity Generator', [modelName '/Patient_Arrivals']);
set_param([modelName '/Patient_Arrivals'], 'IntergenerationTimeAction', sprintf('exprnd(%g)', meanArrivalGapMinutes));
set_param([modelName '/Patient_Arrivals'], 'GenerateActionType', 'user-specified');
set_param([modelName '/Patient_Arrivals'], 'GenerateAction', [ ...
    'r = rand; ' ...
    'if r < 0.827, severity = 0; elseif r < 0.859, severity = 1; ' ...
    'elseif r < 0.963, severity = 2; elseif r < 0.972, severity = 3; else, severity = 4; end; ' ...
    'TriagePort = (severity >= 2) + 1;']);
% Distribution (82.7/3.2/10.4/0.9/2.8% for No DR/Mild/Moderate/Severe/PDR)
% is sourced from a real India-specific community DR screening study
% (ARDA program, ~3,941 gradable images), not a synthetic guess. Also
% means referable (Level 2+) is ~14.1% of screened patients, not the 20%
% this project assumed earlier - see optimizeResourceAllocation.m.

add_block('simevents/Queues/Entity Queue', [modelName '/Bandwidth_Queue']);
set_param([modelName '/Bandwidth_Queue'], 'Capacity', '100');
set_param([modelName '/Bandwidth_Queue'], 'AverageWait', 'on');
set_param([modelName '/Bandwidth_Queue'], 'NumberEntitiesInBlock', 'on');

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

set_param([modelName '/Patient_Arrivals'],       'Position', [ 40 100  90 150]);
set_param([modelName '/Bandwidth_Queue'],        'Position', [160 100 210 150]);
set_param([modelName '/AI_Pipeline_Server'],     'Position', [280 100 330 150]);
set_param([modelName '/Clinical_Triage'],        'Position', [400 100 450 150]);
set_param([modelName '/Cleared_NonReferable'],   'Position', [520  30 570  60]);
set_param([modelName '/Ophthalmologist_Review'], 'Position', [520 170 570 220]);
set_param([modelName '/Reviewed_Referable'],     'Position', [640 170 690 220]);
set_param([modelName '/Bandwidth_Wait_Monitor'],      'Position', [160 260 280 320]);
set_param([modelName '/Ophthalmologist_Wait_Monitor'],'Position', [520 260 640 320]);

add_line(modelName, 'Patient_Arrivals/1', 'Bandwidth_Queue/1');
add_line(modelName, 'Bandwidth_Queue/1', 'AI_Pipeline_Server/1');
add_line(modelName, 'AI_Pipeline_Server/1', 'Clinical_Triage/1');
add_line(modelName, 'Clinical_Triage/1', 'Cleared_NonReferable/1');
add_line(modelName, 'Clinical_Triage/2', 'Ophthalmologist_Review/1');
add_line(modelName, 'Ophthalmologist_Review/1', 'Reviewed_Referable/1');

add_line(modelName, 'Bandwidth_Queue/w', 'Bandwidth_Wait_Monitor/1');
add_line(modelName, 'Ophthalmologist_Review/w', 'Ophthalmologist_Wait_Monitor/1');
end
