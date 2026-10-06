% =========================================================================
% MODULE 5 (fixed): Programmatic Generation of a SimEvents Queuing Model
% Requires: Simulink, SimEvents (both must be licensed/installed separately
%           from base MATLAB - check with `ver` if unsure)
%
% This script now just calls buildTelemedModel.m with demo defaults - see
% that file for the actual block/wiring logic, and optimizeResourceAllocation.m
% for the resource-sweep this feeds into.
%
% FIXES vs. the original draft (kept here since they're still the reason
% this topology looks the way it does):
%   - Six blocks were added but only four add_line calls existed. The
%     Output Switch had just ONE outgoing connection even though routing
%     referable vs. non-referable cases is the entire point of the block;
%     there was no Entity Terminator anywhere for the non-referable ~80%
%     the pitch narrative describes being filtered out automatically; and
%     the Scope was added but never connected to anything.
%   - Nothing ever set an attribute on an entity, so the Output Switch had
%     no actual signal to route on even with a second port wired.
%   - The patient arrival gap (mean 3, implicitly the same time units as
%     the AI server's explicit "2.5 seconds") worked out to roughly 1,200
%     patients/hour - not physically achievable, and it meant the pitch's
%     "500-patient surge" would resolve in about 25 simulated minutes.
%     Arrivals are now in minutes (mean 6, ~10/hour), consistent with an
%     actual screening-camp pace.
%   - Statistics ports are now actually enabled and wired to Scopes.
%
% CAUTION: SimEvents' library block paths and some event-action parameter
% names are more version-sensitive than base MATLAB - a documented
% MathWorks Answers case shows a naive add_block library-path guess
% failing with "no such block in the library" for a different SimEvents
% block. If any add_block/set_param call in buildTelemedModel.m errors,
% place that one block manually from the Simulink Library Browser, then
% run get_param(gcb,'BlockType') and get_param(gcb,'Parent') on it to find
% the exact strings your release uses, and swap them in - a five-minute
% fix, but do it BEFORE a live demo, not during one.
% =========================================================================
disp('Building SimEvents Tele-Ophthalmology Model...');

modelName = buildTelemedModel('DR_Telemed_Simulation', 6, 2.5, 30);

disp('Simulink model constructed with entity routing AND statistics fully wired.');
disp('Save it with: save_system(modelName)');
% ~500 patients at a 6-minute mean gap needs roughly 3000 minutes of
% simulated time to reproduce the pitch's "500-patient surge".
disp('Run the simulation with: sim(modelName, 3000)');

% Uncomment once you've verified the block paths/parameter names in
% buildTelemedModel.m against your own MATLAB/SimEvents release:
% sim(modelName, 3000);
