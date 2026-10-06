function testTelemedSim()
% testTelemedSim: Stage-8 MATLAB/SimEvents harness (synthetic config only).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB/SimEvents here) ***
% Run in MATLAB (Simulink/SimEvents licensed): >> testTelemedSim
% Executable twin: python3 tests/test_stage8_telemed.py (55/55 PASS).
% NOTHING below measures real operations; SimEvents paths additionally
% need the CAUTION-verified block library (see SimEvents_Telemed_Model.m).
%
% Covers: Erlang-C edges, config validation, transmission math, severity
% mix, optimizer SLA/stability, model construction, consistency verdict,
% seed reproducibility, sweep determinism.

nP = 0; nT = 0;
[nP,nT] = tm(nP,nT,'Erlang edges (unstable/zero/MMc)',@cEdges);
[nP,nT] = tm(nP,nT,'config validation',@cConfig);
[nP,nT] = tm(nP,nT,'transmission math',@cTx);
[nP,nT] = tm(nP,nT,'optimizer SLA + stability',@cOpt);
[nP,nT] = tm(nP,nT,'model construction',@cBuild);
[nP,nT] = tm(nP,nT,'consistency methodology',@cConsistent);
[nP,nT] = tm(nP,nT,'seed reproducibility contract',@cSeed);
fprintf('\n=== stage8 telemed (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testTelemedSim:failures','%d failed',nT-nP); end
end

function [p,t] = tm(p,t,name,f)
t=t+1; fprintf('[s8 %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function cEdges()
assert(isinf(erlangCWaitHours(120,120,1)), 'rho=1 must be Inf, not finite');
assert(isinf(erlangCWaitHours(200,120,1)), 'overload must be Inf');
assert(erlangCWaitHours(0,120,1) == 0, 'zero arrival must be zero wait');
w1 = erlangCWaitHours(10,120,1);
mm1 = (10/120)/(120-10);
assert(abs(w1-mm1) < 1e-9, 'M/M/1 closed-form mismatch');
assert(erlangCWaitHours(10,120,2) < w1, 'more servers must reduce wait');
end

function cConfig()
cfg = telemedConfig();
assert(numel(cfg.severityMix)==5 && all(cfg.severityMix>=0), 'mix shape');
assert(abs(sum(cfg.severityMix)-1) < 1e-9, 'mix must sum to 1');
assert(abs(sum(cfg.severityMix(3:5))-cfg.referableFraction) < 1e-12, 'referable must derive from mix');
assert(strcmp(cfg.severityProvenance(1:4),'ARDA'), 'provenance tag drift');
end

function cTx()
cfg = telemedConfig();
t = telemedTransmissionSeconds(cfg);
assert(abs(t - cfg.imageMB*8*(1+cfg.protocolOverheadFraction)/cfg.bandwidthMbps) < 1e-12, 'unit arithmetic drift');
assert(abs(telemedTransmissionSeconds(struct('imageMB',2,'bandwidthMbps',4,'protocolOverheadFraction',0.1)) - 4.4) < 1e-12, '2MB/4Mbps/10% must be 4.4s');
end

function cOpt()
res = optimizeResourceAllocation(100000, 0.141, 2000, 30, 30);
assert(isfield(res,'objective') && isfield(res.table,'utilization') && isfield(res.table,'stability'), 'optimizer surfacing drift');
assert(all([res.table.utilization] < 2), 'sanity');
end

function cBuild()
cfg = telemedConfig();
mdl = buildTelemedModel('DR_Test_Build', cfg.meanArrivalGapMinutes, cfg.aiServiceTimeSeconds, cfg.reviewSecondsOptimistic, cfg);
assert(bdIsLoaded(mdl), 'model not built');
assert(~isempty(find_system(mdl,'SearchDepth',1,'Name','Transmission_Server')), 'transmission server missing');
assert(~isempty(find_system(mdl,'SearchDepth',1,'Name','Review_Wait_Log')), 'wait logging missing');
close_system(mdl, 0);
end

function cConsistent()
cfg = telemedConfig(); cfg.replications = 3; cfg.simDurationMinutes = 600; cfg.warmupMinutes = 60;
res = evaluateTelemedConsistency(cfg, 1);
assert(isfield(res,'analyticComparison') && isfield(res,'bottleneck'), 'contract drift');
assert(ismember(res.analyticComparison.verdict, {'PASS','DIVERGED','UNSTABLE (analytic Inf - sim comparison meaningless)'}), 'verdict enum drift');
end

function cSeed()
% Same seed + replication index must rebuild identically (builder itself
% holds no RNG state; driver seeds before sim).
cfg = telemedConfig();
m1 = buildTelemedModel('DR_Seed_A', 6, 2.5, 30, cfg);
m2 = buildTelemedModel('DR_Seed_B', 6, 2.5, 30, cfg);
b1 = get_param(m1,'Blocks'); b2 = get_param(m2,'Blocks');
assert(isequal(sort(b1),sort(b2)), 'same config must build same topology');
close_system(m1,0); close_system(m2,0);
end
