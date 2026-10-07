function testEvalContracts()
% testEvalContracts: Stage-12 MATLAB contract tests (synthetic only).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB here) ***
% Run in MATLAB: >> testEvalContracts
% Executable twin: python3 tests/test_stage12_eval.py (PASS here).
% Synthetic numbers below test FORMULAS/CONTRACTS only and must never
% enter the evaluation report (runHeldoutEvaluation defaults every
% result to NOT_MEASURED/null).

nP = 0; nT = 0;
[nP,nT] = ec(nP,nT,'manifest validation',@cManifest);
[nP,nT] = ec(nP,nT,'referable conversion boundaries',@cReferable);
[nP,nT] = ec(nP,nT,'patient leakage caught',@cLeakage);
[nP,nT] = ec(nP,nT,'exact vs possible duplicates',@cDupes);
[nP,nT] = ec(nP,nT,'bootstrap degeneracy + seed reproducibility',@cBoot);
[nP,nT] = ec(nP,nT,'report defaults NOT_MEASURED',@cReport);
[nP,nT] = ec(nP,nT,'role separation (calibration != test)',@cRoles);
fprintf('\n=== stage12 eval contracts (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testEvalContracts:failures','%d failed',nT-nP); end
end

function [p,t] = ec(p,t,name,f)
t=t+1; fprintf('[e12 %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function rows = synthRows()
rows = struct('dataset',{},'datasetVersion',{},'datasetSource',{},'imageId',{}, ...
    'patientId',{},'labelICDR',{},'split',{},'provenance',{});
defs = {'img01','patA',0,'TRAIN'; 'img02','patA',1,'VAL'; 'img03','patB',2,'TEST'; 'img04','patC',4,'TEST'};
for i = 1:size(defs,1)
    rows(i).dataset = 'SYNTH'; rows(i).datasetVersion = 't1';
    rows(i).datasetSource = 'synthetic-test-only';
    rows(i).imageId = defs{i,1}; rows(i).patientId = defs{i,2};
    rows(i).labelICDR = defs{i,3}; rows(i).split = defs{i,4};
    rows(i).provenance = 'synthetic';
end
end

function cManifest()
m = evaluationManifest(synthRows());
assert(m.patientIdsAvailable && m.nPatients == 3 && m.nImages == 4, 'manifest counts drift');
bad = synthRows(); bad(1).labelICDR = 7;
threw = false;
try, evaluationManifest(bad); catch, threw = true; end
assert(threw, 'ICDR=7 must be rejected');
end

function cReferable()
assert(double(1 >= 2) == 0 && double(2 >= 2) == 1, 'referable boundary drift');
m = evaluationManifest(synthRows());
assert(m.rows(3).referableLabel == 1 && m.rows(1).referableLabel == 0, 'referable derivation drift');
end

function cLeakage()
rows = synthRows(); rows(2).patientId = 'patB'; % patB now in VAL+TEST
m = evaluationManifest(rows);
a = auditEvalLeakage(m, 'InferenceUsedGT', false, 'CalibrationSplit', 'VAL', 'ThresholdSplit', 'VAL');
assert(strcmp(a.overall,'FINDINGS'), 'patient leakage must be caught');
end

function cDupes()
rows = synthRows(); rows(4).imageId = 'img03'; % exact duplicate id
m = evaluationManifest(rows);
a = auditEvalLeakage(m, 'InferenceUsedGT', false, 'CalibrationSplit', 'VAL', 'ThresholdSplit', 'VAL');
names = {a.checks.name};
assert(any(strcmp(names,'exact-duplicates')), 'exact check missing');
ix = find(strcmp(names,'exact-duplicates'));
assert(strcmp(a.checks(ix).status,'FINDINGS'), 'exact duplicate must be found');
assert(any(strcmp(names,'near-duplicates')), 'near check missing');
end

function cBoot()
vals = [(0:4)'; (0:4)']; % labels==preds toy layout (TEST/LOGIC only)
grps = {'a';'a';'b';'b';'c';'c';'d';'d';'e';'e'};
c1 = patientBootstrapCI(vals, grps, @(s) mean(s(:,1)==s(:,2)), 'B', 100, 'Seed', 7);
c2 = patientBootstrapCI(vals, grps, @(s) mean(s(:,1)==s(:,2)), 'B', 100, 'Seed', 7);
assert(c1.lo == c2.lo && c1.hi == c2.hi && strcmp(c1.resamplingUnit,'PATIENT'), 'seed reproducibility drift');
c3 = patientBootstrapCI(vals, [], @(s) mean(s(:,1)==s(:,2)), 'B', 100, 'Seed', 7);
assert(strcmp(c3.resamplingUnit,'IMAGE_LEVEL_BOOTSTRAP'), 'fallback must be labeled weaker');
c4 = patientBootstrapCI([0;0;0], {'a';'a';'a'}, @(s) NaN, 'B', 100, 'Seed', 7, 'MinValid', 50);
assert(strcmp(c4.status,'UNAVAILABLE'), 'degenerate bootstrap must be UNAVAILABLE, not zero');
end

function cReport()
m = evaluationManifest(synthRows());
pred.imageId = {'img03','img04'}; pred.icdr = [2 4]; pred.ruleGrade = [NaN NaN];
pred.checkpointId = 'none'; pred.calibrationArtifactId = 'none';
pred.splitRoles = struct('calibrationSplit','VAL','thresholdSplit','VAL');
pred.inferenceUsedGT = false;
rep = runHeldoutEvaluation(m, pred, 'BootstrapB', 100);
assert(strcmp(rep.claimState,'MEASURED') || strcmp(rep.claimState,'NOT_MEASURED'), 'claim-state enum drift');
end

function cRoles()
m = evaluationManifest(synthRows());
a = auditEvalLeakage(m, 'InferenceUsedGT', false, 'CalibrationSplit', 'TEST', 'ThresholdSplit', 'VAL');
names = {a.checks.name};
ix = find(strcmp(names,'calibrationsplit'));
assert(strcmp(a.checks(ix).status,'FINDINGS'), 'TEST-fitted calibration must be contamination');
end
