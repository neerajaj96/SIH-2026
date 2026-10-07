function testDataContracts()
% testDataContracts: Stage-13 MATLAB contract tests (synthetic only).
%
% *** UNEXECUTED IN THIS WORKSPACE (no MATLAB here) ***
% Run in MATLAB: >> testDataContracts
% Executable twin: python3 tests/test_stage13_data.py (PASS here).
% Synthetic values below are TEST/LOGIC fixtures and must never enter
% an evaluation report (runHeldoutEvaluation defaults NOT_MEASURED).

nP = 0; nT = 0;
[nP,nT] = dc(nP,nT,'registry structured fields',@cRegistry);
[nP,nT] = dc(nP,nT,'acceptance verdict contract',@cAccept);
[nP,nT] = dc(nP,nT,'freeze order (TEST first)',@cOrder);
[nP,nT] = dc(nP,nT,'timestamp-independent identity',@cIdentity);
[nP,nT] = dc(nP,nT,'calibration mismatch invalidates',@cMismatch);
[nP,nT] = dc(nP,nT,'preprocessing parity pins',@cParity);
fprintf('\n=== stage13 data contracts (MATLAB, UNEXECUTED HERE): %d / %d ===\n', nP, nT);
if nP < nT, error('testDataContracts:failures','%d failed',nT-nP); end
end

function [p,t] = dc(p,t,name,f)
t=t+1; fprintf('[d13 %2d] %s ... ',t,name);
try, f(); fprintf('PASS\n'); p=p+1;
catch E, fprintf('FAIL\n      %s\n',E.message); end
end

function cRegistry()
reg = datasetRegistry();
names = {reg.name};
assert(ismember('Messidor-2', names) && ismember('DDR', names), 'registry names drift');
m = reg(strcmp(names,'Messidor-2'));
assert(strcmp(m.sourceType,'mirror') && m.expectedImageCount == 1748, 'Messidor-2 structured fields drift');
d = reg(strcmp(names,'DDR'));
assert(isnan(d.expectedImageCount), '~10k must stay COUNT_NOT_VERIFIED (NaN)');
assert(isfield(reg,'versionPolicy') && isfield(reg,'expectedMaskCount'), 'structured fields missing');
end

function cAccept()
v = acceptDataset('Messidor-2');
assert(isfield(v,'overallStatus') && isfield(v,'imageCountStatus'), 'verdict contract drift');
assert(strcmp(v.overallStatus,'ACCEPTED_WITH_NOTES') || strcmp(v.overallStatus,'REJECTED'), 'verdict enum drift');
end

function cOrder()
pre = freezeCandidate('PRE_TRAIN', struct('manifestHash','m','trainSplitHash','tr', ...
    'valSplitHash','va','testSplitHash','te','testManifestVersion','v1'));
assert(strcmp(pre.candidateState,'PRE_TRAIN_CANDIDATE'), 'state drift');
fin = freezeCandidate('FINALIZE', struct('preTrainCandidate', pre, ...
    'checkpointId','ckpt','checkpointHash','ch','validationSplitId','VAL', ...
    'calibrationArtifactId','cal','calibrationArtifactHash','cah', ...
    'calibrationProvenance', struct('checkpointId','ckpt','splitId','VAL','classOrdering',(0:4)'), ...
    'modelConfig', struct('arch','densenet121'), ...
    'trainingProvenance', struct('seed',42)));
assert(strcmp(fin.candidateState,'FINALIZED_CANDIDATE'), 'finalize drift');
assert(strcmp(fin.candidateId, pre.candidateId) == 0, 'finalize must mint a new ID');
end

function cIdentity()
base = struct('manifestHash','m','trainSplitHash','tr','valSplitHash','va', ...
    'testSplitHash','te','testManifestVersion','v1');
a = freezeCandidate('PRE_TRAIN', base);
pause(1.1);
b = freezeCandidate('PRE_TRAIN', base);
assert(strcmp(a.candidateId, b.candidateId), 'timestamps leaked into candidate identity');
end

function cMismatch()
pre = freezeCandidate('PRE_TRAIN', struct('manifestHash','m','trainSplitHash','tr', ...
    'valSplitHash','va','testSplitHash','te','testManifestVersion','v1'));
threw = false;
try
    freezeCandidate('FINALIZE', struct('preTrainCandidate', pre, ...
        'checkpointId','ckpt','checkpointHash','ch','validationSplitId','VAL', ...
        'calibrationArtifactId','cal','calibrationArtifactHash','cah', ...
        'calibrationProvenance', struct('checkpointId','OTHER','splitId','VAL','classOrdering',(0:4)'), ...
        'modelConfig', struct('arch','densenet121'), ...
        'trainingProvenance', struct('seed',42)));
catch
    threw = true;
end
assert(threw, 'checkpoint-mismatched calibration must invalidate');
end

function cParity()
t = fileread('train_UNet_Segmentation.m');
assert(~isempty(strfind(t,'preprocessFundusForSegmentation')), 'seg train left canonical preprocessing');
g = fileread('train_DR_Grader.m');
assert(~isempty(strfind(g,'buildGradingFusionTensor')), 'grader train left canonical fusion');
assert(~isempty(strfind(g,'segmentationNotTrained')), 'predicted-mask gate missing');
end
