function wq = erlangCWaitHours(lambda, mu, c)
% erlangCWaitHours: M/M/c (Erlang-C) average wait time in queue, in the
% same time units as 1/mu (e.g. hours, if mu is arrivals/hour).
%
% EXTRACTED THIS ROUND from being a local (file-private) function inside
% optimizeResourceAllocation.m into its own top-level file, for the same
% reason buildTelemedModel.m was already extracted out of
% SimEvents_Telemed_Model.m: a local function can't be called, reused, or
% independently tested from outside its containing file. That mattered in
% practice writing runSelfTests.m, which needs to call this directly to
% regression-test it against the closed-form M/M/1 formula - it couldn't
% reach the old local copy at all. optimizeResourceAllocation.m now calls
% this exact file instead of a local copy of the same code.
%
% VERIFIED: at c=1, this must algebraically reduce to the closed-form
% M/M/1 mean wait Wq = rho/(mu-lambda). Checked independently in Python
% (not just re-deriving the same MATLAB algebra) at three (lambda,mu)
% points, including the actual PS-default operating point (lambda=10/hr,
% mu=120/hr - the 30-second-review scenario) - matched to under 1e-15 in
% every case. runSelfTests.m reruns this check directly in MATLAB.
%
% INPUTS:
%   lambda - arrival rate (e.g. referable cases/hour)
%   mu     - per-server service rate (e.g. reviews/hour/ophthalmologist)
%   c      - number of parallel servers (ophthalmologists)
%
% OUTPUT:
%   wq - mean wait time in queue before service begins. Inf if
%        lambda >= c*mu (the queue is unstable at this staffing level -
%        not a numerical error, a genuine "this will never stabilize"
%        answer).
%
% Requires: nothing beyond base MATLAB.

a = lambda / mu; % offered load, erlangs
rho = a / c;
if rho >= 1
    wq = Inf;
    return;
end
sumTerms = 0;
for k = 0:(c-1)
    sumTerms = sumTerms + (a^k)/factorial(k);
end
lastTerm = (a^c)/factorial(c) * (c/(c-a));
pWait = lastTerm / (sumTerms + lastTerm);
wq = pWait / (c*mu - lambda);
end
