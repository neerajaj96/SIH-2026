function result = assessVenousBeading(vesselMask, quadrantMask, varargin)
% assessVenousBeading: EXTENSION POINT ONLY - NOT a detector.
%
% Venous beading has NO validated automated detector in this release, and
% vessel-caliber heuristics on a binary vessel mask are not credible
% evidence of beading. This stub therefore ALWAYS returns status
% UNAVAILABLE with zeroed counts so the 4-2-1 engine treats the "2"
% trigger as unevaluable rather than absent.
%
% FUTURE dedicated (validated) detectors must return the same struct with
% status VERIFIED or PROXY and per-quadrant counts. No caller changes.
%
% OUTPUT: result struct with .status ('UNAVAILABLE'), .quadrants (0),
%   .quadrantList (false 1x4), .method, .provenance.

result = struct('status', 'UNAVAILABLE', 'quadrants', 0, ...
    'quadrantList', false(1,4), ...
    'method', 'no detector in this release (extension point)', ...
    'provenance', 'assessVenousBeading stub: VB unevaluable from current stack');
end
