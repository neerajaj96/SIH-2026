function result = assessIRMA(vesselMask, quadrantMask, varargin)
% assessIRMA: EXTENSION POINT ONLY - NOT a detector.
%
% IRMA has NO validated automated detector in this release. IRMA must not
% be confused with neovascularization, tortuosity, or ordinary branching,
% so no heuristic on the current vessel representation is offered as
% evidence. This stub ALWAYS returns status UNAVAILABLE with zeroed counts
% so the 4-2-1 engine treats the "1" trigger as unevaluable.
%
% FUTURE dedicated (validated) detectors must return the same struct with
% status VERIFIED or PROXY and per-quadrant counts. No caller changes.
%
% OUTPUT: result struct with .status ('UNAVAILABLE'), .quadrants (0),
%   .quadrantList (false 1x4), .method, .provenance.

result = struct('status', 'UNAVAILABLE', 'quadrants', 0, ...
    'quadrantList', false(1,4), ...
    'method', 'no detector in this release (extension point)', ...
    'provenance', 'assessIRMA stub: IRMA unevaluable from current stack');
end
