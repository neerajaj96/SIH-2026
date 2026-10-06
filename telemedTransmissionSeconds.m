function tSec = telemedTransmissionSeconds(cfg)
% telemedTransmissionSeconds: Canonical payload/bandwidth/overhead ->
% transmission delay. Single implementation; every simulation and every
% analytic transmission computation must call this (never inline the
% arithmetic, where bits/bytes/MB/Mbps get silently mixed).
%
%   t = imageMB * 8 (bits/byte) * (1 + overheadFraction) / bandwidthMbps
%
% Units: imageMB (megabytes) * 8 = megabits; megabits / megabits-per-
% second = seconds. Overhead multiplies the payload (retransmits +
% protocol framing), it does NOT add to bandwidth.
%
% Requires: base MATLAB only.

if nargin < 1 || isempty(cfg), cfg = telemedConfig(); end
assert(cfg.imageMB > 0 && cfg.bandwidthMbps > 0, ...
    'telemedTransmissionSeconds:badInputs - imageMB and bandwidthMbps must be positive.');
assert(cfg.protocolOverheadFraction >= 0, ...
    'telemedTransmissionSeconds:badInputs - overhead fraction cannot be negative.');
payloadMbits = cfg.imageMB * 8;
tSec = payloadMbits * (1 + cfg.protocolOverheadFraction) / cfg.bandwidthMbps;
end
