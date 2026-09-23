function [idxD, idxN, idxF, idxSea] = boundary_segments(xb, cfg)
% Geographically corrected boundary roles (v2.2):
%   idxF   : FREE inflow arcs (Mat NE + Droja/Ishem SE valley mouths) - NO penalty
%   idxN   : no-flow set = eastern hill contacts AND the Adriatic confined
%            shoreline (the confined unit continues offshore beneath the
%            aquitard, so the shoreline is NOT a fixed-head boundary)
%   idxD   : Dirichlet arc(s) - EMPTY in the canonical v2.2 configuration
%   idxSea : subset of idxN flagged as the Adriatic shoreline (plotting only)
% v2.6 canonical config: cfg.idxFree = [1 59; 861 3411] -> Neumann set is the
% Adriatic shoreline ONLY; the entire land perimeter is FREE (lateral
% mountain-front/valley inflow permitted, resolved by data + PDE).
% All ranges are vertex-index intervals [a b] (one per row) on the
% de-duplicated polygon; N = size(xb,2).
N = size(xb,2);
idxF = ranges_(cfg.idxFree, N);
if isfield(cfg,'idxCoast') && ~isempty(cfg.idxCoast)
    idxD = setdiff(ranges_(cfg.idxCoast, N), idxF);
else
    idxD = [];
end
idxN = setdiff(1:N, [idxF, idxD]);
if isfield(cfg,'idxSea') && ~isempty(cfg.idxSea)
    idxSea = intersect(ranges_(cfg.idxSea, N), idxN);
else
    idxSea = [];
end
end
function v = ranges_(R, N)
v = [];
for r = 1:size(R,1)
    a = max(1, R(r,1)); b = min(N, R(r,2));
    v = [v, a:b]; %#ok<AGROW>
end
end
