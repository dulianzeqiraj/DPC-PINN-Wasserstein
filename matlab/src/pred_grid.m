function [XG, inMask, gx, gy] = pred_grid(polyB, nx)
% Prediction grid over the polygon bounding box; CHUNKED isinterior (memory-safe).
[xl,xu] = bounds(polyB.Vertices(:,1)); [yl,yu] = bounds(polyB.Vertices(:,2));
gx = linspace(xl,xu,nx); gy = linspace(yl,yu,round(nx*(yu-yl)/(xu-xl)));
[GX,GY] = meshgrid(gx,gy);
P = [GX(:), GY(:)];  n = size(P,1);
inMask = false(n,1);  CH = 20000;
for s = 1:CH:n
    e = min(s+CH-1, n);
    inMask(s:e) = isinterior(polyB, P(s:e,1), P(s:e,2));
end
XG = [GX(:)'; GY(:)'];
XG = XG(:, inMask);
end
