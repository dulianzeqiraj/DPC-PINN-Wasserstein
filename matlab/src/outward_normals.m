function [nx, ny] = outward_normals(xb, idx)
% Outward unit normals at boundary vertices idx. DUPLICATE-SAFE: walks to the
% nearest non-coincident neighbours (digitized boundaries often repeat vertices).
Nb  = size(xb,2);
tol = 1e-4;                                  % 0.1 m in km units
nx = zeros(1,numel(idx)); ny = zeros(1,numel(idx));
bad = 0;
for k = 1:numel(idx)
    i = idx(k);
    ip = i; dxp = 0; dyp = 0;
    for s = 1:50                              % walk forward to a distinct vertex
        ip = mod(ip, Nb) + 1;
        dxp = xb(1,ip)-xb(1,i); dyp = xb(2,ip)-xb(2,i);
        if hypot(dxp,dyp) > tol, break; end
    end
    im = i; dxm = 0; dym = 0;
    for s = 1:50                              % walk backward to a distinct vertex
        im = mod(im-2, Nb) + 1;
        dxm = xb(1,i)-xb(1,im); dym = xb(2,i)-xb(2,im);
        if hypot(dxm,dym) > tol, break; end
    end
    tx = dxp + dxm; ty = dyp + dym;           % central-difference tangent
    L = hypot(tx,ty);
    if L > tol
        nx(k) = ty/L; ny(k) = -tx/L;
    else
        bad = bad + 1;                        % leave zero normal -> zero flux penalty there
    end
end
% orient outward (away from centroid)
cx = mean(xb(1,:)); cy = mean(xb(2,:));
s = sign( (xb(1,idx)-cx).*nx + (xb(2,idx)-cy).*ny ); s(s==0) = 1;
nx = nx.*s; ny = ny.*s;
if bad>0, fprintf('outward_normals: %d degenerate boundary points neutralized.\n', bad); end
end
