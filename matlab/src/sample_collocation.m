function Xc = sample_collocation(polyB, Nc)
% Rejection sampling inside the polygon, with CHUNKED isinterior calls
% (memory-safe for high-vertex polygons).
CH = 5000;                                   % candidates per isinterior call
[xl, xu] = bounds(polyB.Vertices(:,1)); [yl, yu] = bounds(polyB.Vertices(:,2));
Xc = zeros(2, 0);
while size(Xc,2) < Nc
    m = min(CH, 2*(Nc - size(Xc,2)) + 100);
    P = [xl + (xu-xl)*rand(1,m); yl + (yu-yl)*rand(1,m)];
    in = isinterior(polyB, P(1,:)', P(2,:)');
    Xc = [Xc, P(:,in')]; %#ok<AGROW>
end
Xc = Xc(:, 1:Nc);
end
