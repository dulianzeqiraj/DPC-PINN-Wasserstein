function out = geo_prior(mode, varargin)
% GEO_PRIOR  Major-axis geometric prior for logK (v2.6.1).
%   axesXY = geo_prior('axes', AXT)
%       Densify the ordered river-axis table (river,x_km,y_km,order) to a
%       2xM point set sampled every ~0.1 km along each polyline.
%   P = geo_prior('calib', axesXY, netXY, xw, logKw, cfg)
%       logK_prior(x) = a + b*exp(-(max(0,d_axis-W0)/Ld)^2), least-squares on wells.
%       d_axis uses MAJOR river axes only (paleo-corridor proxy); the full
%       channel network (netXY) is kept solely for local orientation.
%   out = geo_prior('eval', P, Xq, xw)
%       .logKp .w7 .dchan .thLoc   (thLoc from netXY neighbourhood PCA)
%   out = geo_prior('eval', P, Xq, xw, 'noOrientation')
%       as above without .thLoc, which depends on the network only (used per fold)
% Coordinates: 2xN km, rows = [Easting; Northing].
switch mode
    case 'axes'
        % Depositional-valley axes ONLY: the upstream Mat gorge (Valley-2 arm,
        % y > yGorge) is a mountain canyon, not a high-K depositional corridor,
        % and must not carry the valley prior (A. Beqiraj, field knowledge).
        AXT = varargin{1};
        yGorge = 44.0;                                   % km; Bishtez gate ~ lower-valley start
        keep = ~(string(AXT.river)=="MAT" & AXT.y_km > yGorge);
        AXT = AXT(keep,:);
        pts = [];
        for r = unique(string(AXT.river))'
            S = sortrows(AXT(string(AXT.river)==r,:), 'order');
            if height(S) < 2, pts = [pts, [S.x_km'; S.y_km']]; continue, end %#ok<AGROW>
            for k = 1:height(S)-1
                p0 = [S.x_km(k);   S.y_km(k)];
                p1 = [S.x_km(k+1); S.y_km(k+1)];
                L  = norm(p1-p0);  m = max(2, ceil(L/0.1));
                tt = linspace(0,1,m);
                pts = [pts, p0 + (p1-p0)*tt]; %#ok<AGROW>
            end
        end
        out = pts;
    case 'calib'
        [axesXY, netXY, xw, logKw, cfg] = deal(varargin{:});
        P.Ld = cfg.Ld; P.W0 = cfg.W0; P.Rw = cfg.Rw; P.Lth = 1.2;
        P.axesXY = axesXY; P.netXY = netXY;
        [~, dW] = knnsearch(axesXY', xw');
        g  = exp(-(max(0, dW - P.W0)/P.Ld).^2);   % plateau-Gaussian: flat inside the valley body
        A  = [ones(numel(g),1), g(:)];
        ab = A \ logKw(:);
        P.a = ab(1); P.b = ab(2);
        r  = A*ab - logKw(:);
        P.R2 = 1 - sum(r.^2) / sum((logKw(:)-mean(logKw)).^2);
        out = P;
    case 'eval'
        [P, Xq, xw] = deal(varargin{1:3});
        [~, d] = knnsearch(P.axesXY', Xq');
        out.dchan = d(:)';
        out.logKp = P.a + P.b * exp(-(max(0, out.dchan - P.W0)/P.Ld).^2);
        [~, dwell] = knnsearch(xw', Xq');
        out.w7 = 1 - exp(-(dwell(:)'/P.Rw).^2);
        if numel(varargin) > 3 && strcmp(varargin{4}, 'noOrientation'), return, end
        nb = rangesearch(P.netXY', Xq', P.Lth);
        th = nan(1, size(Xq,2));
        for q = 1:numel(nb)
            j = nb{q};
            if numel(j) >= 8
                C = P.netXY(:, j); C = C - mean(C, 2);
                [V, E] = eig(C * C');
                [~, k] = max(diag(E));
                th(q) = mod(atan2(V(2,k), V(1,k)), pi);
            end
        end
        out.thLoc = th;
    otherwise
        error('geo_prior: unknown mode %s', mode);
end
end
