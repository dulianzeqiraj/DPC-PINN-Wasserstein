function [loss, grads, parts] = model_loss(params, cfg, T, B, sc)
% Six-term DPC-PINN loss + boundary penalties. All derivatives by dlgradient.
Xc = B.Xc;                                 % traced: entered via dlfeval inputs

% ---- properties & state at collocation (training=true: dropout regularizes) ----
[logK, n, h] = forward_all(params, Xc, cfg, true);

% first derivatives of h and logK wrt normalized coords
hn = (h - cfg.hMu) / cfg.hSd;                                          % O(1) head
gh = dlgradient(sum(hn,'all'),  Xc, 'EnableHigherDerivatives', true);  % 2×Nc
gK = dlgradient(sum(logK,'all'), Xc, 'EnableHigherDerivatives', true); % 2×Nc

Kn  = exp(logK - cfg.kMu);                                             % K/K0, O(1)
qx = Kn .* gh(1,:);  qy = Kn .* gh(2,:);
dqx = dlgradient(sum(qx,'all'), Xc, 'EnableHigherDerivatives', true);
dqy = dlgradient(sum(qy,'all'), Xc, 'EnableHigherDerivatives', true);
% FULLY non-dimensional steady Darcy residual: constants (b, K0, hSd, L^2) absorbed.
% Valid as-is for N=0; if N~=0 later, use N* = N * L^2 / (K0 * b * hSd).
res = dqx(1,:) + dqy(2,:);
L1 = mean(res.^2);

% ---- data misfits ----
[logKw, ~, hw] = forward_all(params, T.xw, cfg, true);
L2 = mean( ((hw - T.h)/cfg.hSd).^2 );    % O(1) misfit
L3 = mean( (logKw - T.K).^2 );

% ---- L4: DRASTIC auxiliary (standardized, learnable affine, a>0) ----
% v2.7: the canonical configuration sets w.L4 = 0 and the term is NOT evaluated, so
% no DRASTIC information enters the inversion at all (not even through the dropout
% RNG stream). The branch is kept so the ablation can still quantify what the term
% would have contributed. Rationale: over the 43 grid points that carry a pumping
% test, r(DRASTIC, lnK) = +0.15, r^2 = 0.02, so the index does not inform logK here.
if cfg.w.L4 > 0
    [logKd, ~, ~] = forward_all(params, T.xd, cfg, true);
    zK = (logKd - mean(logKd)) ./ (std(logKd) + 1e-8);
    a  = log(1+exp(params.aux.alpha));                   % softplus
    L4 = mean( (zK - (a.*T.D + params.aux.c)).^2 );
else
    L4 = 0*L1;
end

% ---- L5: paleo-channel anisotropic prior in local frame theta ----
ct = cos(B.th); st = sin(B.th);
gt = gK(1,:).*ct + gK(2,:).*st;          % along-channel
gn = -gK(1,:).*st + gK(2,:).*ct;         % across-channel
% R5 sits on the ALONG-channel term. A channel is a body in which logK barely changes as you
% walk along it and drops as you step across, so it is the along-channel gradient that must be
% held down. The weight was on gn until v2.8, which encoded the fabric rotated by ninety degrees:
% a field elongated along theta scored worse in its own frame than in the perpendicular one, by
% 4.0x for a plane wave, 3.3x over 40 realisations of aniso_field (main_synthetic.m) and 1.3x
% for the synthetic truth itself.
L5 = mean( cfg.R5 .* gt.^2 + gn.^2 );

% ---- L6: Kozeny–Carman structural coupling with facies prefactor ----
bF = params.aux.betaF(B.kc);                             % per-point log prefactor (numeric indices)
bF = reshape(bF,1,[]);
L6 = mean( (logK - (bF + log(n.^3./(1-n).^2))).^2 );

% ---- L7: channel-distance geometric prior (v2.6), well-distance-ramped ----
L7 = mean( B.w7(:) .* (logK(:) - B.kp(:)).^2 );

% ---- boundary penalties ----
if size(T.xbD,2) > 0
    [~, ~, hD] = forward_all(params, T.xbD, cfg, true);
    BCd = mean( ((hD - T.hsea)/cfg.hSd).^2 );
else
    BCd = 0*L1;     % no Dirichlet arc in the canonical v2.2 configuration
end
XbN = T.xbN;                               % traced via dlfeval inputs
[logKN, ~, hN] = forward_all(params, XbN, cfg, true); %#ok<ASGLU>
ghN = dlgradient(sum((hN - cfg.hMu)/cfg.hSd,'all'), XbN, 'EnableHigherDerivatives', true);
flux = exp(logKN - cfg.kMu) .* ( ghN(1,:).*T.nrmN(1,:) + ghN(2,:).*T.nrmN(2,:) );
BCn = mean( flux.^2 );
LBC = BCd + BCn;

w = cfg.w;
loss = w.L1*L1 + w.L2*L2 + w.L3*L3 + w.L4*L4 + w.L5*L5 + w.L6*L6 + w.L7*L7 + w.BC*LBC;
grads = dlgradient(loss, params);
parts = double(gather(extractdata([L1 L2 L3 L4 L5 L6 L7 BCd BCn])));
end
