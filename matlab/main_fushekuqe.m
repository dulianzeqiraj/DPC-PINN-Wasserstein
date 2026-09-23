%% DPC-PINN-Wasserstein - Real-data inversion, Fushë-Kuqe aquifer (349 km², 37 wells)
%  Author: Dulian Zeqiraj (UPT-FGJM).  License: MIT.
%  Requires: MATLAB R2023b+ with Deep Learning Toolbox (dlarray/dlgradient/adamupdate).
%
%  Run from the repository root:  >> cd matlab; main_fushekuqe
%  Outputs: results/*.csv, figures/fig5_Khat.png, fig6_nhat.png, figS2_sigma.png
%
%  HONESTY NOTE: every number reported in the manuscript must be produced by THIS
%  script (fixed seed below). Nothing is hand-entered.

clear; clc; close all;
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');
addpath(fullfile(pwd,'src'));
rng(20260610,'twister');                       % fixed, reported seed

%% ---------------- Configuration ----------------
cfg.b        = 40;          % saturated thickness [m]  (CONFIRMED by D.Z., 10-Jun-2026)
cfg.N        = 0;           % areal recharge; N=0 initial setting (CONFIRMED); revisit if residual structure demands
cfg.hsea     = 0;           % (unused when cfg.idxCoast is empty; kept for optional Dirichlet variants)
cfg.R5       = 4.0;         % L5 anisotropy ratio (from ln K variogram, 3–5)
cfg.nMin     = 0.20; cfg.nMax = 0.40;          % porosity band
cfg.w        = struct('L1',1,'L2',10,'L3',10,'L4',0,'L5',0.05,'L6',0.5,'L7',1.5,'BC',1);   % v2.7: L4 OFF (DRASTIC out of the inversion); data terms UP
cfg.Nc       = 800;         % collocation points per epoch
cfg.nEpochs  = 2500;        % Adam epochs (with lr decay; raise only if loss still falling)
cfg.lr       = 1e-3;
cfg.clip     = 1.0;         % global gradient-norm clipping (PINN stability)
cfg.forceCPU = false;                % safety switch: set true to disable GPU
cfg.useGPU   = ~cfg.forceCPU && gpuDeviceCount > 0;   % automatic GPU use when available
cfg.dropout  = 0.10;        % MC-dropout rate
cfg.hidden   = 64; cfg.nLayers = 4;
cfg.mFourier = 32; cfg.sigmaF = 2.0;           % Fourier features
cfg.T_mc     = 50;          % MC-dropout passes (sigma error ~1/sqrt(T); bootstrap carries final UQ)
cfg.prec     = 'single';    % FP32: PINN-standard; FP64 equivalence archived via verify_precision
cfg.patience = 300; cfg.esTol = 0.01; cfg.minEp = 1000;   % convergence-plateau early stopping
cfg.idxCoast = [];                          % v2.2: NO Dirichlet (confined aquifer)
cfg.idxSea   = [60 860];                    % Adriatic confined shoreline -> no-flow (declared)
cfg.idxFree  = [1 59; 861 3411];           % v2.6: ALL land FREE; no-flow ONLY at sea [60 860] % FREE: Mat NE entries + Droja/Ishem SE valley mouths
cfg.printEvery = 50;        % console progress frequency
cfg.poolSize   = 40000;     % precomputed collocation pool (isinterior runs ONCE, not per epoch)
cfg.Ld = 3.0; cfg.W0 = 2.0; cfg.Rw = 3.0;                % v2.6 geo-prior scales (km)
cfg.chanFile = '../data/channel_points_merged.csv';
cfg.rFacies  = 3.0;     % km, facies-prior support radius (Sec. 3.3)
cfg.bcSubD = 2; cfg.bcSubN = 5;   % boundary subsampling (409->~205 Dirichlet, 3002->~600 Neumann)
cfg.sampleDecim = 4;        % boundary decimation for SAMPLING-ONLY polygon (memory/speed; full polygon used elsewhere)

%% ---------------- Load canonical data ----------------
D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
G  = readtable(fullfile('..','data','drastic_grid_180.csv'));
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
Th = readtable(fullfile('..','data','orientation_field_grid.csv'));
assert(height(D)==37, 'Canonical well set must have 37 wells.');

% Work in km offsets (x_km, y_km already in files)
xw = [D.x_km, D.y_km]';                 % 2×37 wells
Kobs = D.K_mday;  hobs = D.head_m;  fac = string(D.facies);
xb = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';   % 2×Nb boundary
% De-duplicate digitized boundary (consecutive vertices closer than 0.1 m)
dseg = hypot(diff(xb(1,:)), diff(xb(2,:)));
keep = [true, dseg > 1e-4];
fprintf('Boundary vertices: %d -> %d after de-duplication.\n', size(xb,2), nnz(keep));
xb = xb(:, keep);
xd = [ (G.X_GK  - min(Bd.X_GK))/1000, (G.Y_GK  - min(Bd.Y_GK))/1000 ]';  % 2×180 DRASTIC
Ddr = G.DRASTIC;

% Output scalings (non-dimensionalization; PINN stability)
cfg.hMu = mean(hobs);  cfg.hSd = std(hobs);     % head: h = hMu + hSd * net
cfg.kMu = mean(log(Kobs));                      % logK = kMu + net  (K/K0 = exp(net))

% Normalization to [-1,1] (PDE residual is therefore non-dimensional; declared)
sc.cx = mean(xb(1,:)); sc.cy = mean(xb(2,:));
sc.s  = max(range(xb(1,:)), range(xb(2,:)))/2;
nrm = @(X) [ (X(1,:)-sc.cx)/sc.s ; (X(2,:)-sc.cy)/sc.s ];

%% ---------------- Boundary segmentation (CONFIRM with figure) ----------------
[bcD, bcN, bcF, bcSea] = boundary_segments(xb, cfg);                 % Dirichlet(empty) / no-flow / FREE / sea-flag
bcDraw = bcD;  bcNraw = bcN;                                          % raw counts (reporting)
bcD = bcD(1:cfg.bcSubD:end);  bcN = bcN(1:cfg.bcSubN:end);   % penalty points: subsampled (speed)
fig = figure('Visible','off'); hold on
plot(xb(1,:), xb(2,:), '-', 'Color',[.75 .75 .75]);
plot(xb(1,setdiff(bcN,bcSea)), xb(2,setdiff(bcN,bcSea)), 'k.', 'MarkerSize',6);
plot(xb(1,bcSea), xb(2,bcSea), 'b.', 'MarkerSize',9);
if ~isempty(bcD), plot(xb(1,bcD), xb(2,bcD), 'c.', 'MarkerSize',9); end
plot(xb(1,bcF), xb(2,bcF), 'r.', 'MarkerSize',9);
plot(xw(1,:), xw(2,:), 'mo', 'MarkerSize',5, 'MarkerFaceColor','m'); axis equal
title({'BLUE = Adriatic confined shoreline (no-flow, h NOT fixed)   RED = FREE inflow (Mat NE + Droja/Ishem SE)','BLACK = no-flow hill contacts - CONFIRM segmentation'});
exportgraphics(fig, fullfile('..','figures','check_boundary_segments.png'), 'Resolution',200);
fprintf('Boundary roles (raw vertices): FREE %d | no-flow %d (of which Adriatic shoreline %d) | Dirichlet %d\n', numel(bcF), numel(bcNraw), numel(bcSea), numel(bcDraw));
fprintf('No-flow penalty points after subsampling: %d. See figures/check_boundary_segments.png\n', numel(bcN));

%% ---------------- Geological fields at fixed points ----------------
thetaF = theta_interp(Th, min(Bd.X_GK), min(Bd.Y_GK));   % axial-safe interpolant, azimuth [deg]
dmm    = dmap_facies(fac);                                % per-well grain diameter [mm] (Table 2)
% Facies-calibrated Kozeny–Carman prefactor (log-space), from the 37 wells:
% classical KC constant overpredicts K in gravels; the coupling constrains the
% STRUCTURAL K–n relation; absolute level is calibrated per facies (declared).
nPrior = D.n_mean;
shapeKC = log( nPrior.^3 ./ (1-nPrior).^2 );
betaF = zeros(3,1); fnames = ["coarse","medium","fine"];
for k=1:3
    m = fac==fnames(k);
    betaF(k) = mean( log(Kobs(m)) - shapeKC(m) );
end
fprintf('KC prefactors (log): coarse %.3f, medium %.3f, fine %.3f\n', betaF);

%% ---------------- Networks ----------------
[params, cfg.Bff] = init_networks(cfg);
if cfg.w.L4 > 0                      % v2.7: only when the DRASTIC term is active
    params.aux.alpha = dlarray(0.0);     % L4 slope a = softplus(alpha) > 0
    params.aux.c     = dlarray(0.0);     % L4 intercept
end
params.aux.betaF = dlarray(betaF);   % KC prefactors (trainable refinement)
params  = dlupdate(@(x) cast(x, cfg.prec), params);   % FP32/FP64 switch
cfg.Bff = cast(cfg.Bff, cfg.prec);
if cfg.useGPU
    fprintf('GPU detected -> training on GPU.\n');
    params = dlupdate(@gpuArray, params);
    cfg.Bff = gpuArray(cfg.Bff);
end

% Pre-pack fixed training tensors
T.xw  = dlarray(nrm(xw));   T.K = dlarray(log(Kobs)'); T.h = dlarray(hobs');
T.xd  = dlarray(nrm(xd));   T.D = dlarray(zscore_(Ddr)');
T.xbD = dlarray(nrm(xb(:,bcD)));
[nx_, ny_] = outward_normals(xb, bcN);
T.xbN = dlarray(nrm(xb(:,bcN))); T.nrmN = dlarray([nx_;ny_]);
T.facW = fac;  T.hsea = cfg.hsea;
fnT = fieldnames(T);
for s_ = 1:numel(fnT)
    if isa(T.(fnT{s_}),'dlarray'), T.(fnT{s_}) = cast(T.(fnT{s_}), cfg.prec); end
end
if cfg.useGPU
    fn = ["xw","K","h","xd","D","xbD","xbN","nrmN"];
    for s = fn, T.(s) = gpuArray(T.(s)); end
end

%% ---------------- Training (Adam) ----------------
avgG=[]; avgS=[]; lossHist = nan(cfg.nEpochs,10);
polyB   = polyshape(xb(1,:), xb(2,:), 'Simplify', true);
polySamp= polyshape(xb(1,1:cfg.sampleDecim:end), xb(2,1:cfg.sampleDecim:end), 'Simplify', true);  % light polygon for sampling
fprintf('Building collocation pool (%d pts, one-time)... ', cfg.poolSize); tPool=tic;
XcPool = sample_collocation(polySamp, cfg.poolSize);
thPool = deg2rad( thetaF(XcPool) );                 % static geometry, computed ONCE
kcPool = facies_pool(XcPool, xw, fac, cfg.rFacies);   % support-limited facies prior (Sec. 3.3)
CH = readtable(cfg.chanFile);                                 % v2.6 channel morphology
chanXY = [CH.x_km, CH.y_km]';
AXT = readtable('../data/river_axes.csv');
axesXY = geo_prior('axes', AXT);
P7 = geo_prior('calib', axesXY, chanXY, xw, log(Kobs), cfg);
fprintf('geo-prior calib: K_far=%.0f K_chan=%.0f m/day  R2=%.2f\n', exp(P7.a), exp(P7.a+P7.b), P7.R2);
G7 = geo_prior('eval', P7, XcPool, xw);
logKpPool = G7.logKp;  w7Pool = G7.w7;
mTh = ~isnan(G7.thLoc);  thPool(mTh) = G7.thLoc(mTh);
XnPool = nrm(XcPool);
fprintf('done (%.1f s).\nTraining...\n', toc(tPool));
tEp1 = tic;
for ep = 1:cfg.nEpochs
    idx  = randperm(cfg.poolSize, cfg.Nc);          % free draw from pool
    B.Xc = dlarray(cast(XnPool(:,idx), cfg.prec));
    B.th = dlarray(cast(thPool(idx)',  cfg.prec));
    if cfg.useGPU, B.Xc = gpuArray(B.Xc); B.th = gpuArray(B.th); end
    B.kc = kcPool(idx)';                            % plain numeric indices (device-agnostic)
    B.kp = dlarray(cast(logKpPool(idx)', cfg.prec));
    B.w7 = dlarray(cast(w7Pool(idx)',    cfg.prec));
    if cfg.useGPU, B.kp = gpuArray(B.kp); B.w7 = gpuArray(B.w7); end
    [loss, grads, parts] = dlfeval(@model_loss, params, cfg, T, B, sc);
    grads = clip_grads(grads, cfg.clip);
    lrNow = cfg.lr * 0.3^( (ep > 0.4*cfg.nEpochs) + (ep > 0.7*cfg.nEpochs) + (ep > 0.9*cfg.nEpochs) );
    [params, avgG, avgS] = adamupdate(params, grads, avgG, avgS, ep, lrNow);
    lossHist(ep,:) = [double(gather(extractdata(loss))), parts];
    if ep==1
        t1=toc(tEp1);
        fprintf('ep 1 done in %.2f s  ->  rough total for %d epochs: %.1f min\n', t1, cfg.nEpochs, t1*cfg.nEpochs/60);
        fprintf('   [VERSION CHECK] scaling active: hMu=%.2f hSd=%.2f kMu=%.2f | clip=%g\n', cfg.hMu, cfg.hSd, cfg.kMu, cfg.clip);
    end
    if isnan(lossHist(ep,1))
        fprintf(2,'NaN at epoch %d - parts: L1 %.2e L2 %.2e L3 %.2e L4 %.2e L5 %.2e L6 %.2e L7 %.2e BCd %.2e BCn %.2e\n', ep, parts);
        error('Training produced NaN. Report the epoch number and the parts line above.');
    end
    if ep<=5 || mod(ep,cfg.printEvery)==0
        fprintf('ep %5d | L=%.3e | L1 %.2e L2 %.2e L3 %.2e L4 %.2e L5 %.2e L6 %.2e BCd %.2e BCn %.2e\n', ...
                 ep, lossHist(ep,1), parts);
    end
    if ep > max(2*cfg.patience, cfg.minEp)
        recent = mean(lossHist(ep-cfg.patience+1:ep,1));
        prev   = mean(lossHist(ep-2*cfg.patience+1:ep-cfg.patience,1));
        if recent > prev*(1 - cfg.esTol)
            fprintf('early stop at epoch %d (plateau %.2f%% over %d ep)\n', ep, 100*(1-recent/prev), cfg.patience);
            break
        end
    end
end
lossHist = lossHist(1:ep,:);                    % trim if early-stopped
writematrix(lossHist, fullfile('..','results','loss_history.csv'));
if cfg.useGPU                                  % save CPU copies (portable .mat)
    params_gpu = params;  cfgBff_gpu = cfg.Bff;
    params = dlupdate(@gather, params);  cfg.Bff = gather(cfg.Bff);
    save(fullfile('..','results','trained_params.mat'), 'params', 'cfg', 'sc');
    params = params_gpu;  cfg.Bff = cfgBff_gpu;  clear params_gpu cfgBff_gpu
else
    save(fullfile('..','results','trained_params.mat'), 'params', 'cfg', 'sc');
end
fprintf('Training done. Parameters saved to results/trained_params.mat\n');
fprintf('Post-processing: building prediction grid... '); tPost=tic;

%% ---------------- MC-Dropout UQ + prediction grid ----------------
[XG, inMask, gx, gy] = pred_grid(polySamp, 400);           % grid over aquifer
fprintf('done (%.1f s). MC-dropout UQ (T=%d) on %d nodes:\n', toc(tPost), cfg.T_mc, size(XG,2));
[muK, sdK, muN, sdN, muH] = mc_dropout(params, cfg, dlarray(nrm(XG)), cfg.T_mc);
fprintf('Writing fields + figures... ');
out = table(XG(1,:)', XG(2,:)', muK', sdK', muN', sdN', muH', ...
      'VariableNames', {'x_km','y_km','logK_mean','logK_sd','n_mean','n_sd','h_mean'});
writetable(out, fullfile('..','results','fields_mcdropout.csv'));

%% ---------------- Single-panel journal figures (no duplication) ----------------
make_map(gx,gy,inMask, muK, xb, 'log K (posterior mean)', 'fig5_Khat.png');
make_map(gx,gy,inMask, muN, xb, 'n (posterior mean)',     'fig6_nhat.png');
make_map(gx,gy,inMask, sdK, xb, '\sigma_{logK} (MC-dropout)', 'figS2_sigma.png');
fprintf('done.\n');

% Well-level fit for Table 5 inputs
predW = mc_dropout(params, cfg, T.xw, cfg.T_mc);
rmseK = sqrt(mean( (predW - log(Kobs)').^2 ));
fprintf('In-sample logK RMSE (diagnostic only; CV is the real metric): %.3f\n', rmseK);

fprintf('\nDONE. Next: 5-fold CV (run main_cv.m), bootstrap ensemble (uq_bootstrap.m), then OED (main_oed_w2.m).\n');

%% ---------------- local helpers ----------------
function z = zscore_(v), z = (v - mean(v))/std(v); end
function f = betaF_nearest(Xc, xw, fac)
    [~,idx] = min( (Xc(1,:)'-xw(1,:)).^2 + (Xc(2,:)'-xw(2,:)).^2, [], 2 );
    key = ["coarse","medium","fine"]; f = zeros(numel(idx),1);
    for k=1:3, f(fac(idx)==key(k)) = k; end
end
