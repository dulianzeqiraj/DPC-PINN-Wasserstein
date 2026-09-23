%% Post-processing ONLY (no retraining): loads results/trained_params.mat
clear; clc; addpath(fullfile(pwd,'src'));
S = load(fullfile('..','results','trained_params.mat'));   % params, cfg, sc
params = S.params; cfg = S.cfg; sc = S.sc;
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
xb = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';
dseg = hypot(diff(xb(1,:)), diff(xb(2,:))); keep = [true, dseg > 1e-4]; xb = xb(:,keep);
nrm = @(X) [ (X(1,:)-sc.cx)/sc.s ; (X(2,:)-sc.cy)/sc.s ];
polySamp = polyshape(xb(1,1:cfg.sampleDecim:end), xb(2,1:cfg.sampleDecim:end), 'Simplify', true);
fprintf('Prediction grid... '); tic
[XG, inMask, gx, gy] = pred_grid(polySamp, 400);
fprintf('done (%.1f s). MC-dropout (T=%d) on %d nodes:\n', toc, cfg.T_mc, size(XG,2));
[muK, sdK, muN, sdN, muH] = mc_dropout(params, cfg, dlarray(nrm(XG)), cfg.T_mc);
out = table(XG(1,:)', XG(2,:)', muK', sdK', muN', sdN', muH', ...
      'VariableNames', {'x_km','y_km','logK_mean','logK_sd','n_mean','n_sd','h_mean'});
writetable(out, fullfile('..','results','fields_mcdropout.csv'));
make_map(gx,gy,inMask, muK, xb, 'log K (posterior mean)', 'fig5_Khat.png');
make_map(gx,gy,inMask, muN, xb, 'n (posterior mean)',     'fig6_nhat.png');
make_map(gx,gy,inMask, sdK, xb, '\sigma_{logK} (MC-dropout)', 'figS2_sigma.png');
fprintf('Figures + CSV written.\n');
