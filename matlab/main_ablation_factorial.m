%% DPC-PINN - FULL 2^3 FACTORIAL ABLATION over the geological/petrophysical priors
%
%  Referee objection this answers (Reviewer 1, point 17):
%    "Losses L5-L7 are activated together. The reported improvement from 0.409 to 0.357
%     therefore cannot be assigned separately to anisotropy, Kozeny-Carman coupling, or
%     channel geometry."
%
%  Design: all eight subsets of {L5, L6, L7} at their canonical weights, each under the
%  SAME 5-fold split as main_cv.m. That gives every main effect and every interaction,
%  not just a one-at-a-time sequence. A ninth mode, base+L4, is kept as the evidence that
%  the DRASTIC term removed in v2.7 was worth nothing.
%
%  Reproducibility: every (mode, fold) cell is seeded independently as
%     rng(SEED0 + 1000*modeIdx + foldIdx)
%  so a cell is identical whether it is run first, last, or after a resume. The price is
%  that the 'full' row is an INDEPENDENT REPLICATE of the canonical main_cv run rather
%  than a bit-for-bit copy of it. That is deliberate: the gap between them is a free
%  estimate of run-to-run spread, which is what Reviewer 1 point 15 asks for when it
%  challenges "statistically indistinguishable".
%
%  Checkpoint/resume: results/ablation_fac/<tag>_f<k>.mat. Existing cells are skipped,
%  so the run can be interrupted and relaunched. Delete the folder to force a full redo.
%
%  Outputs: results/ablation_factorial_wells.csv  (per held-out well)
%           results/ablation_factorial_rows.csv   (per mode)
%           results/ablation_factorial_effects.csv(main effects and interactions)

clear; clc; close all; addpath(fullfile(pwd,'src'));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');
rng(20260610,'twister');

SEED0 = 20260610;

%% config (mirrors main_cv.m exactly; only the L5/L6/L7 weights vary per mode)
cfg.b=40; cfg.N=0; cfg.hsea=0; cfg.R5=4.0; cfg.nMin=0.20; cfg.nMax=0.40;
cfg.w = struct('L1',1,'L2',10,'L3',10,'L4',0,'L5',0.05,'L6',0.5,'L7',1.5,'BC',1);
cfg.Nc=800; cfg.nEpochs=1800; cfg.lr=1e-3; cfg.clip=1.0;
cfg.dropout=0.10; cfg.hidden=64; cfg.nLayers=4; cfg.mFourier=32; cfg.sigmaF=2.0;
cfg.T_mc = 30;
cfg.printEvery = 600;
cfg.idxCoast = [];
cfg.idxSea   = [60 860];
cfg.idxFree  = [1 59; 861 3411];
cfg.poolSize=40000; cfg.bcSubD=2; cfg.bcSubN=5; cfg.sampleDecim=4;
cfg.Ld = 3.0; cfg.W0 = 2.0; cfg.Rw = 3.0;
cfg.chanFile = '../data/channel_points_merged.csv';
cfg.rFacies = 3.0;
cfg.prec = 'single';
cfg.patience = 300; cfg.esTol = 0.01; cfg.minEp = 1000;
cfg.forceCPU = false;
cfg.useGPU = ~cfg.forceCPU && gpuDeviceCount > 0;
if cfg.useGPU, fprintf('GPU detected -> folds train on GPU.\n'); end

%% data prep (verbatim from main_cv.m, so the fold split is the same one)
D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
G  = readtable(fullfile('..','data','drastic_grid_180.csv'));
Th = readtable(fullfile('..','data','orientation_field_grid.csv'));
xw = [D.x_km, D.y_km]';
DAT.Kobs = D.K_mday; DAT.hobs = D.head_m; DAT.fac = string(D.facies); DAT.nPrior = D.n_mean;
xb = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';
dseg = hypot(diff(xb(1,:)), diff(xb(2,:))); xb = xb(:, [true, dseg>1e-4]);
xd = [ (G.X_GK - min(Bd.X_GK))/1000, (G.Y_GK - min(Bd.Y_GK))/1000 ]';
sc.cx = mean(xb(1,:)); sc.cy = mean(xb(2,:)); sc.s = max(range(xb(1,:)),range(xb(2,:)))/2;
nrm = @(X) [ (X(1,:)-sc.cx)/sc.s ; (X(2,:)-sc.cy)/sc.s ];

[bcD,bcN,bcF] = boundary_segments(xb, cfg); %#ok<ASGLU>
bcD=bcD(1:cfg.bcSubD:end); bcN=bcN(1:cfg.bcSubN:end);
[nx_,ny_] = outward_normals(xb, bcN);
DAT.Tfix.xd  = dlarray(nrm(xd));
DAT.Tfix.D   = dlarray( ((G.DRASTIC-mean(G.DRASTIC))/std(G.DRASTIC))' );
DAT.Tfix.xbD = dlarray(nrm(xb(:,bcD)));
DAT.Tfix.xbN = dlarray(nrm(xb(:,bcN)));
DAT.Tfix.nrmN= dlarray([nx_;ny_]);
DAT.Tfix.facW= DAT.fac;
if cfg.useGPU
    for s = ["xd","D","xbD","xbN","nrmN"], DAT.Tfix.(s)=gpuArray(DAT.Tfix.(s)); end
end
DAT.xwN = nrm(xw);

polySamp = polyshape(xb(1,1:cfg.sampleDecim:end), xb(2,1:cfg.sampleDecim:end),'Simplify',true);
fprintf('Collocation pool... '); tic
XcPool = sample_collocation(polySamp, cfg.poolSize);
thetaF = theta_interp(Th, min(Bd.X_GK), min(Bd.Y_GK));
DAT.thPool = deg2rad( thetaF(XcPool) );
DAT.kcPool = facies_pool(XcPool, xw, DAT.fac, cfg.rFacies);
CH = readtable(cfg.chanFile);
chanXY = [CH.x_km, CH.y_km]';
AXT = readtable('../data/river_axes.csv');
axesXY = geo_prior('axes', AXT);
P7 = geo_prior('calib', axesXY, chanXY, xw, log(DAT.Kobs), cfg);
fprintf('geo-prior calib: K_far=%.0f K_chan=%.0f m/day  R2=%.2f\n', exp(P7.a), exp(P7.a+P7.b), P7.R2);
G7 = geo_prior('eval', P7, XcPool, xw);
DAT.logKpPool = G7.logKp;  DAT.w7Pool = G7.w7;
mTh = ~isnan(G7.thLoc);  DAT.thPool(mTh) = G7.thLoc(mTh);
DAT.XnPool = nrm(XcPool);
fprintf('done (%.1f s).\n', toc);

%% same fold split as main_cv.m
perm = randperm(37);
foldID = zeros(37,1); foldID(perm) = mod(0:36,5)+1;
fprintf('Fold sizes: %s\n', mat2str(accumarray(foldID,1)'));

%% the nine modes
W5 = 0.05; W6 = 0.5; W7 = 1.5; W4 = 0.1;
names = {'base','+L5','+L6','+L7','+L5+L6','+L5+L7','+L6+L7','full','base+L4'};
tags  = {'base','L5','L6','L7','L5L6','L5L7','L6L7','full','baseL4'};
on5   = [0 1 0 0 1 1 0 1 0];
on6   = [0 0 1 0 1 0 1 1 0];
on7   = [0 0 0 1 0 1 1 1 0];
on4   = [0 0 0 0 0 0 0 0 1];

ckDir = fullfile('..','results','ablation_fac');
if ~exist(ckDir,'dir'), mkdir(ckDir); end

nM = numel(names);
fprintf('\n%d modes x 5 folds = %d cells. Existing checkpoints are skipped.\n', nM, 5*nM);
tAll = tic;

for m = 1:nM
    cfg.w.L4 = on4(m)*W4;
    cfg.w.L5 = on5(m)*W5;
    cfg.w.L6 = on6(m)*W6;
    cfg.w.L7 = on7(m)*W7;
    fprintf('\n################ MODE %d/%d: %s  (L4=%.2g L5=%.2g L6=%.2g L7=%.2g) ################\n', ...
            m, nM, names{m}, cfg.w.L4, cfg.w.L5, cfg.w.L6, cfg.w.L7);
    for f = 1:5
        fck = fullfile(ckDir, sprintf('%s_f%d.mat', tags{m}, f));
        if isfile(fck)
            fprintf('  %s fold %d: checkpoint exists, skipped.\n', names{m}, f);
            continue
        end
        wTest = find(foldID==f); wTrain = find(foldID~=f);
        fprintf('\n=== %s FOLD %d/5: train %d, test %d ===\n', names{m}, f, numel(wTrain), numel(wTest));
        DATf = fold_priors(DAT, XcPool, xw, axesXY, chanXY, wTrain, cfg);   % v2.8: as main_cv.m
        rng(SEED0 + 1000*m + f, 'twister');          % cell-local seed: order-independent
        [params, cfgF] = train_dpcpinn(cfg, DATf, wTrain, sprintf('%s-F%d', names{m}, f));
        [muK, sdK, ~, ~, muH, sdH] = mc_dropout(params, cfgF, dlarray(DAT.xwN(:,wTest)), cfgF.T_mc);
        cell_mode = names{m};
        cell_wells = string(D.Well_ID(wTest));
        cell_logKobs = log(DAT.Kobs(wTest));
        cell_hobs = DAT.hobs(wTest);
        save(fck, 'cell_mode','f','cell_wells','cell_logKobs','cell_hobs', ...
                  'muK','sdK','muH','sdH');
        fprintf('  saved %s\n', fck);
    end
end
fprintf('\nAll cells present after %.1f min.\n', toc(tAll)/60);

%% ---------------- aggregate ----------------
allRows = {};
for m = 1:nM
    for f = 1:5
        S = load(fullfile(ckDir, sprintf('%s_f%d.mat', tags{m}, f)));
        for j = 1:numel(S.cell_wells)
            allRows(end+1,:) = { string(names{m}), f, S.cell_wells(j), ...
                S.cell_logKobs(j), double(S.muK(j)), double(S.sdK(j)), ...
                S.cell_hobs(j),   double(S.muH(j)), double(S.sdH(j)), ...
                double(abs(S.cell_logKobs(j)-double(S.muK(j))) <= 1.645*double(S.sdK(j))), ...
                double(abs(S.cell_hobs(j)  -double(S.muH(j))) <= 1.645*double(S.sdH(j))) }; %#ok<AGROW>
        end
    end
end
Wt = cell2table(allRows, 'VariableNames', {'mode','fold','Well_ID', ...
     'logK_obs','logK_pred','logK_sd','h_obs','h_pred','h_sd','cov90_K','cov90_h'});
writetable(Wt, fullfile('..','results','ablation_factorial_wells.csv'));

summ = {};
rmseByMode = zeros(nM,1);
for m = 1:nM
    sel = Wt.mode == string(names{m});
    eK = Wt.logK_pred(sel) - Wt.logK_obs(sel);
    eH = Wt.h_pred(sel)    - Wt.h_obs(sel);
    r  = sqrt(mean(eK.^2));
    rmseByMode(m) = r;
    summ(end+1,:) = { string(names{m}), on5(m), on6(m), on7(m), on4(m), r, mean(abs(eK)), exp(r), ...
                      100*mean(Wt.cov90_K(sel)), sqrt(mean(eH.^2)), mean(abs(eH)), ...
                      100*mean(Wt.cov90_h(sel)) }; %#ok<AGROW>
end
Rt = cell2table(summ, 'VariableNames', {'mode','L5','L6','L7','L4', ...
     'logK_RMSE','logK_MAE','factor','cov90_K_pct','head_RMSE','head_MAE','cov90_h_pct'});
writetable(Rt, fullfile('..','results','ablation_factorial_rows.csv'));

fprintf('\n================ FACTORIAL ABLATION (logK, n=37 held out per mode) ================\n');
disp(Rt);

%% ---------------- main effects and interactions over the 2^3 block ----------------
fac = 1:8;                                  % the eight subsets of {L5,L6,L7}
sgn = @(v) 2*v(fac)' - 1;                   % -1 when off, +1 when on
y   = rmseByMode(fac);
s5 = sgn(on5); s6 = sgn(on6); s7 = sgn(on7);
eff = {};
eff(end+1,:) = {'L5 (anisotropy)',        mean(y.*s5)*2};
eff(end+1,:) = {'L6 (Kozeny-Carman)',     mean(y.*s6)*2};
eff(end+1,:) = {'L7 (channel geometry)',  mean(y.*s7)*2};
eff(end+1,:) = {'L5 x L6',                mean(y.*s5.*s6)*2};
eff(end+1,:) = {'L5 x L7',                mean(y.*s5.*s7)*2};
eff(end+1,:) = {'L6 x L7',                mean(y.*s6.*s7)*2};
eff(end+1,:) = {'L5 x L6 x L7',           mean(y.*s5.*s6.*s7)*2};
Et = cell2table(eff, 'VariableNames', {'effect','delta_logK_RMSE'});
writetable(Et, fullfile('..','results','ablation_factorial_effects.csv'));

fprintf('\n---- main effects and interactions on held-out logK RMSE ----\n');
fprintf('(negative = switching the term ON reduces the error)\n');
for i = 1:height(Et)
    fprintf('  %-24s %+0.4f\n', Et.effect{i}, Et.delta_logK_RMSE(i));
end
fprintf('\nbase = %.4f   full = %.4f   difference = %+0.4f\n', ...
        rmseByMode(1), rmseByMode(8), rmseByMode(8)-rmseByMode(1));
fprintf('DRASTIC evidence row: base+L4 = %.4f  (base = %.4f, change %+0.4f)\n', ...
        rmseByMode(9), rmseByMode(1), rmseByMode(9)-rmseByMode(1));
fprintf('\nWritten: results/ablation_factorial_wells.csv, _rows.csv, _effects.csv\n');
