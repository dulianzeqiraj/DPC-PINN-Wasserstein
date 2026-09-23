%% DPC-PINN - 5-fold cross-validation on the 37 wells (Fushë-Kuqe)
%  Fixed seed/folds. Per-fold scalings, KC prefactors, facies prior and geo-prior calibration
%  from TRAIN wells only (the last two since v2.8, via src/fold_priors.m).
%  Outputs: results/cv_results.csv (per-well) + console summary (Table 5 inputs).
clear; clc; close all; addpath(fullfile(pwd,'src'));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');
rng(20260610,'twister');

%% config (mirrors main_fushekuqe; fewer epochs per fold)
cfg.b=40; cfg.N=0; cfg.hsea=0; cfg.R5=4.0; cfg.nMin=0.20; cfg.nMax=0.40;
cfg.w = struct('L1',1,'L2',10,'L3',10,'L4',0,'L5',0.05,'L6',0.5,'L7',1.5,'BC',1);   % v2.7: L4 OFF
cfg.Nc=800; cfg.nEpochs=1800; cfg.lr=1e-3; cfg.clip=1.0;
cfg.dropout=0.10; cfg.hidden=64; cfg.nLayers=4; cfg.mFourier=32; cfg.sigmaF=2.0;
cfg.T_mc = 30;          % MC passes (sigma error ~ 1/sqrt(T); bootstrap carries final UQ)
cfg.printEvery = 300;
cfg.idxCoast = [];                          % v2.2: NO Dirichlet (confined aquifer)
cfg.idxSea   = [60 860];                    % Adriatic confined shoreline -> no-flow
cfg.idxFree  = [1 59; 861 3411];           % v2.6: ALL land FREE; no-flow ONLY at sea [60 860] % FREE: Mat NE + Droja/Ishem SE
cfg.poolSize=40000; cfg.bcSubD=2; cfg.bcSubN=5; cfg.sampleDecim=4;
cfg.Ld = 3.0; cfg.W0 = 2.0; cfg.Rw = 3.0;                % v2.6 geo-prior scales (km)
cfg.chanFile = '../data/channel_points_merged.csv';
cfg.rFacies = 3.0;    % km, facies-prior support radius (Sec. 3.3)
cfg.prec = 'single';      % FP32: PINN-standard precision; equivalence vs FP64 archived (verify_precision)
cfg.patience = 300; cfg.esTol = 0.01; cfg.minEp = 1000;   % convergence-plateau early stopping
cfg.forceCPU = false;                % safety switch: set true to disable GPU
cfg.useGPU = ~cfg.forceCPU && gpuDeviceCount > 0;
if cfg.useGPU, fprintf('GPU detected -> folds train on GPU.\n'); end

%% data prep (as in main)
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

[bcD,bcN,bcF] = boundary_segments(xb, cfg);
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
DAT.kcPool = facies_pool(XcPool, xw, DAT.fac, cfg.rFacies);   % support-limited facies prior
CH = readtable(cfg.chanFile);                                 % v2.6 channel morphology
chanXY = [CH.x_km, CH.y_km]';
AXT = readtable('../data/river_axes.csv');
axesXY = geo_prior('axes', AXT);
P7 = geo_prior('calib', axesXY, chanXY, xw, log(DAT.Kobs), cfg);
fprintf('geo-prior calib: K_far=%.0f K_chan=%.0f m/day  R2=%.2f\n', exp(P7.a), exp(P7.a+P7.b), P7.R2);
G7 = geo_prior('eval', P7, XcPool, xw);
DAT.logKpPool = G7.logKp;  DAT.w7Pool = G7.w7;
mTh = ~isnan(G7.thLoc);  DAT.thPool(mTh) = G7.thLoc(mTh);     % local channel orientation where dense
DAT.XnPool = nrm(XcPool);
fprintf('done (%.1f s).\n', toc);

%% fixed folds (seeded)
perm = randperm(37);
foldID = zeros(37,1); foldID(perm) = mod(0:36,5)+1;
writematrix(foldID, fullfile('..','results','cv_folds.csv'));

%% CV loop
%  v2.7: one checkpoint per fold, plus a fold-local seed.
%  The checkpoint lets an interrupted run resume instead of starting over (the 12 Sept run
%  died in fold 3 and lost two completed folds). The fold-local seed makes a fold give the
%  same answer whether it runs first, last, or after a resume; without it, resuming would
%  silently change the random stream. The fold ASSIGNMENT above is untouched, so
%  main_ablation.m still shares the same split. Delete results/cv_ckpt to force a full redo.
SEED0 = 20260610;
ckDir = fullfile('..','results','cv_ckpt');
if ~exist(ckDir,'dir'), mkdir(ckDir); end
tCV = tic;
for f = 1:5
    fck = fullfile(ckDir, sprintf('fold_%d.mat', f));
    if isfile(fck)
        fprintf('FOLD %d/5: checkpoint exists, skipped.\n', f);
        continue
    end
    wTest = find(foldID==f); wTrain = find(foldID~=f);
    fprintf('\n=== FOLD %d/5: train %d, test %d wells ===\n', f, numel(wTrain), numel(wTest));
    DATf = fold_priors(DAT, XcPool, xw, axesXY, chanXY, wTrain, cfg);   % v2.8: no held-out well in the priors
    rng(SEED0 + f, 'twister');
    [params, cfgF] = train_dpcpinn(cfg, DATf, wTrain, sprintf('F%d',f));
    [muK, sdK, ~, ~, muH, sdH] = mc_dropout(params, cfgF, dlarray(DAT.xwN(:,wTest)), cfgF.T_mc);
    ids = string(D.Well_ID(wTest));
    lgK = log(DAT.Kobs(wTest));
    hob = DAT.hobs(wTest);
    save(fck, 'f','wTest','ids','lgK','hob','muK','sdK','muH','sdH');
    fprintf('  saved %s\n', fck);
end

rows = [];
for f = 1:5
    S = load(fullfile(ckDir, sprintf('fold_%d.mat', f)));
    for j = 1:numel(S.ids)
        rows = [rows; {S.f, S.ids(j), S.lgK(j), double(S.muK(j)), double(S.sdK(j)), ...
                       S.hob(j), double(S.muH(j)), double(S.sdH(j)), ...
                       double(abs(S.lgK(j)-double(S.muK(j))) <= 1.645*double(S.sdK(j))), ...
                       double(abs(S.hob(j)-double(S.muH(j))) <= 1.645*double(S.sdH(j)))}]; %#ok<AGROW>
    end
end
R = cell2table(rows, 'VariableNames', ...
   {'fold','Well_ID','logK_obs','logK_pred','logK_sd','h_obs','h_pred','h_sd','cov90_K','cov90_h'});
writetable(R, fullfile('..','results','cv_results.csv'));

%% summary (Table 5 inputs)
eK = R.logK_pred - R.logK_obs;  eH = R.h_pred - R.h_obs;
fprintf('\n================ 5-FOLD CV SUMMARY (n=37 held-out predictions) ================\n');
fprintf('logK : RMSE = %.3f   MAE = %.3f   (factor = %.2f)\n', sqrt(mean(eK.^2)), mean(abs(eK)), exp(sqrt(mean(eK.^2))));
fprintf('head : RMSE = %.2f m  MAE = %.2f m\n', sqrt(mean(eH.^2)), mean(abs(eH)));
fprintf('90%% interval coverage:  logK %.0f%%   head %.0f%%   (nominal 90%%)\n', 100*mean(R.cov90_K), 100*mean(R.cov90_h));
fprintf('Total CV wall time: %.1f min\n', toc(tCV)/60);
fprintf('Per-well results written to results/cv_results.csv\n');
