%% DPC-PINN - ABLATION on the 37 wells (Fushe-Kuqe): PINN-base and PINN+DRASTIC
%  Clone of main_cv.m (identical seed/folds/prep/metrics); ONLY loss weights differ.
%  Produces the two missing rows of Table 5 (R3). Single precision, seed 20260610.
%  Outputs: results/ablation_rows.csv (+ per-well results/ablation_wells.csv).
clear; clc; close all; addpath(fullfile(pwd,'src'));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');
rng(20260610,'twister');

%% config (mirrors main_fushekuqe; fewer epochs per fold)
cfg.b=40; cfg.N=0; cfg.hsea=0; cfg.R5=4.0; cfg.nMin=0.20; cfg.nMax=0.40;
% cfg.w is set per ablation mode inside the loop (see 'modes' below)
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


%% ablation modes (only weights change).
%  v2.7: the canonical model no longer contains L4, so the DRASTIC row is retained
%  ONLY as evidence for the referee that dropping it costs nothing.
modes = struct( ...
  'name', {'PINN-base','PINN+DRASTIC'}, ...
  'w',    { struct('L1',1,'L2',10,'L3',10,'L4',0,  'L5',0,'L6',0,'L7',0,'BC',1), ...
            struct('L1',1,'L2',10,'L3',10,'L4',0.1,'L5',0,'L6',0,'L7',0,'BC',1) } );

%% fixed folds (seeded) - IDENTICAL consumption to main_cv (one randperm right here)
perm = randperm(37);
foldID = zeros(37,1); foldID(perm) = mod(0:36,5)+1;

allrows = [];
summary = [];
for m = 1:numel(modes)
    cfg.w = modes(m).w;
    fprintf('\n################ ABLATION MODE: %s ################\n', modes(m).name);
    rows = [];
    tCV = tic;
    for f = 1:5
        wTest = find(foldID==f); wTrain = find(foldID~=f);
        fprintf('\n=== %s FOLD %d/5: train %d, test %d ===\n', modes(m).name, f, numel(wTrain), numel(wTest));
        [params, cfgF] = train_dpcpinn(cfg, DAT, wTrain, sprintf('%s-F%d',modes(m).name,f));
        [muK, sdK, ~, ~, muH, sdH] = mc_dropout(params, cfgF, dlarray(DAT.xwN(:,wTest)), cfgF.T_mc);
        for j = 1:numel(wTest)
            i = wTest(j);
            rows = [rows; {modes(m).name, f, string(D.Well_ID{i}), log(DAT.Kobs(i)), muK(j), sdK(j), ...
                           DAT.hobs(i), muH(j), sdH(j), ...
                           double(abs(log(DAT.Kobs(i))-muK(j)) <= 1.645*sdK(j))}]; %#ok<AGROW>
        end
    end
    Rm = cell2table(rows, 'VariableNames', ...
       {'mode','fold','Well_ID','logK_obs','logK_pred','logK_sd','h_obs','h_pred','h_sd','cov90_K'});
    allrows = [allrows; Rm]; %#ok<AGROW>
    eK = Rm.logK_pred - Rm.logK_obs;
    rmse = sqrt(mean(eK.^2)); mae = mean(abs(eK)); fac = exp(rmse); cov = 100*mean(Rm.cov90_K);
    fprintf('\n---- %s: logK RMSE=%.3f  MAE=%.3f  factor=%.2f  cov90=%.0f%%  (%.1f min) ----\n', ...
            modes(m).name, rmse, mae, fac, cov, toc(tCV)/60);
    summary = [summary; {modes(m).name, rmse, mae, fac, cov}]; %#ok<AGROW>
end

writetable(allrows, fullfile('..','results','ablation_wells.csv'));
T = cell2table(summary, 'VariableNames', {'method','RMSE','MAE','factor','cov90_pct'});
writetable(T, fullfile('..','results','ablation_rows.csv'));
fprintf('\n================ ABLATION SUMMARY (Table 5 rows, R3) ================\n');
disp(T);
fprintf('Written: results/ablation_rows.csv  and  results/ablation_wells.csv\n');
