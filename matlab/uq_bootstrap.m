%% Bootstrap-ensemble UQ with out-of-bag (OOB) coverage validation
%  Member m trains on wTrain = randi(37,37,1): resampling WITH replacement.
%  train_dpcpinn indexes wells by wTrain, so duplicates act as natural
%  weights in the data-misfit means - no core changes needed.
%  Wells NOT drawn by member m form its OOB set (~13 wells on average).
%  Aggregating OOB predictions across members yields honest out-of-sample
%  intervals: sigma_total^2 = var_between_members + mean(MC-dropout var).
%  Checkpoint/resume: results/boot/member_XX.mat; existing members are
%  skipped, so the run can be interrupted and relaunched freely. To grow
%  the ensemble later, just raise M_BOOT and rerun (only new members train).
clear; clc; close all; addpath(fullfile(pwd,'src'));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');
rng(20260610,'twister');

M_BOOT = 30;            % ensemble size (resume-friendly: can be raised later)
SEED0  = 20260610;      % member m uses rng(SEED0 + m) for its resample

%% config (mirrors main_cv exactly)
cfg.b=40; cfg.N=0; cfg.hsea=0; cfg.R5=4.0; cfg.nMin=0.20; cfg.nMax=0.40;
cfg.w = struct('L1',1,'L2',10,'L3',10,'L4',0,'L5',0.05,'L6',0.5,'L7',1.5,'BC',1);   % v2.7: L4 OFF
cfg.Nc=800; cfg.nEpochs=1800; cfg.lr=1e-3; cfg.clip=1.0;
cfg.dropout=0.10; cfg.hidden=64; cfg.nLayers=4; cfg.mFourier=32; cfg.sigmaF=2.0;
cfg.T_mc=30; cfg.printEvery=600;
cfg.idxCoast = [];                          % no Dirichlet (confined aquifer)
cfg.idxSea   = [60 860];                    % Adriatic confined shoreline -> no-flow
cfg.idxFree  = [1 59; 861 3411];           % v2.6: ALL land FREE; no-flow ONLY at sea [60 860] % FREE: Mat NE + Droja/Ishem SE
cfg.poolSize=40000; cfg.bcSubD=2; cfg.bcSubN=5; cfg.sampleDecim=4;
cfg.Ld = 3.0; cfg.W0 = 2.0; cfg.Rw = 3.0;                % v2.6 geo-prior scales (km)
cfg.chanFile = '../data/channel_points_merged.csv';
cfg.rFacies = 3.0;    % km, facies-prior support radius (Sec. 3.3)
cfg.prec = 'single';      % FP32: PINN-standard precision; equivalence vs FP64 archived (verify_precision)
cfg.patience = 300; cfg.esTol = 0.01; cfg.minEp = 1000;   % convergence-plateau early stopping
cfg.forceCPU = false;
cfg.useGPU = ~cfg.forceCPU && gpuDeviceCount > 0;
USE_PARFOR = ~cfg.useGPU && ~isempty(which('parpool'));
if cfg.useGPU
    fprintf('GPU detected -> members train sequentially on GPU.\n');
elseif USE_PARFOR
    fprintf('CPU mode -> parfor over members.\n');
end

%% data prep (verbatim from main_cv)
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

if ~exist(fullfile('..','results','boot'),'dir'), mkdir(fullfile('..','results','boot')); end

%% member loop (checkpoint/resume; parfor only in CPU mode)
fprintf('Bootstrap: M=%d members x %d epochs. Existing checkpoints are skipped.\n', M_BOOT, cfg.nEpochs);
tAll = tic;
GP = struct('XcPool', XcPool, 'xw', xw, 'axesXY', axesXY, 'chanXY', chanXY);
if USE_PARFOR
    parfor m = 1:M_BOOT
        run_member(m, SEED0, cfg, DAT, GP);
    end
else
    for m = 1:M_BOOT
        run_member(m, SEED0, cfg, DAT, GP);
    end
end
fprintf('All members present after %.1f min.\n', toc(tAll)/60);

%% aggregation: production stats + honest OOB validation
mkK=nan(37,M_BOOT); mkH=nan(37,M_BOOT); msK=nan(37,M_BOOT); msH=nan(37,M_BOOT);
oob=false(37,M_BOOT);
for m = 1:M_BOOT
    S = load(fullfile('..','results','boot',sprintf('member_%02d.mat',m)));
    mkK(:,m)=S.muK; mkH(:,m)=S.muH; msK(:,m)=S.sdK; msH(:,m)=S.sdH; oob(S.oobIdx,m)=true;
end
lgK = log(DAT.Kobs); hob = DAT.hobs;
muK = mean(mkK,2);  sBK = std(mkK,0,2);  sMK = sqrt(mean(msK.^2,2));  sTK = sqrt(sBK.^2+sMK.^2);
muH = mean(mkH,2);  sBH = std(mkH,0,2);  sMH = sqrt(mean(msH.^2,2));  sTH = sqrt(sBH.^2+sMH.^2);
muKo=nan(37,1); sKo=nan(37,1); muHo=nan(37,1); sHo=nan(37,1); nO=zeros(37,1);
for i=1:37
    j=oob(i,:); nO(i)=nnz(j);
    if nO(i)>=3
        muKo(i)=mean(mkK(i,j)); sKo(i)=sqrt(var(mkK(i,j),0)+mean(msK(i,j).^2));
        muHo(i)=mean(mkH(i,j)); sHo(i)=sqrt(var(mkH(i,j),0)+mean(msH(i,j).^2));
    end
end
v = nO>=3;
eK = muKo(v)-lgK(v); eH = muHo(v)-hob(v);
fprintf('\n========== BOOTSTRAP OOB VALIDATION (n=%d wells with >=3 OOB members) ==========\n', nnz(v));
fprintf('logK : OOB RMSE = %.3f  MAE = %.3f | 90%% coverage (sigma_total) = %.0f%%\n', ...
        sqrt(mean(eK.^2)), mean(abs(eK)), 100*mean(abs(eK)<=1.645*sKo(v)));
fprintf('head : OOB RMSE = %.2f m MAE = %.2f m | 90%% coverage (sigma_total) = %.0f%%\n', ...
        sqrt(mean(eH.^2)), mean(abs(eH)), 100*mean(abs(eH)<=1.645*sHo(v)));
fprintf('sigma ratio (median over wells): sigma_boot/sigma_MC = %.1f (logK), %.1f (head)\n', ...
        median(sBK./sMK), median(sBH./sMH));
T = table(string(D.Well_ID), lgK, hob, muK, sTK, muH, sTH, muKo, sKo, muHo, sHo, nO, ...
  'VariableNames', {'Well_ID','logK_obs','h_obs','logK_mean','logK_sd_tot','h_mean','h_sd_tot', ...
                    'logK_oob','logK_oob_sd','h_oob','h_oob_sd','n_oob'});
writetable(T, fullfile('..','results','uq_bootstrap_wells.csv'));
fprintf('Per-well results written to results/uq_bootstrap_wells.csv\n');

%% ---------------- local function ----------------
function run_member(m, SEED0, cfg, DAT, GP)
fout = fullfile('..','results','boot', sprintf('member_%02d.mat', m));
if isfile(fout), fprintf('member %02d: checkpoint exists, skipped.\n', m); return; end
rng(SEED0 + m, 'twister');
wTrain = randi(37, 37, 1);                       % bootstrap resample (with replacement)
oobIdx = setdiff(1:37, unique(wTrain));
% v2.8: facies and geo-prior from the resample only, so the OOB wells stay out of the priors
DAT = fold_priors(DAT, GP.XcPool, GP.xw, GP.axesXY, GP.chanXY, wTrain, cfg);
[params, cfgM] = train_dpcpinn(cfg, DAT, wTrain, sprintf('B%02d', m));
[muK, sdK, ~, ~, muH, sdH] = mc_dropout(params, cfgM, dlarray(DAT.xwN), cfg.T_mc);
muK=double(muK(:)); sdK=double(sdK(:)); muH=double(muH(:)); sdH=double(sdH(:));
save(fout, 'muK','sdK','muH','sdH','oobIdx','wTrain');
fprintf('member %02d done (OOB n=%d).\n', m, numel(oobIdx));
end
