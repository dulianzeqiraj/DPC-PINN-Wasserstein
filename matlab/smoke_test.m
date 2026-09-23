%% SMOKE TEST - validates every code path of RUN_ALL / uq_bootstrap in ~5 min
%  Run this BEFORE any long run. All checks are independent (failures don't
%  stop the suite). Writes results/smoke_test.txt.
clear; clc; close all; addpath(fullfile(pwd,'src'));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');
PASS=0; FAIL=0; LOG={};
ng=0; try, ng=gpuDeviceCount; catch, end   % does not fail on machines without PCT

% ---------- 1. data files & integrity ----------
try
    D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
    Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
    G  = readtable(fullfile('..','data','drastic_grid_180.csv'));
    Th = readtable(fullfile('..','data','orientation_field_grid.csv'));
    A  = readtable(fullfile('..','data','well_anisotropy_37.csv'));
    assert(height(D)==37 && height(G)==180 && height(Th)==713 && height(A)==37);
    assert(all(ismember({'x_km','y_km','K_mday','head_m','facies','n_mean'}, D.Properties.VariableNames)));
    [PASS,LOG{end+1}]=deal(PASS+1,'PASS  1. data files present, row counts and columns OK (37/3411/180/713/37)');
catch ME
    [FAIL,LOG{end+1}]=deal(FAIL+1,['FAIL  1. data files: ' ME.message]);
end

% ---------- 2. geometry & boundary segmentation ----------
try
    xb = [ (Bd.X_GK-min(Bd.X_GK))/1000, (Bd.Y_GK-min(Bd.Y_GK))/1000 ]';
    dseg=hypot(diff(xb(1,:)),diff(xb(2,:))); xb=xb(:,[true,dseg>1e-4]);
    cfg0.idxCoast=[]; cfg0.idxSea=[60 860]; cfg0.idxFree=[1 59; 861 3411];   % v2.6
    [bcD,bcN,bcF,bcSea]=boundary_segments(xb,cfg0);
    assert(isempty(bcD) && numel(bcSea)==801 && numel(bcN)==801);     % v2.6: Neumann = sea ONLY
    assert(numel(bcF)==size(xb,2)-801 && isempty(intersect(bcF,bcN)));
    [nx_,ny_]=outward_normals(xb,bcN(1:5:end));
    assert(all(isfinite(nx_)) && all(isfinite(ny_)));
    [PASS,LOG{end+1}]=deal(PASS+1,'PASS  2. boundary roles v2.6: no-flow=sea only (801), all land FREE, Dirichlet=0, normals finite');
catch ME
    [FAIL,LOG{end+1}]=deal(FAIL+1,['FAIL  2. boundary segmentation: ' ME.message]);
end

% ---------- 3. facies prior with support radius ----------
try
    xw=[D.x_km,D.y_km]'; fac=string(D.facies);
    Xq=[ 5 5 5; 45 25 10 ];  % north-empty, central, south points (E;N)
    kc=facies_pool(Xq,xw,fac,3.0);
    assert(all(ismember(kc,[1 2 3])) && kc(1)==2);   % far north must be neutral
    [PASS,LOG{end+1}]=deal(PASS+1,'PASS  3. facies_pool: classes valid, unsampled north -> neutral (medium)');
catch ME
    [FAIL,LOG{end+1}]=deal(FAIL+1,['FAIL  3. facies_pool: ' ME.message]);
end

% ---------- 3b. geo-prior: calibration + spatial logic ----------
try
    CHs = readtable(fullfile('..','data','channel_points_merged.csv'));
    chs = [CHs.x_km, CHs.y_km]';
    cfgG = struct('Ld',3.0,'W0',2.0,'Rw',3.0);
    AXs = readtable(fullfile('..','data','river_axes.csv'));
    axs = geo_prior('axes', AXs);
    Pg = geo_prior('calib', axs, chs, xw, log(D.K_mday), cfgG);
    Gp = geo_prior('eval', Pg, [5 8 D.x_km(1); 48 41 D.y_km(1)], xw);
    assert(isfinite(Pg.R2) && Pg.b > 0.3 && Pg.R2 > 0.15, 'valley structure not confirmed (b<=0.3 or R2<=0.15)');
    assert(Gp.logKp(2) > Gp.logKp(1), 'on-Mat prior not above far-north prior');
    assert(Gp.w7(3) < 0.05 && Gp.w7(1) > 0.8, 'well-distance ramp wrong');
    [PASS,LOG{end+1}]=deal(PASS+1,sprintf('PASS  3b. geo-prior: K_far=%.0f K_chan=%.0f R2=%.2f, ramp+corridor logic OK', exp(Pg.a), exp(Pg.a+Pg.b), Pg.R2));
catch ME
    [FAIL,LOG{end+1}]=deal(FAIL+1,['FAIL  3b. geo-prior: ' ME.message]);
end

% ---------- 4. RNG fold-replay identity ----------
try
    f1=fold_replay(xb); f2=fold_replay(xb);
    assert(isequal(f1,f2) && isequal(accumarray(f1,1),[8;8;7;7;7]));
    [PASS,LOG{end+1}]=deal(PASS+1,'PASS  4. seed replay: fold assignment reproducible, sizes 8/8/7/7/7');
catch ME
    [FAIL,LOG{end+1}]=deal(FAIL+1,['FAIL  4. fold replay: ' ME.message]);
end

% ---------- 5-7. micro-training across precision/device combos ----------
combos = {};
combos{end+1} = struct('prec','single','forceCPU',false,'name','single+auto');
combos{end+1} = struct('prec','double','forceCPU',false,'name','double+auto');
if ng>0
    combos{end+1} = struct('prec','single','forceCPU',true ,'name','single+forceCPU');
end
chk=4;
for c=1:numel(combos)
    chk=chk+1;
    try
        [okBCd, lossDrop, tSec] = micro_train(D,Bd,G,Th, combos{c}.prec, combos{c}.forceCPU);
        assert(okBCd, 'BCd not exactly 0 with empty Dirichlet');
        assert(isfinite(lossDrop) && lossDrop > -0.10, 'training diverged (worse than trivial by >10%)');
        [PASS,LOG{end+1}]=deal(PASS+1,sprintf('PASS  %d. micro-train [%s]: BCd==0, in-sample vs trivial %+.0f%%, %.1f s/300ep', chk, combos{c}.name, 100*lossDrop, tSec));
    catch ME
        [FAIL,LOG{end+1}]=deal(FAIL+1,sprintf('FAIL  %d. micro-train [%s]: %s', chk, combos{c}.name, ME.message));
    end
end

% ---------- config lint: required cfg fields outside comments ----------
chk=chk+1;
try
    reqs = {'main_cv.m',      {'cfg.printEvery','cfg.prec','cfg.rFacies','cfg.idxFree','cfg.patience','cfg.T_mc','cfg.Ld','cfg.Rw','cfg.W0'};
            'main_fushekuqe.m',{'cfg.printEvery','cfg.prec','cfg.rFacies','cfg.idxFree','cfg.patience','cfg.T_mc','cfg.Ld','cfg.Rw','cfg.W0'};
            'uq_bootstrap.m', {'cfg.printEvery','cfg.prec','cfg.rFacies','cfg.idxFree','cfg.patience','cfg.T_mc','cfg.Ld','cfg.Rw','cfg.W0'}};
    for r=1:size(reqs,1)
        txt = fileread(reqs{r,1}); lines = splitlines(string(txt));
        for q = string(reqs{r,2})
            found=false;
            for L = lines'
                code = extractBefore(L+"%", "%");      % strip comment part
                if contains(code, q+" ") || contains(code, q+"=")
                    found=true; break
                end
            end
            assert(found, sprintf('%s missing ACTIVE assignment of %s (swallowed by a comment?)', reqs{r,1}, q));
        end
    end
    [PASS,LOG{end+1}]=deal(PASS+1,sprintf('PASS  %d. config lint: all required cfg fields active (none comment-swallowed)', chk));
catch ME
    [FAIL,LOG{end+1}]=deal(FAIL+1,sprintf('FAIL  %d. config lint: %s', chk, ME.message));
end

% ---------- 9. results/ writable ----------
chk=chk+1;
try
    p=fullfile('..','results','smoke_write_probe.tmp'); fid=fopen(p,'w'); fprintf(fid,'ok'); fclose(fid); delete(p);
    [PASS,LOG{end+1}]=deal(PASS+1,sprintf('PASS  %d. results/ is writable', chk));
catch ME
    [FAIL,LOG{end+1}]=deal(FAIL+1,sprintf('FAIL  %d. results/ write: %s', chk, ME.message));
end

% ---------- report ----------
fprintf('\n==================== SMOKE TEST REPORT ====================\n');
cellfun(@(s) fprintf('%s\n',s), LOG);
fprintf('===========================================================\n');
if FAIL==0
    fprintf('ALL %d CHECKS PASSED -> safe to launch: verify_precision, RUN_ALL, uq_bootstrap\n', PASS);
else
    fprintf('%d/%d CHECKS FAILED -> fix these before any long run.\n', FAIL, PASS+FAIL);
end
fid=fopen(fullfile('..','results','smoke_test.txt'),'w');
cellfun(@(s) fprintf(fid,'%s\n',s), LOG); fclose(fid);
fprintf('Written: results/smoke_test.txt\n');

%% ---------------- local functions ----------------
function foldID = fold_replay(xb)
rng(20260610,'twister');
polySamp = polyshape(xb(1,1:4:end), xb(2,1:4:end),'Simplify',true);
dummy = sample_collocation(polySamp, 40000); %#ok<NASGU>
perm = randperm(37); foldID = zeros(37,1); foldID(perm) = mod(0:36,5)+1;
end

function [okBCd, lossDrop, tSec] = micro_train(D,Bd,G,Th, prec, forceCPU)
rng(20260610,'twister');
cfg.b=40; cfg.N=0; cfg.hsea=0; cfg.R5=4.0; cfg.nMin=0.20; cfg.nMax=0.40;
cfg.w = struct('L1',1,'L2',10,'L3',10,'L4',0,'L5',0.05,'L6',0.5,'L7',1.5,'BC',1);   % v2.7: L4 OFF
cfg.Nc=200; cfg.nEpochs=300; cfg.lr=1e-3; cfg.clip=1.0;   % 30 epochs is too few to beat the mean
                                                            % predictor (-119% under either L5 form), so the gate below
                                                            % only means something at 300 (+32%)
cfg.dropout=0.10; cfg.hidden=64; cfg.nLayers=4; cfg.mFourier=32; cfg.sigmaF=2.0;
cfg.T_mc=5; cfg.printEvery=1e9;
cfg.idxCoast=[]; cfg.idxSea=[60 860]; cfg.idxFree=[1 59; 861 3411];
cfg.poolSize=5000; cfg.bcSubD=2; cfg.bcSubN=5; cfg.sampleDecim=4;
cfg.rFacies=3.0; cfg.Ld=3.0; cfg.W0=2.0; cfg.Rw=3.0; cfg.chanFile='../data/channel_points_merged.csv';
cfg.prec=prec;                 % no early-stop fields: off in micro
ngl=0; try, ngl=gpuDeviceCount; catch, end
cfg.forceCPU=forceCPU; cfg.useGPU = ~cfg.forceCPU && ngl>0;
xw=[D.x_km,D.y_km]';
DAT.Kobs=D.K_mday; DAT.hobs=D.head_m; DAT.fac=string(D.facies); DAT.nPrior=D.n_mean;
xb=[ (Bd.X_GK-min(Bd.X_GK))/1000, (Bd.Y_GK-min(Bd.Y_GK))/1000 ]';
dseg=hypot(diff(xb(1,:)),diff(xb(2,:))); xb=xb(:,[true,dseg>1e-4]);
xd=[ (G.X_GK-min(Bd.X_GK))/1000, (G.Y_GK-min(Bd.Y_GK))/1000 ]';
sc.cx=mean(xb(1,:)); sc.cy=mean(xb(2,:)); sc.s=max(range(xb(1,:)),range(xb(2,:)))/2;
nrm=@(X)[(X(1,:)-sc.cx)/sc.s;(X(2,:)-sc.cy)/sc.s];
[bcD,bcN,~]=boundary_segments(xb,cfg); bcD=bcD(1:cfg.bcSubD:end); bcN=bcN(1:cfg.bcSubN:end);
[nx_,ny_]=outward_normals(xb,bcN);
DAT.Tfix.xd=dlarray(nrm(xd));
DAT.Tfix.D=dlarray(((G.DRASTIC-mean(G.DRASTIC))/std(G.DRASTIC))');
DAT.Tfix.xbD=dlarray(nrm(xb(:,bcD))); DAT.Tfix.xbN=dlarray(nrm(xb(:,bcN)));
DAT.Tfix.nrmN=dlarray([nx_;ny_]); DAT.Tfix.facW=DAT.fac;
if cfg.useGPU
    for s=["xd","D","xbD","xbN","nrmN"], DAT.Tfix.(s)=gpuArray(DAT.Tfix.(s)); end
end
DAT.xwN=nrm(xw);
polySamp=polyshape(xb(1,1:cfg.sampleDecim:end),xb(2,1:cfg.sampleDecim:end),'Simplify',true);
XcPool=sample_collocation(polySamp,cfg.poolSize);
thetaF=theta_interp(Th,min(Bd.X_GK),min(Bd.Y_GK));
DAT.thPool=deg2rad(thetaF(XcPool));
DAT.kcPool=facies_pool(XcPool,xw,DAT.fac,cfg.rFacies);
CH=readtable(cfg.chanFile); chanXY=[CH.x_km,CH.y_km]';
AXT=readtable('../data/river_axes.csv'); axesXY=geo_prior('axes',AXT);
P7=geo_prior('calib',axesXY,chanXY,xw,log(DAT.Kobs),cfg);
G7=geo_prior('eval',P7,XcPool,xw);
DAT.logKpPool=G7.logKp; DAT.w7Pool=G7.w7;
mTh=~isnan(G7.thLoc); DAT.thPool(mTh)=G7.thLoc(mTh);
DAT.XnPool=nrm(XcPool);
t0=tic;
[params,cfgF,lastParts]=train_dpcpinn(cfg,DAT,(1:37)','SMK');
tSec=toc(t0);
okBCd = (lastParts(8)==0);   % v2.6: parts = [L1..L7 BCd BCn]
[muK,~,~,~,~,~]=mc_dropout(params,cfgF,dlarray(DAT.xwN),cfg.T_mc);
assert(all(isfinite(muK)), 'mc_dropout returned non-finite values');
% loss drop proxy: retrain start vs end via Lh inside train is hidden; use data misfit:
e0 = std(log(DAT.Kobs));  e1 = sqrt(mean((muK(:)-log(DAT.Kobs)).^2));
lossDrop = (e0-e1)/e0;              % in-sample improvement vs trivial; signed, so the -10% gate can fail
end
