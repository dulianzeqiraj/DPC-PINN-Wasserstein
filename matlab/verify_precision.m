%% Precision equivalence test: FP32 vs FP64 on CV fold 1 (same seed, same wells)
%  Acceptance: |RMSE(single) - RMSE(double)| within seed-to-seed noise (~0.005 logK).
%  Writes results/precision_check.txt; archive it in the repository.
clear; clc; addpath(fullfile(pwd,'src'));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');

out = fullfile('..','results','precision_check.txt'); fid = fopen(out,'w');
res = struct();
for PREC = ["double","single"]
    rng(20260610,'twister');
    cfg = struct();
    cfg.b=40; cfg.N=0; cfg.hsea=0; cfg.R5=4.0; cfg.nMin=0.20; cfg.nMax=0.40;
    cfg.w = struct('L1',1,'L2',10,'L3',10,'L4',0,'L5',0.05,'L6',0.5,'L7',1.5,'BC',1);   % v2.7: L4 OFF
    cfg.Nc=800; cfg.nEpochs=1800; cfg.lr=1e-3; cfg.clip=1.0;
    cfg.dropout=0.10; cfg.hidden=64; cfg.nLayers=4; cfg.mFourier=32; cfg.sigmaF=2.0;
    cfg.T_mc=30; cfg.printEvery=600;
    cfg.idxCoast=[]; cfg.idxSea=[60 860]; cfg.idxFree=[1 59; 861 3411];   % v2.6 land-free BC
    cfg.poolSize=40000; cfg.bcSubD=2; cfg.bcSubN=5; cfg.sampleDecim=4;
    cfg.Ld=3.0; cfg.W0=2.0; cfg.Rw=3.0; cfg.chanFile='../data/channel_points_merged.csv';
    cfg.rFacies=3.0; cfg.prec=char(PREC);
    cfg.patience=300; cfg.esTol=0.01; cfg.minEp=1000;
    cfg.forceCPU=false; cfg.useGPU = ~cfg.forceCPU && gpuDeviceCount>0;

    D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
    Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
    G  = readtable(fullfile('..','data','drastic_grid_180.csv'));
    Th = readtable(fullfile('..','data','orientation_field_grid.csv'));
    xw = [D.x_km, D.y_km]';
    DAT.Kobs=D.K_mday; DAT.hobs=D.head_m; DAT.fac=string(D.facies); DAT.nPrior=D.n_mean;
    xb = [ (Bd.X_GK-min(Bd.X_GK))/1000, (Bd.Y_GK-min(Bd.Y_GK))/1000 ]';
    dseg=hypot(diff(xb(1,:)),diff(xb(2,:))); xb=xb(:,[true,dseg>1e-4]);
    xd = [ (G.X_GK-min(Bd.X_GK))/1000, (G.Y_GK-min(Bd.Y_GK))/1000 ]';
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
    perm=randperm(37); foldID=zeros(37,1); foldID(perm)=mod(0:36,5)+1;
    wTest=find(foldID==1); wTrain=find(foldID~=1);

    tP=tic;
    [params,cfgF]=train_dpcpinn(cfg,DAT,wTrain,char(PREC));
    [muK,~,~,~,muH,~]=mc_dropout(params,cfgF,dlarray(DAT.xwN(:,wTest)),cfg.T_mc);
    wall=toc(tP)/60;
    eK=muK(:)-log(DAT.Kobs(wTest)); eH=muH(:)-DAT.hobs(wTest);
    res.(char(PREC))=struct('rmseK',sqrt(mean(eK.^2)),'rmseH',sqrt(mean(eH.^2)),'min',wall);
    L=sprintf('%-6s : fold-1 logK RMSE=%.6f  head RMSE=%.5f m  wall=%.1f min', char(PREC), ...
              res.(char(PREC)).rmseK, res.(char(PREC)).rmseH, wall);
    disp(L); fprintf(fid,'%s\n',L);
end
dK=abs(res.single.rmseK-res.double.rmseK); dH=abs(res.single.rmseH-res.double.rmseH);
sp=res.double.min/res.single.min;
L1=sprintf('DELTA  : |dRMSE logK|=%.4f (noise threshold ~0.005)  |dRMSE head|=%.3f m', dK, dH);
L2=sprintf('SPEED  : single is %.1fx faster on this machine', sp);
L3=sprintf('VERDICT: %s', string(ternary(dK<=0.01,'EQUIVALENT - single adopted as canonical','REVIEW NEEDED - keep double precision')));
disp(L1);disp(L2);disp(L3); fprintf(fid,'%s\n%s\n%s\n',L1,L2,L3); fclose(fid);
fprintf('Written: results/precision_check.txt\n');
function o=ternary(c,a,b), if c, o=a; else, o=b; end, end
