%% DPC-PINN - synthetic recovery of known conductivity and porosity fields
%
%  Verification ladder item (i). The README of v1.0.0 listed this script and the repository did
%  not contain it: the recovery it promised had never been run. This is that experiment.
%
%  Truth construction. A channelized porosity field with the NW-SE fabric of the real aquifer is
%  drawn on the real domain, and the conductivity field is generated FROM it through the same
%  Kozeny-Carman relation the model uses as a soft prior, plus an independent smooth residual.
%  The residual matters: if logK were an exact function of n the coupling term L6 could not be
%  wrong, and the test would be rigged. Heads are then obtained by a finite-difference solve of
%  the steady Darcy equation on that conductivity field, so the physics residual L1 has a true
%  solution to find.
%
%  One honest difference from the field case. A forward solve needs a well-posed boundary-value
%  problem, so the synthetic truth carries Dirichlet heads on the land perimeter, interpolated
%  from the 37 observed heads and clamped to their range, and no-flow at the Adriatic segment. The field model
%  leaves the land perimeter free, as section 3.1 explains. The synthetic study therefore tests
%  the inversion machinery, not the boundary treatment.
%
%  Wells. The 37 virtual wells sit at the real well coordinates, so the sampling density and its
%  clustering are the real ones. Observations carry the same noise level the trained model leaves
%  unexplained at the real wells.
%
%  Fifteen configurations are run on the identical synthetic data and the identical seed: the
%  loss ablation, a weight sensitivity with each prior weight halved and doubled (Reviewer 4,
%  point 3), and an architecture ablation over depth and width (Reviewer 4, point 7). One fit per
%  configuration suffices here because the truth is known at every cell, which is what makes the
%  marginal contribution of each prior measurable in a way it is not on real data.
%
%  Checkpoint/resume: results/synth/<tag>.mat. Existing runs are skipped. The full model also
%  stores its parameters, because main_closed_loop.m starts from that posterior.
%
%  Outputs: results/synthetic_truth.csv     the truth on the evaluation grid
%           results/synthetic_recovery.csv  one row per configuration
%           results/synthetic_kriging.csv   the same fields from kriged conductivity
%           figures/fig11_synthetic.png     truth against recovery for the full model

clear; clc; close all; addpath(fullfile(pwd,'src'));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');

SEED = 20260610;
rng(SEED,'twister');

syn.dx       = 0.25;    % km, finite-difference cell size
syn.theta    = 135;     % degrees, paleo-channel fabric, as in section 2.3
syn.aniso    = 4.0;     % correlation-length ratio along and across the fabric
syn.corrLong = 6.0;     % km, correlation length along the fabric
syn.nMid     = 0.30;    % centre of the synthetic porosity field
% nAmp and etaSd are set so the synthetic conductivity spans roughly the range the real wells
% span, log(50) to log(200), a width of 1.39. The first attempt used 0.055 and 0.20 and produced
% 20 to 359 m/day, more than twice the observed spread, which makes the recovery task easier in
% the tails than the real one. dKC/dn is about 12.9 near n = 0.30, so a porosity half-range of
% 0.020 contributes about 1.0 in log-conductivity and the residual supplies the rest.
syn.nAmp     = 0.020;   % half-range of the channelized component
syn.etaSd    = 0.12;    % sd of the independent log-conductivity residual
syn.sigK     = 0.25;    % observation noise on log-conductivity at the virtual wells
syn.sigH     = 0.60;    % m, observation noise on heads

ckDir = fullfile('..','results','synth');
if ~exist(ckDir,'dir'), mkdir(ckDir); end

%% ---------------------------------------------------------------- real geometry
D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
G  = readtable(fullfile('..','data','drastic_grid_180.csv'));
Th = readtable(fullfile('..','data','orientation_field_grid.csv'));

xw = [D.x_km, D.y_km]';
xb = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';
dseg = hypot(diff(xb(1,:)), diff(xb(2,:)));
keepV = [true, dseg>1e-4];
xb = xb(:, keepV);
vidx = find(keepV);                          % original vertex index of each kept vertex
xd = [ (G.X_GK - min(Bd.X_GK))/1000, (G.Y_GK - min(Bd.Y_GK))/1000 ]';

cfg0.sampleDecim = 4;
polyB = polyshape(xb(1,1:cfg0.sampleDecim:end), xb(2,1:cfg0.sampleDecim:end),'Simplify',true);

%% ------------------------------------------------------------ evaluation grid
gx = min(xb(1,:)) : syn.dx : max(xb(1,:));
gy = min(xb(2,:)) : syn.dx : max(xb(2,:));
[GX, GY] = meshgrid(gx, gy);
inM = isinterior(polyB, GX(:), GY(:));
inM = reshape(inM, size(GX));
nCell = nnz(inM);
fprintf('[SYN] grid %d x %d at %.2f km, %d cells inside the aquifer.\n', ...
        numel(gy), numel(gx), syn.dx, nCell);

%% ------------------------------------------------- truth: porosity, then conductivity
% Anisotropic smoothing of white noise: a Gaussian kernel elongated along the fabric.
F1 = aniso_field(size(GX), syn.dx, syn.corrLong, syn.corrLong/syn.aniso, syn.theta);
F2 = aniso_field(size(GX), syn.dx, syn.corrLong, syn.corrLong/syn.aniso, syn.theta);
nTrue = syn.nMid + syn.nAmp * F1;
nTrue = min(max(nTrue, 0.205), 0.395);

KCterm = log( nTrue.^3 ./ (1 - nTrue).^2 );
eta    = syn.etaSd * F2;
beta   = median(log(D.K_mday)) - median(KCterm(inM));
logKTrue = beta + KCterm + eta;
fprintf('[SYN] truth: logK in [%.2f, %.2f], K in [%.0f, %.0f] m/day, n in [%.3f, %.3f].\n', ...
        min(logKTrue(inM)), max(logKTrue(inM)), exp(min(logKTrue(inM))), ...
        exp(max(logKTrue(inM))), min(nTrue(inM)), max(nTrue(inM)));
fprintf('[SYN] correlation of logK with the Kozeny-Carman term: %.3f (residual sd %.2f).\n', ...
        corr(logKTrue(inM), KCterm(inM)), std(eta(inM)));

%% ------------------------------------------------------------- truth: heads
% The land perimeter carries Dirichlet values interpolated from the 37 observed heads and clamped
% to their range. A plane fitted to the same heads was tried first and rejected: it is a good fit
% at the wells (R2 0.82) but it extrapolates to negative head beyond about 17 km, so the synthetic
% truth had heads down to -9 m and the virtual wells sampled 0 to 22 m against the observed 4 to
% 45 m. Interpolating instead cannot leave the observed range, by construction.
hLo = min(D.head_m); hHi = max(D.head_m);
Fbnd = scatteredInterpolant(xw(1,:)', xw(2,:)', D.head_m, 'natural', 'nearest');
hbFun = @(x, y) min(max(Fbnd(x, y), hLo), hHi);
fprintf('[SYN] land-boundary heads interpolated from the wells, clamped to [%.1f, %.1f] m.\n', ...
        hLo, hHi);

% Boundary cells and their type. Vertices 60..860 of the ORIGINAL polygon are the Adriatic
% segment (cfg.idxSea in every driver); everything else is land.
isSeaVert = false(1, size(xb,2));
isSeaVert( vidx >= 60 & vidx <= 860 ) = true;

hTrue = darcy_solve(exp(logKTrue), inM, gx, gy, xb, isSeaVert, hbFun);
fprintf('[SYN] head field: %.1f to %.1f m (observed range %.1f to %.1f m).\n', ...
        min(hTrue(inM)), max(hTrue(inM)), min(D.head_m), max(D.head_m));

%% ----------------------------------------------------- virtual wells at the real sites
% logK and n are analytic and defined on the whole grid, so a gridded interpolant is safe for
% them. hTrue is NaN outside the aquifer mask, and linear interpolation returns NaN for any well
% whose neighbouring cells include one, which silently poisoned the loss. Its interpolant is
% therefore built from the in-mask cells only.
Fk = griddedInterpolant({gy, gx}, logKTrue, 'linear', 'nearest');
Fn = griddedInterpolant({gy, gx}, nTrue,    'linear', 'nearest');
Fh = scatteredInterpolant(GX(inM), GY(inM), hTrue(inM), 'natural', 'nearest');
logKw = Fk(xw(2,:)', xw(1,:)');
nw    = Fn(xw(2,:)', xw(1,:)');
hw    = Fh(xw(1,:)', xw(2,:)');
assert(all(isfinite(logKw)) && all(isfinite(nw)) && all(isfinite(hw)), ...
       'a virtual well fell outside the interpolable region: %d logK, %d n, %d head not finite', ...
       sum(~isfinite(logKw)), sum(~isfinite(nw)), sum(~isfinite(hw)));

rng(SEED+1,'twister');
KobsS = exp(logKw + syn.sigK*randn(37,1));
hobsS = hw + syn.sigH*randn(37,1);
assert(all(isfinite(KobsS)) && all(isfinite(hobsS)), 'a virtual observation is not finite');
fprintf('[SYN] virtual wells: K %.0f to %.0f m/day, head %.1f to %.1f m, all finite.\n', ...
        min(KobsS), max(KobsS), min(hobsS), max(hobsS));

TT = table(GX(inM), GY(inM), logKTrue(inM), nTrue(inM), hTrue(inM), ...
     'VariableNames', {'x_km','y_km','logK_true','n_true','h_true'});
writetable(TT, fullfile('..','results','synthetic_truth.csv'));

%% -------------------------------------------------------------- model configuration
cfg.b=40; cfg.N=0; cfg.hsea=0; cfg.R5=4.0; cfg.nMin=0.20; cfg.nMax=0.40;
cfg.w = struct('L1',1,'L2',10,'L3',10,'L4',0,'L5',0.05,'L6',0.5,'L7',1.5,'BC',1);
cfg.Nc=800; cfg.nEpochs=1800; cfg.lr=1e-3; cfg.clip=1.0;
cfg.dropout=0.10; cfg.hidden=64; cfg.nLayers=4; cfg.mFourier=32; cfg.sigmaF=2.0;
cfg.T_mc = 30; cfg.printEvery = 600;
cfg.idxCoast = []; cfg.idxSea = [60 860]; cfg.idxFree = [1 59; 861 3411];
cfg.poolSize=40000; cfg.bcSubD=2; cfg.bcSubN=5; cfg.sampleDecim=4;
cfg.Ld = 3.0; cfg.W0 = 2.0; cfg.Rw = 3.0;
cfg.chanFile = '../data/channel_points_merged.csv';
cfg.rFacies = 3.0; cfg.prec = 'single';
cfg.patience = 300; cfg.esTol = 0.01; cfg.minEp = 1000;
cfg.forceCPU = false;
cfg.useGPU = ~cfg.forceCPU && gpuDeviceCount > 0;

sc.cx = mean(xb(1,:)); sc.cy = mean(xb(2,:)); sc.s = max(range(xb(1,:)),range(xb(2,:)))/2;
nrm = @(X) [ (X(1,:)-sc.cx)/sc.s ; (X(2,:)-sc.cy)/sc.s ];

DAT.Kobs = KobsS; DAT.hobs = hobsS; DAT.fac = string(D.facies); DAT.nPrior = D.n_mean;
[bcD,bcN,~] = boundary_segments(xb, cfg);
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
XcPool = sample_collocation(polySamp, cfg.poolSize);
thetaF = theta_interp(Th, min(Bd.X_GK), min(Bd.Y_GK));
DAT.thPool = deg2rad( thetaF(XcPool) );
DAT.kcPool = facies_pool(XcPool, xw, DAT.fac, cfg.rFacies);
CH = readtable(cfg.chanFile); chanXY = [CH.x_km, CH.y_km]';
AXT = readtable('../data/river_axes.csv'); axesXY = geo_prior('axes', AXT);
P7 = geo_prior('calib', axesXY, chanXY, xw, log(DAT.Kobs), cfg);
fprintf('[SYN] geo-prior recalibrated on the synthetic wells: K_far=%.0f K_chan=%.0f R2=%.2f\n', ...
        exp(P7.a), exp(P7.a+P7.b), P7.R2);
G7 = geo_prior('eval', P7, XcPool, xw);
DAT.logKpPool = G7.logKp; DAT.w7Pool = G7.w7;
mTh = ~isnan(G7.thLoc); DAT.thPool(mTh) = G7.thLoc(mTh);
DAT.XnPool = nrm(XcPool);

%% ------------------------------------------------------------------ configurations
%  Three blocks, all on the same synthetic data and the same wells:
%    loss        the 2^3-style subsets of {L5, L6, L7} that matter, plus the full model
%    weights     each prior weight halved and doubled (Reviewer 4, point 3; Reviewer 3, detail 3)
%    architecture depth, width and activation (Reviewer 4, point 7)
%  On real data a weight or architecture sweep would need five folds per configuration to say
%  anything. Here the truth is known at every cell, so one fit per configuration is enough, and
%  the error being compared is a field error rather than a score at 37 sampled points.
W5 = 0.05; W6 = 0.5; W7 = 1.5;
cfgBase = cfg;

CFGS = {};   % {name, tag, w5, w6, w7, nLayers, hidden, act, block}
CFGS(end+1,:) = {'base',        'base',  0,     0,    0,    4, 64, 'tanh', 'loss'};
CFGS(end+1,:) = {'+L5',         'L5',    W5,    0,    0,    4, 64, 'tanh', 'loss'};
CFGS(end+1,:) = {'+L6',         'L6',    0,     W6,   0,    4, 64, 'tanh', 'loss'};
CFGS(end+1,:) = {'+L7',         'L7',    0,     0,    W7,   4, 64, 'tanh', 'loss'};
CFGS(end+1,:) = {'full',        'full',  W5,    W6,   W7,   4, 64, 'tanh', 'loss'};
CFGS(end+1,:) = {'w5 x 0.5',    'w5lo',  W5/2,  W6,   W7,   4, 64, 'tanh', 'weights'};
CFGS(end+1,:) = {'w5 x 2',      'w5hi',  W5*2,  W6,   W7,   4, 64, 'tanh', 'weights'};
CFGS(end+1,:) = {'w6 x 0.5',    'w6lo',  W5,    W6/2, W7,   4, 64, 'tanh', 'weights'};
CFGS(end+1,:) = {'w6 x 2',      'w6hi',  W5,    W6*2, W7,   4, 64, 'tanh', 'weights'};
CFGS(end+1,:) = {'w7 x 0.5',    'w7lo',  W5,    W6,   W7/2, 4, 64, 'tanh', 'weights'};
CFGS(end+1,:) = {'w7 x 2',      'w7hi',  W5,    W6,   W7*2, 4, 64, 'tanh', 'weights'};
CFGS(end+1,:) = {'2 layers',    'L2w64', W5,    W6,   W7,   2, 64, 'tanh', 'architecture'};
CFGS(end+1,:) = {'6 layers',    'L6w64', W5,    W6,   W7,   6, 64, 'tanh', 'architecture'};
CFGS(end+1,:) = {'32 units',    'L4w32', W5,    W6,   W7,   4, 32, 'tanh', 'architecture'};
CFGS(end+1,:) = {'128 units',   'L4w128',W5,    W6,   W7,   4,128, 'tanh', 'architecture'};

XeN = nrm([GX(inM)'; GY(inM)']);
kTrueV = logKTrue(inM); nTrueV = nTrue(inM); hTrueV = hTrue(inM);

rows = {};
for m = 1:size(CFGS,1)
    nameM = CFGS{m,1}; tagM = CFGS{m,2};
    fck = fullfile(ckDir, sprintf('%s.mat', tagM));
    if isfile(fck)
        fprintf('[SYN] %-11s checkpoint exists, loading.\n', nameM);
        Sm = load(fck);
    else
        cfg = cfgBase;
        cfg.w.L5 = CFGS{m,3}; cfg.w.L6 = CFGS{m,4}; cfg.w.L7 = CFGS{m,5};
        cfg.nLayers = CFGS{m,6}; cfg.hidden = CFGS{m,7};
        fprintf(['\n########## SYNTHETIC %s  (L5=%.3g L6=%.3g L7=%.3g, %d layers, ' ...
                 '%d units) ##########\n'], nameM, cfg.w.L5, cfg.w.L6, cfg.w.L7, ...
                cfg.nLayers, cfg.hidden);
        rng(SEED + 100*m, 'twister');
        tC = tic;
        [params, cfgF] = train_dpcpinn(cfg, DAT, 1:37, sprintf('SYN-%s', tagM));
        [muK, ~, muN, ~, muH] = mc_dropout(params, cfgF, dlarray(XeN), cfgF.T_mc);
        Sm = struct('muK', muK(:), 'muN', muN(:), 'muH', muH(:), 'minutes', toc(tC)/60);
        if strcmp(tagM, 'full')
            % the closed-loop study needs this posterior, not just its summary
            Sm.params = params; Sm.cfgF = cfgF;
        end
        save(fck, '-struct', 'Sm');
        fprintf('  saved %s (%.1f min)\n', fck, Sm.minutes);
    end
    eK = Sm.muK - kTrueV; eN = Sm.muN - nTrueV; eH = Sm.muH - hTrueV;
    rows(end+1,:) = { string(CFGS{m,9}), string(nameM), CFGS{m,3}, CFGS{m,4}, CFGS{m,5}, CFGS{m,6}, CFGS{m,7}, ...
        sqrt(mean(eK.^2)), mean(eK), corr(Sm.muK, kTrueV), ...
        sqrt(mean(eN.^2)), mean(eN), corr(Sm.muN, nTrueV), ...
        sqrt(mean(eH.^2)), corr(Sm.muH, hTrueV) }; %#ok<AGROW>
    if strcmp(tagM, 'full'), full_muK = Sm.muK; full_muN = Sm.muN; end

    % written after EVERY configuration, not only at the end: if the run is interrupted (memory
    % pressure, a shared machine), whatever has finished is already on disk and usable.
    RT = cell2table(rows, 'VariableNames', {'block','config','w5','w6','w7','layers','units', ...
         'logK_RMSE','logK_bias','logK_corr','n_RMSE','n_bias','n_corr','head_RMSE','head_corr'});
    writetable(RT, fullfile('..','results','synthetic_recovery.csv'));
end

%% ------------------------ what any K estimator gets for free (Reviewer 4, point 5)
% Ordinary kriging of the 37 synthetic conductivities, then the same Kozeny-Carman inverse,
% scored against the porosity field that is actually known here.
[ak, c0k, c1k] = fit_expvario_syn(xw', log(KobsS));
kOK = ok_predict_syn(xw', log(KobsS), [GX(inM), GY(inM)], ak, c0k, c1k);
bAll = mean(log(KobsS) - log(D.n_mean.^3 ./ (1-D.n_mean).^2));
nOK = arrayfun(@(t) invert_kc_syn(t - bAll, 0.20, 0.40), kOK);
fprintf('\n===== porosity from kriged K against the known field =====\n');
fprintf('kriging     logK RMSE %.4f   n RMSE %.4f\n', ...
        sqrt(mean((kOK-kTrueV).^2)), sqrt(mean((nOK-nTrueV).^2)));
if exist('full_muK','var')
    fprintf('DPC-PINN    logK RMSE %.4f   n RMSE %.4f\n', ...
            sqrt(mean((full_muK-kTrueV).^2)), sqrt(mean((full_muN-nTrueV).^2)));
end
KR = table(sqrt(mean((kOK-kTrueV).^2)), sqrt(mean((nOK-nTrueV).^2)), ...
     'VariableNames', {'kriging_logK_RMSE','kriging_n_RMSE'});
writetable(KR, fullfile('..','results','synthetic_kriging.csv'));

fprintf('\n=========== SYNTHETIC RECOVERY, %d evaluation cells ===========\n', nCell);
disp(RT);
fprintf(['\nRead this as: the truth is known everywhere, so these errors are field errors, not\n' ...
         'errors at the 37 sampled points. A prior that only interpolates better at the wells\n' ...
         'will not show up here.\n']);

%% ------------------------------------------------------------------- Figure 11
if exist('full_muK','var')
    fig11(gx, gy, inM, logKTrue, nTrue, full_muK, full_muN, xb, xw);
    fprintf('\nWritten: results/synthetic_truth.csv, results/synthetic_recovery.csv,\n');
    fprintf('         figures/fig11_synthetic.png\n');
end


%% ======================================================================= functions

function F = aniso_field(sz, dx, Llong, Lcross, thetaDeg)
% Unit-variance smooth field with an elliptical correlation structure rotated to thetaDeg,
% built by convolving white noise with the matching Gaussian kernel.
    r = ceil(3*Llong/dx);
    [KX, KY] = meshgrid(-r:r, -r:r);
    t = deg2rad(thetaDeg);
    u =  KX*cos(t) + KY*sin(t);
    v = -KX*sin(t) + KY*cos(t);
    Kk = exp( -0.5*((u*dx/Llong).^2 + (v*dx/Lcross).^2) );
    Kk = Kk / sqrt(sum(Kk(:).^2));
    F = conv2(randn(sz), Kk, 'same');
    F = (F - mean(F(:))) / std(F(:));
end

function h = darcy_solve(K, inM, gx, gy, xb, isSeaVert, hbFun)
% Five-point finite volume for div(K grad h) = 0 with harmonic interface conductivities.
% Faces leaving the domain at the Adriatic segment carry zero flux; faces leaving it on land
% carry a Dirichlet value evaluated AT THE FACE. The first version evaluated it at the cell
% centre, half a cell away, and left 0.34 m of residual on a case whose answer is exactly linear.
    [ny, nx] = size(K);
    dx = gx(2)-gx(1); dy = gy(2)-gy(1);
    h = nan(ny, nx);
    h(inM) = fv_solve(K, inM, gx, gy, dx, dy, xb, isSeaVert, hbFun);

    % Validation on a case with a known answer: uniform conductivity, every outside face
    % Dirichlet, and a linear boundary field, for which the solution is that same linear field.
    linf = @(x,y) 12.0 + 0.7*x - 0.4*y;
    hv = fv_solve(ones(ny,nx), inM, gx, gy, dx, dy, xb, false(size(isSeaVert)), linf);
    [XX, YY] = meshgrid(gx, gy);
    err = max(abs(hv - linf(XX(inM), YY(inM))));
    fprintf('[SYN] solver check, uniform K and a linear boundary field: max error %.2e m\n', err);
    assert(err < 1e-8, ...
        'the Darcy solver does not reproduce a linear field (%.2e m); the truth is not usable', err);
end

function hv = fv_solve(K, inM, gx, gy, dx, dy, xb, isSeaVert, hbFun)
    [ny, nx] = size(K);
    id = zeros(ny,nx); id(inM) = 1:nnz(inM);
    N = nnz(inM);
    I = zeros(5*N,1); J = I; V = I; b = zeros(N,1); p = 0;
    har = @(a,c) 2*a.*c ./ (a + c);
    hasSea = any(isSeaVert);
    for jy = 1:ny
        for jx = 1:nx
            if ~inM(jy,jx), continue; end
            k0 = id(jy,jx); diag0 = 0;
            for sdir = 1:4
                switch sdir
                    case 1, iy=jy;   ix=jx-1; step=dx; area=dy; fx=gx(jx)-dx/2; fy=gy(jy);
                    case 2, iy=jy;   ix=jx+1; step=dx; area=dy; fx=gx(jx)+dx/2; fy=gy(jy);
                    case 3, iy=jy-1; ix=jx;   step=dy; area=dx; fx=gx(jx);      fy=gy(jy)-dy/2;
                    case 4, iy=jy+1; ix=jx;   step=dy; area=dx; fx=gx(jx);      fy=gy(jy)+dy/2;
                end
                inside = iy>=1 && iy<=ny && ix>=1 && ix<=nx && inM(iy,ix);
                if inside
                    T = har(K(jy,jx), K(iy,ix)) * area / step;
                    p=p+1; I(p)=k0; J(p)=id(iy,ix); V(p)=-T;
                    diag0 = diag0 + T;
                else
                    if hasSea
                        [~, jv] = min( (xb(1,:)-fx).^2 + (xb(2,:)-fy).^2 );
                        if isSeaVert(jv), continue; end        % no flux at the shoreline
                    end
                    T = K(jy,jx) * area / (step/2);
                    diag0 = diag0 + T;
                    b(k0) = b(k0) + T * hbFun(fx, fy);
                end
            end
            p=p+1; I(p)=k0; J(p)=k0; V(p)=diag0;
        end
    end
    A = sparse(I(1:p), J(1:p), V(1:p), N, N);
    hv = A \ b;
end

function fig11(gx, gy, inM, logKTrue, nTrue, muK, muN, xb, xw)
    Kh = nan(size(inM)); Kh(inM) = muK;
    Nh = nan(size(inM)); Nh(inM) = muN;
    Kt = logKTrue; Kt(~inM) = NaN;
    Nt = nTrue;    Nt(~inM) = NaN;
    f = figure('Color','w','Units','centimeters','Position',[2 2 17.4 12.4]);
    P = {Kt, Kh, Nt, Nh};
    Tl = {'(a) log K, truth','(b) log K, recovered','(c) n, truth','(d) n, recovered'};
    for i = 1:4
        subplot(2,2,i);
        imagesc(gx, gy, P{i}, 'AlphaData', ~isnan(P{i})); set(gca,'YDir','normal'); hold on
        plot(xb(1,:), xb(2,:), 'k-', 'LineWidth', 0.5);
        plot(xw(1,:), xw(2,:), 'k^', 'MarkerSize', 3, 'MarkerFaceColor', [.6 .6 .6]);
        axis equal tight; colorbar; title(Tl{i}, 'FontWeight','normal');
        xlabel('x (km)'); ylabel('y (km)'); set(gca,'FontSize',8);
        if i <= 2, caxis([min(Kt(inM)) max(Kt(inM))]); else, caxis([min(Nt(inM)) max(Nt(inM))]); end
    end
    set(f,'PaperUnits','centimeters','PaperPosition',[0 0 17.4 12.4],'PaperSize',[17.4 12.4]);
    print(f, fullfile('..','figures','fig11_synthetic.png'), '-dpng', '-r600');
    close(f);
end

function [a,c0,c1] = fit_expvario_syn(X, v)
% Same estimator as main_baselines.m, kept local so this script has no cross-file dependency.
n = numel(v); [I,J] = find(triu(true(n),1));
h = hypot(X(I,1)-X(J,1), X(I,2)-X(J,2));
g = 0.5*(v(I)-v(J)).^2;
hs = sort(h);
edges = hs( max(1, round(linspace(1, 0.75*numel(hs), 9))) );
hb=[]; gb=[];
for k=1:numel(edges)-1
    sM = h>=edges(k) & h<edges(k+1);
    if nnz(sM)>=5, hb(end+1)=mean(h(sM)); gb(end+1)=mean(g(sM)); end %#ok<AGROW>
end
best = inf; a = max(hb)/4; c0 = 0; c1 = var(v);
for a_try = linspace(max(hb)/20, max(hb)*2, 60)
    Phi = [ones(numel(hb),1), 1-exp(-hb(:)/a_try)];
    cc = Phi \ gb(:);  cc = max(cc, 0);
    r = norm(Phi*cc - gb(:));
    if r < best, best = r; a = a_try; c0 = cc(1); c1 = max(cc(2), 1e-6); end
end
end

function mu = ok_predict_syn(Xtr, v, Xte, a, c0, c1)
n = size(Xtr,1);
Hk = hypot(Xtr(:,1)-Xtr(:,1)', Xtr(:,2)-Xtr(:,2)');
Cm = c1*exp(-Hk/a); Cm(1:n+1:end) = c0 + c1;
A  = [Cm, ones(n,1); ones(1,n), 0];
mu = nan(size(Xte,1),1);
for q = 1:size(Xte,1)
    h0 = hypot(Xtr(:,1)-Xte(q,1), Xtr(:,2)-Xte(q,2));
    cvec = c1*exp(-h0/a);
    w = A \ [cvec; 1];
    mu(q) = w(1:n)' * v(:);
end
end

function n = invert_kc_syn(target, lo, hi)
f = @(nn) log(nn.^3 ./ (1-nn).^2) - target;
if f(lo) > 0, n = lo; return; end
if f(hi) < 0, n = hi; return; end
for it = 1:60
    mid = 0.5*(lo+hi);
    if f(mid) > 0, hi = mid; else, lo = mid; end
end
n = 0.5*(lo+hi);
end
