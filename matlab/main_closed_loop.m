%% DPC-PINN - closed-loop validation of the Wasserstein-2 design
%
%  Reviewer 3, general comment 3: the design section shows how candidates are ranked but never
%  shows that the selected wells improve the estimate. On real data that cannot be shown, because
%  the wells have not been drilled. On the synthetic aquifer of main_synthetic.m it can: the truth
%  is known everywhere, so a design can be executed and then scored.
%
%  Protocol, identical for every design:
%    1. take the posterior fitted to the 37 virtual wells (results/synth/full.mat)
%    2. rank the same 400 candidates with that design's rule
%    3. drill its first five, reading the truth at those points and adding the same observation
%       noise the 37 wells carry
%    4. refit from scratch on 42 wells, same seed, same configuration
%    5. score the refitted fields against the truth over the whole aquifer
%
%  Designs compared: the Wasserstein-2 criterion, the Lambda-weighted variant, A-optimality on the
%  same posterior, a space-filling maximin design, and one random draw. The random row is a single
%  draw rather than an average, because each row costs a full refit; it is there as a floor, not
%  as an estimate of the mean random design.
%
%  Checkpoint/resume: results/closed_loop/<design>.mat
%
%  Outputs: results/closed_loop.csv, figures/fig12_closed_loop.png

clear; clc; close all; addpath(fullfile(pwd,'src'));
OL = oed_lib();
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');

SEED = 20260610;
rng(SEED,'twister');

% nY was 24. At that size the ranking does not reproduce between two runs of the same
% configuration (Spearman 0.45); at 256 it does (0.897). The earlier result, that the
% Wasserstein design does not improve the refitted field, could not be separated from
% that sampling error, so it is recomputed here.
oed.S = 200; oed.nCand = 400; oed.nY = 256; oed.eps = 0.05; oed.nIter = 200; oed.kSelect = 5;

ckDir = fullfile('..','results','closed_loop');
if ~exist(ckDir,'dir'), mkdir(ckDir); end

fFull = fullfile('..','results','synth','full.mat');
fTruth = fullfile('..','results','synthetic_truth.csv');
assert(isfile(fFull) && isfile(fTruth), ...
       'run main_synthetic.m first: this study needs its truth and its 37-well posterior.');

%% ------------------------------------------------------------------ the synthetic case
% The truth is read from synthetic_truth.csv, not rebuilt. The constants below are those of
% main_synthetic.m; only dx, sigK and sigH are used here.
syn.dx = 0.25; syn.theta = 135; syn.aniso = 4.0; syn.corrLong = 6.0;
syn.nMid = 0.30; syn.nAmp = 0.020; syn.etaSd = 0.12; syn.sigK = 0.25; syn.sigH = 0.60;

D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
G  = readtable(fullfile('..','data','drastic_grid_180.csv'));
Th = readtable(fullfile('..','data','orientation_field_grid.csv'));
TR = readtable(fTruth);

xw = [D.x_km, D.y_km]';
xb = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';
dseg = hypot(diff(xb(1,:)), diff(xb(2,:))); xb = xb(:, [true, dseg>1e-4]);
xd = [ (G.X_GK - min(Bd.X_GK))/1000, (G.Y_GK - min(Bd.Y_GK))/1000 ]';

LD = load(fFull);
params0 = LD.params; cfg = LD.cfgF;
sc.cx = mean(xb(1,:)); sc.cy = mean(xb(2,:)); sc.s = max(range(xb(1,:)),range(xb(2,:)))/2;
nrm = @(X) [ (X(1,:)-sc.cx)/sc.s ; (X(2,:)-sc.cy)/sc.s ];

% the synthetic observations, regenerated with the same seed as main_synthetic.m
gx = min(xb(1,:)) : syn.dx : max(xb(1,:));
gy = min(xb(2,:)) : syn.dx : max(xb(2,:));
[GX, GY] = meshgrid(gx, gy);
polyB = polyshape(xb(1,1:cfg.sampleDecim:end), xb(2,1:cfg.sampleDecim:end),'Simplify',true);
inM = reshape(isinterior(polyB, GX(:), GY(:)), size(GX));
FkT = scatteredInterpolant(TR.x_km, TR.y_km, TR.logK_true, 'natural','nearest');
FnT = scatteredInterpolant(TR.x_km, TR.y_km, TR.n_true,    'natural','nearest');
FhT = scatteredInterpolant(TR.x_km, TR.y_km, TR.h_true,    'natural','nearest');

rng(SEED+1,'twister');
KobsS = exp(FkT(xw(1,:)', xw(2,:)') + syn.sigK*randn(37,1));
hobsS = FhT(xw(1,:)', xw(2,:)') + syn.sigH*randn(37,1);

kTrueV = TR.logK_true; nTrueV = TR.n_true; hTrueV = TR.h_true;
XeN = nrm([TR.x_km'; TR.y_km']);

%% ------------------------------------------------------------- rank the candidates
rng(SEED,'twister');
Xcand = sample_collocation(polyB, oed.nCand);
xg = xd;                                   % the 180 DRASTIC nodes carry Lambda
nG = size(xg,2); nC = size(Xcand,2);

Xall = [xg, Xcand, xw];
XallN = dlarray(cast(nrm(Xall), cfg.prec));
if cfg.useGPU, XallN = gpuArray(XallN); end
THK = zeros(oed.S, nG); THN = zeros(oed.S, nG); CK = zeros(oed.S, nC); WK = zeros(oed.S, 37);
fprintf('[CL] %d posterior passes... ', oed.S); t0 = tic;
for s = 1:oed.S
    [lk, nn, ~] = forward_all(params0, XallN, cfg, true);
    lk = double(gather(extractdata(lk))); nn = double(gather(extractdata(nn)));
    THK(s,:) = lk(1:nG); THN(s,:) = nn(1:nG);
    CK(s,:)  = lk(nG+1:nG+nC); WK(s,:) = lk(nG+nC+1:end);
end
fprintf('done (%.1f s).\n', toc(t0));

sigma_y = sqrt(mean((mean(WK,1)' - log(KobsS)).^2));
sK = std(THK(:)); sN = std(THN(:));
ZK = THK/sK; ZN = THN/sN;
Dn = (G.DRASTIC - min(G.DRASTIC)) / (max(G.DRASTIC) - min(G.DRASTIC));
lam = Dn(:)'/mean(Dn);
Cm  = OL.cost_matrix(ZK, ZN, ones(1,nG)) / nG;  cS  = mean(Cm(:));
CL  = OL.cost_matrix(ZK, ZN, lam) / sum(lam);   cLS = mean(CL(:));
fprintf('[CL] sigma_y = %.4f\n', sigma_y);

selW2  = OL.greedy(CK, Cm/cS,  cS,  sigma_y, oed, ZK, ZN, ones(1,nG), 'w2');
selLam = OL.greedy(CK, CL/cLS, cLS, sigma_y, oed, ZK, ZN, lam,        'w2');
selA   = OL.greedy(CK, Cm/cS,  cS,  sigma_y, oed, ZK, ZN, ones(1,nG), 'trace');
selSp  = maximin(Xcand, xw, oed.kSelect);
rng(SEED+7,'twister');
selRnd = randperm(nC, oed.kSelect);

designs = {'W2', selW2; 'W2-Lambda', selLam; 'A-optimal', selA; ...
           'space-filling', selSp; 'random', selRnd};

%% -------------------------------------------------- fixed parts of the training data
DATbase.fac0 = string(D.facies); DATbase.n0 = D.n_mean;
[bcD,bcN,~] = boundary_segments(xb, cfg);
bcD=bcD(1:cfg.bcSubD:end); bcN=bcN(1:cfg.bcSubN:end);
[nx_,ny_] = outward_normals(xb, bcN);
Tfix.xd  = dlarray(nrm(xd));
Tfix.D   = dlarray( ((G.DRASTIC-mean(G.DRASTIC))/std(G.DRASTIC))' );
Tfix.xbD = dlarray(nrm(xb(:,bcD)));
Tfix.xbN = dlarray(nrm(xb(:,bcN)));
Tfix.nrmN= dlarray([nx_;ny_]);
polySamp = polyshape(xb(1,1:cfg.sampleDecim:end), xb(2,1:cfg.sampleDecim:end),'Simplify',true);
rng(SEED,'twister');
XcPool = sample_collocation(polySamp, cfg.poolSize);
thetaF = theta_interp(Th, min(Bd.X_GK), min(Bd.Y_GK));
thPool0 = deg2rad( thetaF(XcPool) );
CH = readtable(cfg.chanFile); chanXY = [CH.x_km, CH.y_km]';
AXT = readtable('../data/river_axes.csv'); axesXY = geo_prior('axes', AXT);

%% ------------------------------------------------------------------ run the designs
rows = {};
% row 0: the 37-well fit itself
[mK0, ~, mN0, ~, mH0] = mc_dropout(params0, cfg, dlarray(XeN), cfg.T_mc);
rows(end+1,:) = {"none (37 wells)", 0, rmse(mK0(:),kTrueV), rmse(mN0(:),nTrueV), ...
                 rmse(mH0(:),hTrueV)};

for d = 1:size(designs,1)
    nameD = designs{d,1}; sel = designs{d,2};
    fck = fullfile(ckDir, sprintf('%s.mat', matlab.lang.makeValidName(nameD)));
    if isfile(fck)
        fprintf('[CL] %-14s checkpoint exists.\n', nameD);
        Sm = load(fck);
    else
        xn = Xcand(:,sel);
        xwE = [xw, xn];
        % the new wells read the truth, with the same noise the existing ones carry
        rng(SEED + 500 + d, 'twister');
        Knew = exp(FkT(xn(1,:)', xn(2,:)') + syn.sigK*randn(oed.kSelect,1));
        hnew = FhT(xn(1,:)', xn(2,:)') + syn.sigH*randn(oed.kSelect,1);
        % facies and porosity prior of a new well: those of the nearest existing well, which is
        % what a driller would have before the log is read
        facE = DATbase.fac0; nprE = DATbase.n0;
        for q = 1:oed.kSelect
            [~,j] = min(hypot(xw(1,:)-xn(1,q), xw(2,:)-xn(2,q)));
            facE(end+1,1) = DATbase.fac0(j); %#ok<AGROW>
            nprE(end+1,1) = DATbase.n0(j);   %#ok<AGROW>
        end
        DAT = struct();
        DAT.Kobs = [KobsS; Knew]; DAT.hobs = [hobsS; hnew];
        DAT.fac = facE; DAT.nPrior = nprE;
        DAT.Tfix = Tfix; DAT.Tfix.facW = facE;
        DAT.xwN = nrm(xwE);
        DAT.thPool = thPool0;
        DAT.kcPool = facies_pool(XcPool, xwE, facE, cfg.rFacies);
        P7 = geo_prior('calib', axesXY, chanXY, xwE, log(DAT.Kobs), cfg);
        G7 = geo_prior('eval', P7, XcPool, xwE);
        DAT.logKpPool = G7.logKp; DAT.w7Pool = G7.w7;
        mTh = ~isnan(G7.thLoc); DAT.thPool(mTh) = G7.thLoc(mTh);
        DAT.XnPool = nrm(XcPool);
        if cfg.useGPU
            for f = ["xd","D","xbD","xbN","nrmN"], DAT.Tfix.(f) = gpuArray(DAT.Tfix.(f)); end
        end
        fprintf('\n########## CLOSED LOOP: %s, 37 + %d wells ##########\n', nameD, oed.kSelect);
        rng(SEED + 900 + d, 'twister');
        [pE, cE] = train_dpcpinn(cfg, DAT, 1:numel(DAT.Kobs), sprintf('CL-%s', nameD));
        [mK, ~, mN, ~, mH] = mc_dropout(pE, cE, dlarray(XeN), cE.T_mc);
        Sm = struct('muK', mK(:), 'muN', mN(:), 'muH', mH(:), ...
                    'x_new', xn(1,:)', 'y_new', xn(2,:)');
        save(fck, '-struct', 'Sm');
    end
    rows(end+1,:) = {string(nameD), oed.kSelect, rmse(Sm.muK,kTrueV), ...
                     rmse(Sm.muN,nTrueV), rmse(Sm.muH,hTrueV)}; %#ok<AGROW>

    % written after EVERY design, not only at the end, so a memory-forced interruption still
    % leaves whatever finished on disk and usable.
    CLp = cell2table(rows, 'VariableNames', {'design','new_wells','logK_RMSE','n_RMSE','head_RMSE'});
    CLp.logK_gain_pct = 100*(CLp.logK_RMSE(1) - CLp.logK_RMSE)/CLp.logK_RMSE(1);
    CLp.n_gain_pct    = 100*(CLp.n_RMSE(1)    - CLp.n_RMSE)   /CLp.n_RMSE(1);
    writetable(CLp, fullfile('..','results','closed_loop.csv'));
end

CL = CLp;
fprintf('\n============ CLOSED-LOOP VALIDATION, field errors against the truth ============\n');
disp(CL);
fprintf(['\nThe question this answers is not which ranking looks sensible on a map, but whether\n' ...
         'drilling where the criterion says improves the field estimate more than drilling\n' ...
         'elsewhere. Positive gain means it does.\n']);

fig12(CL);
fprintf('\nWritten: results/closed_loop.csv, figures/fig12_closed_loop.png\n');


%% ======================================================================= functions

function r = rmse(a, b)
    r = sqrt(mean((a(:) - b(:)).^2));
end

function sel = maximin(Xcand, xw, k)
    sel = zeros(1,k); ref = xw;
    for i = 1:k
        d = min(sqrt((Xcand(1,:)' - ref(1,:)).^2 + (Xcand(2,:)' - ref(2,:)).^2), [], 2);
        [~,b] = max(d); sel(i) = b; ref = [ref, Xcand(:,b)]; %#ok<AGROW>
    end
end

function fig12(CL)
    f = figure('Color','w','Units','centimeters','Position',[2 2 17.4 7.0]);
    nm = string(CL.design);
    for panel = 1:2
        subplot(1,2,panel); hold on; box on
        if panel == 1, y = CL.logK_RMSE; lab = 'log K field RMSE';
        else,          y = CL.n_RMSE;    lab = 'porosity field RMSE'; end
        b = bar(y, 0.6); b.FaceColor = [0.62 0.70 0.80]; b.EdgeColor = [0.25 0.30 0.38];
        yline(y(1), '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.8);
        set(gca, 'XTick', 1:numel(y), 'XTickLabel', nm, 'XTickLabelRotation', 30, 'FontSize', 8);
        ylabel(lab);
        if panel == 1, title('(a) conductivity', 'FontWeight','normal');
        else,          title('(b) porosity', 'FontWeight','normal'); end
    end
    set(f,'PaperUnits','centimeters','PaperPosition',[0 0 17.4 7.0],'PaperSize',[17.4 7.0]);
    print(f, fullfile('..','figures','fig12_closed_loop.png'), '-dpng', '-r600');
    close(f);
end
