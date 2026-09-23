%% DPC-PINN - Wasserstein-2 Bayesian optimal experimental design, Eqs (11)-(13)
%
%  WHAT CHANGED IN v2.7.0, AND WHY
%
%  The v1.0.0 version of this script did not compute the criterion the paper defines. For each
%  candidate it drew a one-dimensional cloud of dropout samples of logK AT THAT SINGLE POINT and
%  measured its optimal-transport distance to a synthetic Gaussian centred on the geometric
%  prior. That is a "how unlike the prior is the model here" score. It contains no simulated
%  datum, no Bayesian update, no importance weights and no joint field theta = (logK, n), so
%  Eq. (12) was never evaluated and Propositions 2 and 3 were proved for a criterion the code did
%  not compute. The DRASTIC variant multiplied the score by a scalar (0.5 + 0.5*w_D) instead of
%  changing the ground metric, so it was not the Lambda-weighted utility of Proposition 2 either.
%  Reviewer 1 (points 11-13) and Reviewer 3 (point 3) both suspected this from the outside.
%
%  This version implements (11)-(13) as written:
%    theta_s   the s-th posterior sample is the JOINT FIELD (logK, n) on a fixed support of G
%              nodes, drawn by MC dropout from the trained model of section 3.4
%    y         a simulated pumping-test log-conductivity at the candidate, drawn from the current
%              predictive distribution and corrupted by the observation noise sigma_y
%    w_s(y)    self-normalized importance weights proportional to p(y | theta_s, xi), Eq. (12)
%    U(xi)     the entropic-OT squared distance (13) between the equal-weight measure and the
%              reweighted one on the SAME support, averaged over simulated data
%    Lambda    a diagonal metric built from the normalized DRASTIC field, applied to the ground
%              cost as in Proposition 2, not as a scalar multiplier on the utility
%    greedy    after each placement the sample weights are updated with a fantasy datum equal to
%              the current predictive mean, so overlap between wells is handled by the posterior
%              rather than by an ad hoc distance penalty
%
%  It also answers the comparison an earlier draft promised and never delivered (verification
%  ladder item iv; Reviewer 4 point 6): four designs on the IDENTICAL posterior, scored under two
%  metrics, as a function of the number of wells.
%
%  No training happens here. The posterior is the one already stored in results/trained_params.mat
%  by main_fushekuqe.m, the same one behind Figures 5 and 6, so the design study and the maps
%  cannot drift apart.
%
%  Outputs: results/oed_utility_field.csv      per candidate: U, U_Lambda, trace reduction, w_D
%           results/oed_selected_plain.csv     greedy W2 sequence
%           results/oed_selected_drastic.csv   greedy Lambda-weighted sequence
%           results/oed_design_comparison.csv  five designs x five wells x two metrics
%           results/oed_sensitivity.csv        rank stability in epsilon, S and sigma_y
%           results/oed_settings.csv           every constant used here
%           figures/fig10_design_comparison.png
%  Figure 9 is drawn afterwards by make_fig9_oed.m from these CSVs.

clear; clc; close all; addpath(fullfile(pwd,'src'));
OL = oed_lib();
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');

SEED = 20260610;
rng(SEED,'twister');

%% ------------------------------------------------------------------ settings
oed.S        = 200;    % posterior samples theta_s (MC-dropout passes)
oed.nCand    = 400;    % candidate locations
% Simulated data per candidate for the outer expectation of (11). This was 32, and at that size
% the per-candidate utility was dominated by Monte Carlo error: two runs of the identical
% configuration agreed on the ranking only to a Spearman of 0.45, and the vulnerability-weighted
% design scored higher on the UNWEIGHTED metric than the design that optimizes it, which cannot
% happen except through noise. Coupling the draws across candidates helped only marginally,
% 0.45 to 0.49, because it removes noise within a call and not between calls. Eight times the
% sample cuts the error by a factor of about three.
oed.nY       = 256;
oed.eps      = 0.05;   % entropic regularization, on the mean-normalized ground cost
oed.nIter    = 200;    % Sinkhorn iterations
oed.kSelect  = 5;      % wells placed
oed.nRandRep = 20;     % replicates for the random design
oed.forceCPU = true;   % the evaluation is cheap; leave the GPU to the training scripts

%% ------------------------------------------------------- posterior and geometry
LD = load(fullfile('..','results','trained_params.mat'));   % params, cfg, sc
params = LD.params; cfg = LD.cfg; sc = LD.sc;
cfg.forceCPU = oed.forceCPU;
cfg.useGPU   = ~cfg.forceCPU && gpuDeviceCount > 0;
fprintf('[OED] posterior loaded from results/trained_params.mat (no retraining).\n');
fprintf('[OED] loss weights of that model: L1=%g L2=%g L3=%g L4=%g L5=%g L6=%g L7=%g\n', ...
        cfg.w.L1, cfg.w.L2, cfg.w.L3, cfg.w.L4, cfg.w.L5, cfg.w.L6, cfg.w.L7);
assert(cfg.w.L4 == 0, 'trained_params.mat is not a v2.7 model: L4 is still on.');

D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
GD = readtable(fullfile('..','data','drastic_grid_180.csv'));

xw   = [D.x_km, D.y_km]';                  % 2 x 37 wells, km
Kobs = D.K_mday;
xb = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';
dseg = hypot(diff(xb(1,:)), diff(xb(2,:))); xb = xb(:, [true, dseg>1e-4]);
xg = [ (GD.X_GK - min(Bd.X_GK))/1000, (GD.Y_GK - min(Bd.Y_GK))/1000 ]';   % 2 x 180 support
nrm = @(X) [ (X(1,:)-sc.cx)/sc.s ; (X(2,:)-sc.cy)/sc.s ];

polyC = polyshape(xb(1,1:cfg.sampleDecim:end), xb(2,1:cfg.sampleDecim:end), 'Simplify', true);
Xcand = sample_collocation(polyC, oed.nCand);
nG = size(xg,2); nC = size(Xcand,2); nW = size(xw,2);
fprintf('[OED] support %d nodes, %d candidates, %d existing wells.\n', nG, nC, nW);

%% ----------------------------------------- joint posterior samples, support of Eq. (12)
% One dropout mask per pass, evaluated on support, candidates and wells TOGETHER, so that the
% field sample and the datum predictor in the same row come from the same network realization.
Xall  = [xg, Xcand, xw];
XallN = dlarray(cast(nrm(Xall), cfg.prec));
if cfg.useGPU, XallN = gpuArray(XallN); end

THK = zeros(oed.S, nG);     % logK on the support
THN = zeros(oed.S, nG);     % n on the support
CK  = zeros(oed.S, nC);     % logK at the candidates
WK  = zeros(oed.S, nW);     % logK at the existing wells
fprintf('[OED] %d posterior passes over %d points... ', oed.S, size(Xall,2)); t0 = tic;
for s = 1:oed.S
    [lk, nn, ~] = forward_all(params, XallN, cfg, true);   % dropout ACTIVE
    lk = double(gather(extractdata(lk)));
    nn = double(gather(extractdata(nn)));
    THK(s,:) = lk(1:nG);
    THN(s,:) = nn(1:nG);
    CK(s,:)  = lk(nG+1 : nG+nC);
    WK(s,:)  = lk(nG+nC+1 : end);
end
fprintf('done (%.1f s).\n', toc(t0));

%% ------------------------------------------------------- observation noise sigma_y
% A new well contributes a pumping-test log-conductivity. Its error is the scatter the trained
% model already fails to explain at the 37 existing wells, which folds measurement error and
% point-to-block representativeness into one number. It is reported, and the ranking is tested
% against halving and doubling it (oed_sensitivity.csv).
resid   = mean(WK,1)' - log(Kobs);
sigma_y = sqrt(mean(resid.^2));
fprintf('[OED] sigma_y = %.4f log-units (in-sample residual RMS at the 37 wells).\n', sigma_y);

%% ------------------------------------- calibrate the posterior before designing on it
% The abstract and section 3.5 both say the design operates on the CALIBRATED posterior. The raw
% dropout cloud is not that, and a design built on it cannot move: section 5.1 shows the ensemble
% is overconfident, so the spread a new well is meant to inform is narrower than the error of the
% well itself. The first run of this script made that concrete, with an effective sample size of
% 199.7 out of 200 after the Bayesian update and a trace reduction indistinguishable from zero.
%
% Two measured factors bring the samples to the calibrated scale. Neither is chosen here; both are
% read from the stored results, so they cannot drift from the numbers in the paper:
%   rTot  the between-member variance that a dropout-only cloud omits, from the bootstrap ensemble
%   qCal  the split-conformal multiplier that replaces 1.645, from conformal_calibration.csv
%
% The conductivity block is inflated and the porosity block is not. There is no porosity
% observation at this site, so no measured basis exists for calibrating that block; inflating it
% by the conductivity factor would stack an assumption on an assumption. The effect is to weight
% porosity relatively less in the ground cost, which is the conservative direction.
BO = readtable(fullfile('..','results','uq_bootstrap_wells.csv'));
CV = readtable(fullfile('..','results','cv_results.csv'));
CF = readtable(fullfile('..','results','conformal_calibration.csv'));
rTot = median(BO.logK_oob_sd) / median(CV.logK_sd);
iq   = find(string(CF.set) == "bootstrap OOB" & string(CF.quantity) == "logK", 1);
assert(~isempty(iq), 'no out-of-bag logK row in conformal_calibration.csv');
qCal = CF.q_full(iq) / 1.645;
infl = rTot * qCal;
fprintf(['[OED] calibrating the posterior: %.2f for the between-member variance the dropout ' ...
         'cloud omits,\n      %.2f for the conformal multiplier (q = %.3f against 1.645), ' ...
         'product %.2f\n'], rTot, qCal, CF.q_full(iq), infl);
spreadRaw = median(std(CK, 0, 1));
mK = mean(THK, 1);  THK = mK + infl * (THK - mK);
mC = mean(CK,  1);  CK  = mC + infl * (CK  - mC);
fprintf(['[OED] candidate spread %.4f raw, %.4f calibrated, against sigma_y %.4f ' ...
         '(ratio %.2f)\n'], spreadRaw, median(std(CK, 0, 1)), sigma_y, ...
        median(std(CK, 0, 1)) / sigma_y);

%% --------------------------------------------------------------- ground cost, Eq. (10)
% Both blocks are standardized so that neither dominates the metric through its physical units,
% and the two scale factors are written to oed_settings.csv. Lambda is the normalized DRASTIC
% field rescaled to mean one, so the weighted and unweighted costs stay on the same scale.
sK = std(THK(:));  sN = std(THN(:));
ZK = THK / sK;     ZN = THN / sN;
Dn  = (GD.DRASTIC - min(GD.DRASTIC)) / (max(GD.DRASTIC) - min(GD.DRASTIC));
lam = Dn(:)' / mean(Dn);                        % 1 x nG, mean 1

C  = OL.cost_matrix(ZK, ZN, ones(1,nG)) / nG;        % plain      ||theta_s - theta_s'||^2 / G
CL = OL.cost_matrix(ZK, ZN, lam)        / sum(lam);  % Lambda-weighted, Proposition 2
cS  = mean(C(:));   Cn  = C  / cS;              % mean-normalized, so epsilon is dimensionless
cLS = mean(CL(:));  CLn = CL / cLS;
fprintf('[OED] mean ground cost: %.4f plain, %.4f Lambda-weighted.\n', cS, cLS);

%% --------------------------------------- Sinkhorn self-check against the 1-D closed form
% The support of the design problem is one point cloud with two sets of weights, so the check is
% run on the case with a known answer: two equally weighted one-dimensional Gaussian clouds, for
% which W2^2 is the mean squared gap of the order statistics.
gA = randn(oed.S,1); gB = 1.7*randn(oed.S,1) + 0.8;
w2exact = mean((sort(gA) - sort(gB)).^2);
u1 = ones(oed.S,1)/oed.S;
Cg  = (gA - gB').^2; cg = mean(Cg(:));
Caa = (gA - gA').^2; caa = mean(Caa(:));
Cbb = (gB - gB').^2; cbb = mean(Cbb(:));
rawS = OL.sinkhorn(Cg/cg,   u1, u1, oed.eps, oed.nIter) * cg;
sAA  = OL.sinkhorn(Caa/caa, u1, u1, oed.eps, oed.nIter) * caa;
sBB  = OL.sinkhorn(Cbb/cbb, u1, u1, oed.eps, oed.nIter) * cbb;
divS = rawS - 0.5*sAA - 0.5*sBB;
fprintf('[OED] solver check on a case with a known answer: exact %.5f\n', w2exact);
fprintf('      raw entropic %.5f (gap %.2f%%), debiased divergence %.5f (gap %.2f%%)\n', ...
        rawS, 100*abs(rawS-w2exact)/w2exact, divS, 100*abs(divS-w2exact)/w2exact);

%% ------------------------------------------------------ utility of every candidate, Eq. (11)
aUni = ones(oed.S,1)/oed.S;
fprintf('[OED] scoring %d candidates, %d simulated data each...\n', nC, oed.nY);
tS = tic;
[U,  dTr] = OL.score_all(CK, Cn,  cS,  aUni, sigma_y, oed, ZK, ZN, ones(1,nG));
[UL, ~  ] = OL.score_all(CK, CLn, cLS, aUni, sigma_y, oed, ZK, ZN, lam);
fprintf('[OED] scoring done (%.1f s).\n', toc(tS));

%% ------------------------------------------------------------------ greedy designs
fprintf('[OED] greedy sequences...\n');
selW2  = OL.greedy(CK, Cn,  cS,  sigma_y, oed, ZK, ZN, ones(1,nG), 'w2');
selLam = OL.greedy(CK, CLn, cLS, sigma_y, oed, ZK, ZN, lam,        'w2');
selA   = OL.greedy(CK, Cn,  cS,  sigma_y, oed, ZK, ZN, ones(1,nG), 'trace');

selRnd = cell(oed.nRandRep,1);
for r = 1:oed.nRandRep
    selRnd{r} = randperm(nC, oed.kSelect);
end
selSpace = maximin_design(Xcand, xw, oed.kSelect);

%% -------------------------------------------------- head-to-head on the same posterior
% Same update AND same randomness for every design: entry k is cumulative E[W2^2] and cumulative
% trace removed, so a gap between two rows is the designs, not the draws.
% Draws placed here, after every selection above, so the greedy sequences do not depend on them.
% They still change whenever the posterior changes.
rng(SEED + 777);
U01 = rand(oed.nY, oed.kSelect);      % uniforms, mapped through each design's own weights
Zn  = randn(oed.nY, oed.kSelect);     % observation noise, shared across designs
names = {'W2', 'W2-Lambda', 'A-optimal', 'space-filling', 'random'};
picks = {selW2, selLam, selA, selSpace, []};
rows  = {};
for d = 1:numel(names)
    if strcmp(names{d}, 'random')
        Wm = zeros(oed.nRandRep, oed.kSelect); Tm = Wm;
        for r = 1:oed.nRandRep
            [Wm(r,:), Tm(r,:)] = OL.replay(selRnd{r}, CK, Cn, cS, sigma_y, oed, ZK, ZN, ...
                                           U01, Zn);
        end
        w2c = mean(Wm,1); trc = mean(Tm,1); w2s = std(Wm,0,1);
    else
        [w2c, trc] = OL.replay(picks{d}, CK, Cn, cS, sigma_y, oed, ZK, ZN, U01, Zn);
        w2s = zeros(1, oed.kSelect);
    end
    for k = 1:oed.kSelect
        rows(end+1,:) = {string(names{d}), k, w2c(k), w2s(k), trc(k)}; %#ok<AGROW>
    end
end
CMP = cell2table(rows, 'VariableNames', ...
      {'design','n_wells','W2sq_cumulative','W2sq_sd','trace_removed_frac'});
writetable(CMP, fullfile('..','results','oed_design_comparison.csv'));
fprintf('\n============= DESIGN COMPARISON, identical posterior =============\n');
disp(CMP);

%% ------------------------------------------------------------------- sensitivity
% Section 3.5.2 claims the RANKING, not the value, is stable in epsilon and S. Test it, and test
% sigma_y too, since that constant is an assumption rather than a measurement.
srows = {};
for e = [0.02 0.05 0.10]
    o2 = oed; o2.eps = e;
    Ue = OL.score_all(CK, Cn, cS, aUni, sigma_y, o2, ZK, ZN, ones(1,nG));
    srows(end+1,:) = {'epsilon', e, spearman(U, Ue), top5_overlap(U, Ue)}; %#ok<AGROW>
end
for f = [0.5 1.0 2.0]
    Us = OL.score_all(CK, Cn, cS, aUni, f*sigma_y, oed, ZK, ZN, ones(1,nG));
    srows(end+1,:) = {'sigma_y factor', f, spearman(U, Us), top5_overlap(U, Us)}; %#ok<AGROW>
end
for Ssub = [100 150 200]
    idx = 1:Ssub;
    o2 = oed; o2.S = Ssub;
    Cs = C(idx,idx); css = mean(Cs(:));
    Uz = OL.score_all(CK(idx,:), Cs/css, css, ones(Ssub,1)/Ssub, sigma_y, o2, ...
                   ZK(idx,:), ZN(idx,:), ones(1,nG));
    srows(end+1,:) = {'S', Ssub, spearman(U, Uz), top5_overlap(U, Uz)}; %#ok<AGROW>
end
SEN = cell2table(srows, 'VariableNames', {'knob','value','spearman_vs_canonical','top5_overlap'});
writetable(SEN, fullfile('..','results','oed_sensitivity.csv'));
fprintf('\n---- ranking stability (Spearman against the canonical setting) ----\n');
disp(SEN);

%% ------------------------------------------------------------------------ outputs
Tall = table(Xcand(1,:)', Xcand(2,:)', U(:), UL(:), interp_drastic(xg, Dn, Xcand)', dTr(:), ...
     'VariableNames', {'x_km','y_km','W2sq','W2sq_weighted','DRASTICw','trace_removed_frac'});
writetable(Tall, fullfile('..','results','oed_utility_field.csv'));

Tp = table((1:oed.kSelect)', Xcand(1,selW2)',  Xcand(2,selW2)',  U(selW2)', ...
     'VariableNames', {'rank','x_km','y_km','W2sq_utility'});
Tw = table((1:oed.kSelect)', Xcand(1,selLam)', Xcand(2,selLam)', UL(selLam)', ...
     'VariableNames', {'rank','x_km','y_km','W2sq_weighted'});
writetable(Tp, fullfile('..','results','oed_selected_plain.csv'));
writetable(Tw, fullfile('..','results','oed_selected_drastic.csv'));

SET = table({'S';'nCand';'nY';'epsilon';'nIter';'kSelect';'sigma_y';'sd_logK';'sd_n';'seed'}, ...
            [oed.S; oed.nCand; oed.nY; oed.eps; oed.nIter; oed.kSelect; sigma_y; sK; sN; SEED], ...
            'VariableNames', {'setting','value'});
writetable(SET, fullfile('..','results','oed_settings.csv'));

fig10(CMP, oed);

fprintf('\nWritten: results/oed_utility_field.csv, oed_selected_plain.csv, oed_selected_drastic.csv,\n');
fprintf('         oed_design_comparison.csv, oed_sensitivity.csv, oed_settings.csv\n');
fprintf('         figures/fig10_design_comparison.png\n');
fprintf('Run make_fig9_oed.m for Figure 9.\n');


%% ======================================================================= functions

function sel = maximin_design(Xcand, xw, k)
% Space-filling: each new point maximizes its distance to the existing wells and to the points
% already chosen. No posterior enters, which is the point of including it.
    sel = zeros(1,k);
    ref = xw;
    for i = 1:k
        d = min(pdist2c(Xcand, ref), [], 2);
        [~, b] = max(d);
        sel(i) = b;
        ref = [ref, Xcand(:,b)]; %#ok<AGROW>
    end
end

function Dm = pdist2c(A, B)
    Dm = sqrt( (A(1,:)' - B(1,:)).^2 + (A(2,:)' - B(2,:)).^2 );
end

function r = spearman(x, y)
    [~,ix] = sort(x); rx = zeros(size(x)); rx(ix) = 1:numel(x);
    [~,iy] = sort(y); ry = zeros(size(y)); ry(iy) = 1:numel(y);
    rx = rx - mean(rx); ry = ry - mean(ry);
    r = sum(rx.*ry) / sqrt(sum(rx.^2)*sum(ry.^2));
end

function o = top5_overlap(x, y)
    [~,ix] = sort(x,'descend'); [~,iy] = sort(y,'descend');
    o = numel(intersect(ix(1:5), iy(1:5))) / 5;
end

function wD = interp_drastic(xg, Dn, Xc)
    wD = zeros(1, size(Xc,2));
    for c = 1:size(Xc,2)
        [~,j] = min(hypot(xg(1,:)-Xc(1,c), xg(2,:)-Xc(2,c)));
        wD(c) = Dn(j);
    end
end

function fig10(CMP, oed)
% The comparison Reviewer 1 point 20 and Reviewer 4 point 6 asked for: every design on the same
% posterior, under both metrics, as a function of how many wells are drilled.
    f = figure('Color','w','Units','centimeters','Position',[2 2 17.4 7.2]);
    nm = unique(CMP.design, 'stable');
    mk = {'o','s','^','d','v'};
    for panel = 1:2
        subplot(1,2,panel); hold on; box on
        for i = 1:numel(nm)
            sel = CMP.design == nm(i);
            if panel == 1
                yv = CMP.W2sq_cumulative(sel);
            else
                yv = 100*CMP.trace_removed_frac(sel);
            end
            plot(CMP.n_wells(sel), yv, ['-' mk{i}], 'LineWidth', 1.2, 'MarkerSize', 5);
        end
        xlabel('number of new wells'); xlim([0.7 oed.kSelect+0.3]); xticks(1:oed.kSelect);
        if panel == 1
            ylabel('cumulative expected W_2^2');
            title('(a) transport utility', 'FontWeight','normal');
            legend(cellstr(nm), 'Location','northwest', 'Box','off', 'FontSize', 7);
        else
            ylabel('posterior trace removed (%)');
            title('(b) variance reduction', 'FontWeight','normal');
        end
        set(gca,'FontSize',8);
    end
    set(f,'PaperUnits','centimeters','PaperPosition',[0 0 17.4 7.2],'PaperSize',[17.4 7.2]);
    print(f, fullfile('..','figures','fig10_design_comparison.png'), '-dpng', '-r600');
    close(f);
end
