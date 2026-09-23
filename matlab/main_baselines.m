%% Baselines under the IDENTICAL 5-fold CV protocol
%  Folds reproduced exactly: same seed -> same pool draw -> same permutation.
%  All fitting (variogram, regression) uses TRAIN wells only per fold.
%
%  v2.7.0 adds the two baselines Reviewer 4 asked for and the earlier version did not have:
%    heads     ordinary kriging of the 37 head observations, on the same folds. The claim that
%              the framework is "the only method that predicts heads" was true only because
%              kriging had been applied to conductivity alone (Reviewer 4, point 4).
%    porosity  the Kozeny-Carman inverse of the kriged conductivity, which is the porosity field
%              any K estimator can produce (Reviewer 4, point 5). There is no measured porosity
%              to score it against here; that comparison is made on the known field in
%              main_synthetic.m. What this row shows is how far the two fields differ.
clear; clc; close all; addpath(fullfile(pwd,'src'));
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
warning('off','MATLAB:polyshape:repairedBySimplify');
rng(20260610,'twister');

%% data
D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
G  = readtable(fullfile('..','data','drastic_grid_180.csv'));
xw = [D.x_km, D.y_km]; lgK = log(D.K_mday);
xb = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';
dseg = hypot(diff(xb(1,:)), diff(xb(2,:))); xb = xb(:, [true, dseg>1e-4]);
xd = [ (G.X_GK - min(Bd.X_GK))/1000, (G.Y_GK - min(Bd.Y_GK))/1000 ];

%% reproduce EXACT folds of main_cv (same rng consumption order)
cfgTmp.sampleDecim = 4; cfgTmp.poolSize = 40000;
polySamp = polyshape(xb(1,1:cfgTmp.sampleDecim:end), xb(2,1:cfgTmp.sampleDecim:end),'Simplify',true);
dummyPool = sample_collocation(polySamp, cfgTmp.poolSize); clear dummyPool;  % consume identical rand stream
perm = randperm(37);
foldID = zeros(37,1); foldID(perm) = mod(0:36,5)+1;
fcheck = fullfile('..','results','cv_folds.csv');
if isfile(fcheck)
    fSaved = readmatrix(fcheck);
    assert(isequal(fSaved, foldID), 'Fold mismatch vs cv_folds.csv - investigate before comparing!');
    fprintf('Folds verified identical to main_cv run. \n');
else
    fprintf('NOTE: results/cv_folds.csv not found (older main_cv). Folds reproduced by seed replay.\n');
end

%% DRASTIC interpolated to wells (feature for RK / RF)
Fd = scatteredInterpolant(xd(:,1), xd(:,2), G.DRASTIC, 'natural','nearest');
Dw = Fd(xw(:,1), xw(:,2)); Dz = (Dw - mean(Dw))/std(Dw);

%% containers
M = struct('OK',[], 'RK',[], 'RF',[]);
pred = nan(37,3); psd = nan(37,3); haveRF = true;
hpred = nan(37,1); hsd = nan(37,1);          % ordinary kriging of heads (Reviewer 4, point 4)
nKrig = nan(37,1);                           % Kozeny-Carman inverse of the kriged logK (point 5)
hobs = D.head_m;
fac  = string(D.facies);
nPri = D.n_mean;

for f = 1:5
    te = find(foldID==f); tr = find(foldID~=f);
    % ---------- Ordinary Kriging of the HEAD field, same folds ----------
    [ah,c0h,c1h] = fit_expvario(xw(tr,:), hobs(tr));
    [hpred(te), hsd(te)] = ok_predict(xw(tr,:), hobs(tr), xw(te,:), ah, c0h, c1h);
    % ---------- Ordinary Kriging (exponential, train-only fit) ----------
    [a,c0,c1] = fit_expvario(xw(tr,:), lgK(tr));
    [pred(te,1), psd(te,1)] = ok_predict(xw(tr,:), lgK(tr), xw(te,:), a, c0, c1);
    % ---------- Regression-Kriging on DRASTIC ----------
    b = [ones(numel(tr),1) Dz(tr)] \ lgK(tr);            % train-only OLS
    res = lgK(tr) - [ones(numel(tr),1) Dz(tr)]*b;
    [ar,c0r,c1r] = fit_expvario(xw(tr,:), res);
    [rk_res, rk_sd] = ok_predict(xw(tr,:), res, xw(te,:), ar, c0r, c1r);
    pred(te,2) = [ones(numel(te),1) Dz(te)]*b + rk_res;  psd(te,2) = rk_sd;
    % ---------- Random Forest (if toolbox present) ----------
    if haveRF
        try
            X = [xw(tr,:), Dz(tr)];
            B = TreeBagger(300, X, lgK(tr), 'Method','regression','MinLeafSize',3);
            Pt = nan(numel(te), 300);
            Xte = [xw(te,:), Dz(te)];
            for t = 1:300, Pt(:,t) = predict(B.Trees{t}, Xte); end
            pred(te,3) = mean(Pt,2); psd(te,3) = std(Pt,0,2);
        catch ME
            haveRF = false; fprintf('RF skipped (%s)\n', ME.message);
        end
    end
    % ---------- porosity from the kriged conductivity, same transform as the model ----------
    % beta is calibrated per facies on the TRAIN wells only, exactly as main_fushekuqe
    % initializes it, so the held-out wells never enter their own transform.
    kc = @(nn) log(nn.^3 ./ (1-nn).^2);
    bF = containers.Map('KeyType','char','ValueType','double');
    for cls = unique(fac)'
        s = tr(fac(tr) == cls);
        if isempty(s), s = tr; end
        bF(char(cls)) = mean(lgK(s) - kc(nPri(s)));
    end
    for q = te'
        b0 = bF(char(fac(q)));
        nKrig(q) = invert_kc(pred(q,1) - b0, 0.20, 0.40);
    end
    fprintf('fold %d done\n', f);
end

%% summary table
names = {'OK','RK+DRASTIC','RF'};
fprintf('\n========= BASELINES - identical 5-fold CV (n=37) =========\n');
fprintf('%-12s  %-6s  %-6s  %-8s  %-9s\n','method','RMSE','MAE','factor','cov90');
out = table();
for m = 1:3
    if m==3 && ~haveRF, fprintf('%-12s  (skipped: Statistics Toolbox unavailable)\n','RF'); continue; end
    e = pred(:,m) - lgK;
    cv90 = mean( abs(e) <= 1.645*psd(:,m) );
    fprintf('%-12s  %.3f  %.3f  %.2f      %.0f%%\n', names{m}, sqrt(mean(e.^2)), mean(abs(e)), exp(sqrt(mean(e.^2))), 100*cv90);
    out.(matlab.lang.makeValidName(names{m})) = pred(:,m);
    out.([matlab.lang.makeValidName(names{m}) '_sd']) = psd(:,m);   % v2.8: so the coverage can be recomputed
end
% the reference row is read from the stored run, never typed in
fcv = fullfile('..','results','cv_results.csv');
if isfile(fcv)
    R = readtable(fcv);
    eR = R.logK_pred - R.logK_obs;
    fprintf('(DPC-PINN from results/cv_results.csv: RMSE %.3f, MAE %.3f, factor %.2f)\n', ...
            sqrt(mean(eR.^2)), mean(abs(eR)), exp(sqrt(mean(eR.^2))));
else
    fprintf('(results/cv_results.csv not present, so no DPC-PINN reference row is printed)\n');
end

%% heads: kriging against the physics-based prediction
eh = hpred - hobs;
% climatology on the same folds: each held-out well predicted by the mean of its training wells
% (v2.8; until v2.7 the mean of all 37, an in-sample figure no fold could produce)
clim = hobs - arrayfun(@(i) mean(hobs(foldID ~= foldID(i))), (1:numel(hobs))');
fprintf('\n========= HEADS, same folds (Reviewer 4, point 4) =========\n');
fprintf('%-22s RMSE %6.2f m   MAE %6.2f m   cov90 %3.0f%%\n', 'ordinary kriging', ...
        sqrt(mean(eh.^2)), mean(abs(eh)), 100*mean(abs(eh) <= 1.645*hsd));
fprintf('%-22s RMSE %6.2f m\n', 'climatological mean', sqrt(mean(clim.^2)));
if isfile(fcv) && any(strcmp(R.Properties.VariableNames, 'h_pred'))
    ep = R.h_pred - R.h_obs;
    fprintf('%-22s RMSE %6.2f m   MAE %6.2f m\n', 'DPC-PINN', ...
            sqrt(mean(ep.^2)), mean(abs(ep)));
end

%% porosity: what any K estimator can produce
fprintf('\n========= POROSITY from kriged K (Reviewer 4, point 5) =========\n');
fprintf('kriging-derived n at the held-out wells: %.3f to %.3f, mean %.3f\n', ...
        min(nKrig), max(nKrig), mean(nKrig));
fprintf(['There is no measured porosity to score this against. The comparison that can be ' ...
         'scored\nis on the known field in main_synthetic.m.\n']);

out.Well_ID = D.Well_ID; out.logK_obs = lgK; out.fold = foldID;
out.head_obs = hobs; out.head_OK = hpred; out.head_OK_sd = hsd; out.n_from_OK = nKrig;
writetable(out, fullfile('..','results','baselines_results.csv'));
fprintf('\nPer-well predictions written to results/baselines_results.csv\n');

%% ---------- local functions ----------
function [a,c0,c1] = fit_expvario(X, v)
% Empirical semivariogram + exponential fit gamma(h)=c0+c1*(1-exp(-h/a)).
% Toolbox-free: grid search over range a, linear LS for (c0,c1>=0).
n = numel(v); [I,J] = find(triu(true(n),1));
h = hypot(X(I,1)-X(J,1), X(I,2)-X(J,2));
g = 0.5*(v(I)-v(J)).^2;
hs = sort(h);                                        % toolbox-free quantile bins
edges = hs( max(1, round(linspace(1, 0.75*numel(hs), 9))) );
hb=[]; gb=[];
for k=1:numel(edges)-1
    s = h>=edges(k) & h<edges(k+1);
    if nnz(s)>=5, hb(end+1)=mean(h(s)); gb(end+1)=mean(g(s)); end %#ok<AGROW>
end
best = inf;
for a_try = linspace(max(hb)/20, max(hb)*2, 60)
    Phi = [ones(numel(hb),1), 1-exp(-hb(:)/a_try)];
    cc = Phi \ gb(:);  cc = max(cc, 0);
    r = norm(Phi*cc - gb(:));
    if r < best, best = r; a = a_try; c0 = cc(1); c1 = max(cc(2), 1e-6); end
end
end

function n = invert_kc(target, lo, hi)
% Solve log(n^3/(1-n)^2) = target for n by bisection. The map is strictly increasing on (0,1),
% so the bracket is enough; values outside the admissible band are clipped to it.
f = @(nn) log(nn.^3 ./ (1-nn).^2) - target;
if f(lo) > 0, n = lo; return; end
if f(hi) < 0, n = hi; return; end
for it = 1:60
    mid = 0.5*(lo+hi);
    if f(mid) > 0, hi = mid; else, lo = mid; end
end
n = 0.5*(lo+hi);
end

function [mu, sd] = ok_predict(Xtr, v, Xte, a, c0, c1)
% Ordinary kriging with exponential covariance C(h)=c1*exp(-h/a), nugget c0.
n = size(Xtr,1);
Hk = hypot(Xtr(:,1)-Xtr(:,1)', Xtr(:,2)-Xtr(:,2)');   % full distance matrix, base MATLAB
C  = c1*exp(-Hk/a); C(1:n+1:end) = c0 + c1;
A  = [C, ones(n,1); ones(1,n), 0];
mu = nan(size(Xte,1),1); sd = mu;
for q = 1:size(Xte,1)
    h0 = hypot(Xtr(:,1)-Xte(q,1), Xtr(:,2)-Xte(q,2));
    cvec = c1*exp(-h0/a);
    w = A \ [cvec; 1];
    mu(q) = w(1:n)' * v(:);
    sd(q) = sqrt(max(c0 + c1 - w(1:n)'*cvec - w(end), 1e-12));
end
end


