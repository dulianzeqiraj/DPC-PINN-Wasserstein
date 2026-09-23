%% Split-conformal calibration of the predictive intervals  (v2.7)
%
%  This stage was missing from the v1.0.0 package: the calibrated coverages quoted in
%  the manuscript had no script behind them. It is written here so that every number
%  in Figure 7 and Section 5.1 comes from a run.
%
%  Two prediction sets are calibrated SEPARATELY, because they are different protocols
%  and the manuscript previously mixed them:
%     (A) 5-fold spatial cross-validation      -> results/cv_results.csv
%     (B) bootstrap out-of-bag                 -> results/uq_bootstrap_wells.csv
%
%  Two conformal estimates are reported for each, because they answer different questions:
%     full   - q is the ceil((n+1)(1-alpha))-th smallest conformity score over all n wells,
%              then applied to those same n wells. In-sample for the quantile, therefore
%              OPTIMISTIC. It is the multiplier you would actually ship.
%     LOO    - for each well i, q is recomputed from the other n-1 wells and tested on i.
%              This is the honest estimate of the procedure's coverage, and it is what the
%              manuscript means by "validated leave-one-out".
%
%  Finite-sample guarantee, split conformal with a calibration set of size m:
%     P(Y in C) >= ceil((m+1)(1-alpha)) / (m+1)
%  For m = 37 and alpha = 0.1 this is 35/38 = 0.9211, NOT 0.87.
%
%  Outputs: results/conformal_calibration.csv, results/conformal_scores.csv
%           and a console block ready to be read into the text.

clear; clc;
warning('off','MATLAB:table:ModifiedAndSavedVarnames');

ALPHA = 0.10;                 % nominal 90% intervals
ZNOM  = 1.645;                % the same constant main_cv.m uses for its cov90 columns

rows = {};                    % {set, quantity, n, raw_cov, q_full, cov_full, cov_loo, bound, worst}
scoreRows = {};

%% ---------------------------------------------------------------- A: 5-fold CV
fCV = fullfile('..','results','cv_results.csv');
if isfile(fCV)
    R = readtable(fCV);
    [rows, scoreRows] = do_set(rows, scoreRows, '5-fold CV', 'logK', ...
        R.logK_obs, R.logK_pred, R.logK_sd, ALPHA, ZNOM, string(R.Well_ID));
    [rows, scoreRows] = do_set(rows, scoreRows, '5-fold CV', 'head', ...
        R.h_obs, R.h_pred, R.h_sd, ALPHA, ZNOM, string(R.Well_ID));
else
    fprintf(2, 'MISSING %s - run main_cv.m first\n', fCV);
end

%% ------------------------------------------------------------ B: bootstrap OOB
fUQ = fullfile('..','results','uq_bootstrap_wells.csv');
if isfile(fUQ)
    U = readtable(fUQ);
    v = ~isnan(U.logK_oob) & ~isnan(U.logK_oob_sd);     % wells with enough OOB members
    [rows, scoreRows] = do_set(rows, scoreRows, 'bootstrap OOB', 'logK', ...
        U.logK_obs(v), U.logK_oob(v), U.logK_oob_sd(v), ALPHA, ZNOM, string(U.Well_ID(v)));
    w = ~isnan(U.h_oob) & ~isnan(U.h_oob_sd);
    [rows, scoreRows] = do_set(rows, scoreRows, 'bootstrap OOB', 'head', ...
        U.h_obs(w), U.h_oob(w), U.h_oob_sd(w), ALPHA, ZNOM, string(U.Well_ID(w)));
else
    fprintf(2, 'MISSING %s - run uq_bootstrap.m first\n', fUQ);
end

if isempty(rows)
    error('No input files found. Run main_cv.m and uq_bootstrap.m first.');
end

%% ------------------------------------------------------------------- report
T = cell2table(rows, 'VariableNames', ...
    {'set','quantity','n','raw_cov_pct','q_full','cov_full_pct','cov_loo_pct','bound','worst_z'});

fprintf('\n=========== SPLIT-CONFORMAL CALIBRATION (alpha = %.2f, nominal %d%%) ===========\n', ...
        ALPHA, round(100*(1-ALPHA)));
fprintf('%-14s %-6s %3s %9s %8s %10s %9s %8s %8s\n', ...
        'set','qty','n','raw cov','q','cov(full)','cov(LOO)','bound','worst');
for i = 1:height(T)
    fprintf('%-14s %-6s %3d %8.1f%% %8.3f %9.1f%% %8.1f%% %8.4f %8.2f\n', ...
        T.set(i), T.quantity(i), T.n(i), T.raw_cov_pct(i), T.q_full(i), ...
        T.cov_full_pct(i), T.cov_loo_pct(i), T.bound(i), T.worst_z(i));
end
fprintf('\nRead the columns as:\n');
fprintf('  raw cov    empirical coverage of mean +- %.3f*sigma_total, before any calibration\n', ZNOM);
fprintf('  q          the conformal multiplier that replaces %.3f\n', ZNOM);
fprintf('  cov(full)  coverage of that multiplier on the wells that produced it (optimistic)\n');
fprintf('  cov(LOO)   coverage when the multiplier is recomputed without the tested well (honest)\n');
fprintf('  bound      finite-sample guarantee ceil((m+1)(1-alpha))/(m+1) for the LOO calibration size\n');
fprintf('  worst      largest standardized residual, i.e. the well no ensemble anticipates\n\n');

writetable(T, fullfile('..','results','conformal_calibration.csv'));
S = cell2table(scoreRows, 'VariableNames', {'set','quantity','Well_ID','residual','sigma','score','covered_loo'});
writetable(S, fullfile('..','results','conformal_scores.csv'));
fprintf('Written: results/conformal_calibration.csv and results/conformal_scores.csv\n');

%% ------------------------------------------------------------------ helpers
function [rows, scoreRows] = do_set(rows, scoreRows, setName, qty, obs, pred, sd, alpha, znom, ids)
    obs = double(obs(:)); pred = double(pred(:)); sd = double(sd(:));
    keep = isfinite(obs) & isfinite(pred) & isfinite(sd) & sd > 0;
    obs = obs(keep); pred = pred(keep); sd = sd(keep); ids = ids(keep);
    n = numel(obs);
    resid = abs(obs - pred);
    score = resid ./ sd;                       % conformity scores

    rawCov = 100 * mean(resid <= znom * sd);

    % full calibration: k-th smallest of all n scores
    s  = sort(score);
    kF = ceil((n + 1) * (1 - alpha));
    if kF > n
        qFull = inf;                            % too few points to certify at this alpha
    else
        qFull = s(kF);
    end
    covFull = 100 * mean(score <= qFull);

    % honest leave-one-out: drop i, recompute the quantile from the other n-1
    covered = false(n,1);
    for i = 1:n
        other = score([1:i-1, i+1:n]);
        m  = numel(other);                      % = n-1
        so = sort(other);
        k  = ceil((m + 1) * (1 - alpha));
        if k > m
            qi = inf;
        else
            qi = so(k);
        end
        covered(i) = score(i) <= qi;
    end
    covLoo = 100 * mean(covered);

    m     = n - 1;                              % calibration size seen by the LOO protocol
    bound = min(ceil((m + 1) * (1 - alpha)), m + 1) / (m + 1);

    rows(end+1,:) = {string(setName), string(qty), n, rawCov, qFull, covFull, covLoo, bound, max(score)}; %#ok<AGROW>
    for i = 1:n
        scoreRows(end+1,:) = {string(setName), string(qty), ids(i), resid(i), sd(i), score(i), covered(i)}; %#ok<AGROW>
    end
end
