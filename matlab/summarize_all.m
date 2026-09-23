%% Consolidated summary - recomputes EVERYTHING from the per-well files
%  (single source of truth; no numbers carried over from earlier consoles).
clear; clc;
R = readtable(fullfile('..','results','cv_results.csv'));        % DPC-PINN CV
B = readtable(fullfile('..','results','baselines_results.csv')); % baselines
lgK = R.logK_obs; n = height(R);
% Climatological floor on the same folds: each held-out well is predicted by the mean of its
% training wells, as every other method is fitted on the training wells only. Until v2.8 this
% was std() over all 37 wells, an in-sample figure that no fold could have produced.
mK = arrayfun(@(i) mean(R.logK_obs(R.fold ~= R.fold(i))), (1:n)');
mH = arrayfun(@(i) mean(R.h_obs(R.fold ~= R.fold(i))),    (1:n)');
trivK = sqrt(mean((lgK - mK).^2));  trivH = sqrt(mean((R.h_obs - mH).^2));
rows = {};
eK = R.logK_pred - R.logK_obs;  eH = R.h_pred - R.h_obs;
rows(end+1,:) = {'DPC-PINN', sqrt(mean(eK.^2)), mean(abs(eK)), 100*mean(abs(eK)<=1.645*R.logK_sd)};
[~,iR]=sort(string(R.Well_ID)); [~,iB]=sort(string(B.Well_ID));
assert(isequal(R.fold(iR), B.fold(iB)), 'FOLDS DO NOT MATCH between CV and baselines!');
for nm = ["OK","RK_DRASTIC","RF"]
    if ismember(nm, string(B.Properties.VariableNames))
        e = B.(nm)(iB) - B.logK_obs(iB);
        rows(end+1,:) = {char(nm), sqrt(mean(e.^2)), mean(abs(e)), NaN}; %#ok<SAGROW>
    end
end
fprintf('\n================ TABLE 5 DRAFT (logK, n=%d held-out) ================\n', n);
fprintf('%-12s  %-6s  %-6s  %-7s  %-6s\n','method','RMSE','MAE','factor','cov90');
T = table('Size',[0 5],'VariableTypes',{'string','double','double','double','double'}, ...
          'VariableNames',{'method','RMSE','MAE','factor','cov90'});
for r = 1:size(rows,1)
    fprintf('%-12s  %.3f  %.3f  %.2f    %s\n', rows{r,1}, rows{r,2}, rows{r,3}, exp(rows{r,2}), cov2str(rows{r,4}));
    T(end+1,:) = {string(rows{r,1}), rows{r,2}, rows{r,3}, exp(rows{r,2}), rows{r,4}}; %#ok<SAGROW>
end
fprintf('%-12s  %.3f  %s\n','(trivial)', trivK, '- training-fold mean predictor');
T(end+1,:) = {"climatology", trivK, mean(abs(lgK - mK)), exp(trivK), NaN};
fprintf('\nHEAD (DPC-PINN only): RMSE = %.2f m  MAE = %.2f m  cov90 = %.0f%%   (trivial %.2f m)\n', ...
        sqrt(mean(eH.^2)), mean(abs(eH)), 100*mean(abs(eH)<=1.645*R.h_sd), trivH);
T(end+1,:) = {"head DPC-PINN", sqrt(mean(eH.^2)), mean(abs(eH)), NaN, 100*mean(abs(eH)<=1.645*R.h_sd)};
T(end+1,:) = {"head climatology", trivH, mean(abs(R.h_obs - mH)), NaN, NaN};
writetable(T, fullfile('..','results','TABLE5_draft.csv'));
fprintf('Written: results/TABLE5_draft.csv\n');
function s = cov2str(c), if isnan(c), s=' - '; else, s=sprintf('%.0f%%',c); end, end
