%% Figure 8 - what Table 4 cannot show   (v2.7.0)
%
%  The June figure was a bar chart of the same held-out errors the table already listed.
%  Reviewer 3, detail 6, was right that it added nothing. It has been replaced by the two
%  things a table of aggregate scores cannot carry and that two other reviewers asked for.
%
%  Panel (a), Reviewer 1 point 15. The claim that the methods are "statistically
%  indistinguishable" was asserted without support. This shows the per-fold held-out RMSE of
%  every method, so the reader can see the fold-to-fold spread against which the differences
%  between methods have to be judged, together with the paired per-fold difference between the
%  full model and the best baseline.
%
%  Panel (b), Reviewer 1 point 17 and Reviewer 4 point 3. The three geological and petrophysical
%  priors were previously switched on together, so no term could be credited individually. This
%  shows the main effect of each and every interaction, computed over the full 2^3 factorial
%  under the same five folds. A negative bar means that switching the term on lowers the error.
%
%  Every value is read from a results file. Nothing is typed in.
%
%  Input : results/cv_results.csv, results/baselines_results.csv,
%          results/ablation_factorial_effects.csv, results/ablation_factorial_wells.csv
%  Output: figures/fig8_rmse_bars.png

clear; clc; close all;
LOGLABEL = 'log K';

R = readtable(fullfile('..','results','cv_results.csv'));
B = readtable(fullfile('..','results','baselines_results.csv'));

%% ---------------- panel (a): per-fold scores -------------------------------------------
folds = unique(R.fold);
meth = {'DPC-PINN', 'Ordinary kriging', 'RK + DRASTIC', 'Random forest'};
perFold = nan(numel(folds), numel(meth));
for k = 1:numel(folds)
    s = R.fold == folds(k);
    perFold(k,1) = sqrt(mean((R.logK_pred(s) - R.logK_obs(s)).^2));
    sB = B.fold == folds(k);
    perFold(k,2) = sqrt(mean((B.OK(sB)         - B.logK_obs(sB)).^2));
    perFold(k,3) = sqrt(mean((B.RK_DRASTIC(sB) - B.logK_obs(sB)).^2));
    if any(strcmp(B.Properties.VariableNames,'RF'))
        perFold(k,4) = sqrt(mean((B.RF(sB) - B.logK_obs(sB)).^2));
    end
end
pooled = [sqrt(mean((R.logK_pred - R.logK_obs).^2)), ...
          sqrt(mean((B.OK         - B.logK_obs).^2)), ...
          sqrt(mean((B.RK_DRASTIC - B.logK_obs).^2)), ...
          NaN];
if any(strcmp(B.Properties.VariableNames,'RF'))
    pooled(4) = sqrt(mean((B.RF - B.logK_obs).^2));
end

% paired difference against the best baseline, per fold
[~, jBest] = min(pooled(2:end)); jBest = jBest + 1;
dPair = perFold(:,1) - perFold(:,jBest);
tstat = mean(dPair) / (std(dPair)/sqrt(numel(dPair)));
fprintf('paired per-fold difference, DPC-PINN minus %s:\n', meth{jBest});
fprintf('  mean %+0.4f, sd %0.4f, n = %d folds, t = %+0.2f\n', ...
        mean(dPair), std(dPair), numel(dPair), tstat);
fprintf('  per fold: %s\n', mat2str(round(dPair',4)));

%% ---------------- panel (b): what each prior contributes ---------------------------------
% Preferred source is the full factorial, which gives main effects and interactions. When only
% the one-at-a-time modes have been run, fall back to those: each term against the
% physics-and-data baseline, which is what Reviewer 1 point 17 asks for and what the five
% completed modes deliver on their own. The panel says which of the two it is drawing.
fE = fullfile('..','results','ablation_factorial_effects.csv');
fR = fullfile('..','results','ablation_factorial_rows.csv');
haveE = false; panelB = '';
if isfile(fE)
    % Delimiter must be explicit. Effect names are "L5 x L6", so autodetection picks the space:
    % 7 rows read back as 4 x 5, first data row promoted to header, panel (b) drawn empty with
    % y-labels L5, L6, L5, L5. Only this file of the eight (diag_reads.m).
    E = readtable(fE, 'Delimiter', ',', 'VariableNamingRule', 'preserve');
    % Columns by position, not by name: this file has a fixed two-column layout.
    lab = string(E{:,1});
    v = E{:,2};
    if iscell(v), v = str2double(v); end     % readtable may hand back text
    v = double(v);
    assert(numel(v) == 7 && ~any(isnan(v)), ...
           'effects file read back as %d values with %d NaN', numel(v), sum(isnan(v)));
    haveE = true;
    panelB = '(b) main effects and interactions';
elseif isfile(fR)
    Ab = readtable(fR);
    mode = string(Ab.mode);
    ib = find(mode == "base", 1);
    if ~isempty(ib)
        pick = ["+L5", "L5 anisotropy"; "+L6", "L6 Kozeny-Carman"; "+L7", "L7 channel geometry"];
        lab = strings(0,1); v = [];
        for k = 1:size(pick,1)
            j = find(mode == pick(k,1), 1);
            if ~isempty(j)
                lab(end+1,1) = pick(k,2);                              %#ok<AGROW>
                v(end+1,1) = Ab.logK_RMSE(j) - Ab.logK_RMSE(ib);       %#ok<AGROW>
            end
        end
        haveE = ~isempty(v);
        panelB = '(b) contribution of each prior, one at a time';
        fprintf(['note: the full factorial is incomplete, so panel (b) shows the ' ...
                 'one-at-a-time contributions against the baseline of %.4f\n'], Ab.logK_RMSE(ib));
    end
end
if ~haveE
    fprintf('note: no ablation summary found, panel (b) omitted\n');
end

%% ---------------- draw -------------------------------------------------------------------
fig = figure('Color','w','Units','centimeters','Position',[2 2 17.4 7.6]);

subplot(1,2,1); hold on; box on
cols = [0.10 0.40 0.68; 0.42 0.47 0.49; 0.62 0.66 0.70; 0.80 0.55 0.20];
for m = 1:numel(meth)
    if all(isnan(perFold(:,m))), continue; end
    jit = (m - (numel(meth)+1)/2) * 0.13;
    plot(m + 0*folds + jit*0, perFold(:,m), 'o', 'MarkerSize', 4.5, ...
         'MarkerFaceColor', cols(m,:), 'MarkerEdgeColor', 'none');
    plot([m-0.28 m+0.28], [pooled(m) pooled(m)], '-', 'Color', cols(m,:), 'LineWidth', 1.8);
end
set(gca, 'XTick', 1:numel(meth), 'XTickLabel', meth, 'XTickLabelRotation', 22, 'FontSize', 8);
xlim([0.5 numel(meth)+0.5]);
ylabel(sprintf('held-out %s RMSE per fold', LOGLABEL));
title(sprintf('(a) per fold; paired mean %+0.3f', mean(dPair)), 'FontWeight','normal');
set(gca, 'TitleFontSizeMultiplier', 1);   % the longer title ran over the y-axis labels

if haveE
    subplot(1,2,2); hold on; box on
    [v, ord] = sort(v, 'descend');
    lab = lab(ord);
    for i = 1:numel(v)
        c = [0.72 0.36 0.30];
        if v(i) < 0, c = [0.24 0.48 0.36]; end
        barh(i, v(i), 0.6, 'FaceColor', c, 'EdgeColor', [0.25 0.25 0.25]);
    end
    xline(0, 'k-', 'LineWidth', 0.8);
    xr = [min([v(:); 0]) max([v(:); 0])]; pad = 0.12 * diff(xr);
    xlim([xr(1) - pad, xr(2) + pad]);      % the bars touched the frame with the automatic limits
    set(gca, 'YTick', 1:numel(v), 'YTickLabel', lab, 'FontSize', 8, ...
        'YLim', [0.4 numel(v)+0.6]);
    xlabel(sprintf('change in held-out %s RMSE', LOGLABEL));
    title(panelB, 'FontWeight','normal');
    set(gca, 'TitleFontSizeMultiplier', 1);
end

set(fig,'PaperUnits','centimeters','PaperPosition',[0 0 17.4 7.6],'PaperSize',[17.4 7.6]);
print(fig, fullfile('..','figures','fig8_rmse_bars.png'), '-dpng', '-r600');
fprintf('Written: figures/fig8_rmse_bars.png\n');

T = table(folds, perFold(:,1), perFold(:,2), perFold(:,3), perFold(:,4), dPair, ...
    'VariableNames', {'fold','DPC_PINN','OK','RK_DRASTIC','RF','paired_diff'});
writetable(T, fullfile('..','results','per_fold_rmse.csv'));
fprintf('Written: results/per_fold_rmse.csv\n');
disp(T);
