%% Figure 7 - reliability of the predictive intervals   (v2.7)
%
%  This figure had no generating script in the v1.0.0 package: the PNG existed only inside
%  the manuscript, under a path (figs/fig7_reliability.png) that is nowhere in the repository.
%  It is written here so the figure can be regenerated with the rest of the results.
%
%  Two corrections against the version submitted in June:
%    - the axis was labelled log10 K. Every logarithm in this code is natural
%      (main_cv.m and uq_bootstrap.m both use log(K_mday)), so the label was wrong.
%      LOGLABEL below controls it; change it to 'ln K' if you prefer the explicit form.
%    - no numbers are burned into the annotation beyond what the data give, so a rerun
%      cannot leave a stale figure behind.
%
%  Input : results/uq_bootstrap_wells.csv   (out-of-bag predictions and their sigma)
%  Output: figures/fig7_reliability.png
%
%  Protocol, stated so the caption can state it too: the curves are built on the bootstrap
%  OUT-OF-BAG predictions, not on the cross-validation folds. Those are different protocols
%  and they give different coverage; keep the two apart in the text.

clear; clc; close all;
LOGLABEL = 'log K';        % <- 'ln K' if you want the base written out
ALPHAMARK = 0.10;          % the nominal level singled out with a star

F = fullfile('..','results','uq_bootstrap_wells.csv');
assert(isfile(F), 'missing %s - run uq_bootstrap.m first', F);
U = readtable(F);

v = ~isnan(U.logK_oob) & ~isnan(U.logK_oob_sd);
w = ~isnan(U.h_oob)    & ~isnan(U.h_oob_sd);
sets = { struct('obs',U.logK_obs(v),'mu',U.logK_oob(v),'sd',U.logK_oob_sd(v), ...
                'name',sprintf('(a)  %s', LOGLABEL)), ...
         struct('obs',U.h_obs(w),   'mu',U.h_oob(w),   'sd',U.h_oob_sd(w), ...
                'name','(b)  hydraulic head h (m)') };

nom = 0.05:0.01:0.99;                      % nominal coverage grid

fig = figure('Color','w','Position',[100 100 1100 520]);
for p = 1:2
    S = sets{p};
    n = numel(S.obs);
    r = abs(S.obs - S.mu) ./ S.sd;         % standardized residuals = conformity scores
    rs = sort(r);

    covRaw  = zeros(size(nom));
    covConf = zeros(size(nom));
    for k = 1:numel(nom)
        z = norminv(1 - (1-nom(k))/2);     % two-sided normal multiplier
        covRaw(k) = mean(r <= z);
        % v2.8: leave-one-out, as the text reports it. Well i is judged against the quantile of the
        % other n-1 scores, so no well helps to set its own interval. The full-sample version
        % plotted until v2.7 gives 94.6% at 90% for any n = 37 and did not match the 91.9% in the text.
        hit = false(n,1);
        for i = 1:n
            ro = sort(r([1:i-1, i+1:n]));
            idx = ceil(n * nom(k));         % (n-1)+1 calibration points
            if idx > n-1, hit(i) = true; else, hit(i) = r(i) <= ro(idx); end
        end
        covConf(k) = mean(hit);
    end

    % 95% binomial band around the nominal line
    lo = zeros(size(nom)); hi = zeros(size(nom));
    for k = 1:numel(nom)
        se = sqrt(nom(k)*(1-nom(k))/n);
        lo(k) = max(0, nom(k) - 1.96*se);
        hi(k) = min(1, nom(k) + 1.96*se);
    end

    subplot(1,2,p); hold on; box on
    fill([nom fliplr(nom)]*100, [lo fliplr(hi)]*100, [.85 .85 .85], ...
         'EdgeColor','none', 'DisplayName','95% binomial band');
    plot([0 100],[0 100],'k--','LineWidth',1,'DisplayName','perfect calibration');
    plot(nom*100, covRaw*100,  '-o', 'Color',[0.80 0.15 0.10], 'MarkerSize',3, ...
         'MarkerFaceColor',[0.80 0.15 0.10], 'LineWidth',1.4, 'DisplayName','raw \sigma_{tot}');
    plot(nom*100, covConf*100, '-s', 'Color',[0.10 0.40 0.68], 'MarkerSize',3, ...
         'MarkerFaceColor',[0.10 0.40 0.68], 'LineWidth',1.4, 'DisplayName','conformal (leave-one-out)');

    [~, km] = min(abs(nom - (1-ALPHAMARK)));
    plot(nom(km)*100, covRaw(km)*100,  'p', 'MarkerSize',15, ...
         'MarkerFaceColor',[0.80 0.15 0.10], 'MarkerEdgeColor','w', 'HandleVisibility','off');
    plot(nom(km)*100, covConf(km)*100, 'p', 'MarkerSize',15, ...
         'MarkerFaceColor',[0.10 0.40 0.68], 'MarkerEdgeColor','w', 'HandleVisibility','off');
    % labels placed off the curves: raw above-left of its star, conformal below-right of its star
    text(nom(km)*100 - 4, covRaw(km)*100 + 8, sprintf('%.0f%%', 100*covRaw(km)), ...
         'Color',[0.80 0.15 0.10], 'FontWeight','bold', 'HorizontalAlignment','right');
    text(nom(km)*100 - 1, covConf(km)*100 - 13, sprintf('%.1f%%', 100*covConf(km)), ...
         'Color',[0.10 0.40 0.68], 'FontWeight','bold', 'HorizontalAlignment','left');

    axis([0 100 0 100]); axis square
    xlabel('Nominal coverage (%)'); ylabel('Empirical coverage (%)');
    title(S.name, 'FontWeight','normal');
    if p == 1, legend('Location','northwest','Box','on'); end
    fprintf('%s : n=%d, raw cov at %d%% = %.1f%%, conformal = %.1f%%\n', ...
            S.name, n, round(100*(1-ALPHAMARK)), 100*covRaw(km), 100*covConf(km));
end

out = fullfile('..','figures','fig7_reliability.png');
exportgraphics(fig, out, 'Resolution', 400);
fprintf('Written: %s\n', out);
