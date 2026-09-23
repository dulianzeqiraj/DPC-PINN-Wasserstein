%% Figure 11 - truth and recovery on the synthetic aquifer, redrawn from stored results
%
%  main_synthetic.m draws this figure at the end of a two-hour run. Drawing it here instead, from
%  results/synthetic_truth.csv and results/synth/full.mat, means the layout can be changed without
%  retraining anything.
%
%  The aquifer is a narrow north-south strip, so a two by two grid leaves close to half the frame
%  empty and shrinks the maps. Four panels in a row suit the geometry and give each map roughly
%  twice the height at the same printed width.
%
%  Output: figures/fig11_synthetic.png at the placement width used in the manuscript.

clear; clc; close all;

W_CM = 15.5;            % the width the figure is placed at in the document
H_CM = 9.6;

T = readtable(fullfile('..','results','synthetic_truth.csv'));
S = load(fullfile('..','results','synth','full.mat'));
D = readtable(fullfile('..','data','inversion_wells_37.csv'));
B = readtable(fullfile('..','data','boundary_349km2.csv'));

xb = [ (B.X_GK - min(B.X_GK))/1000, (B.Y_GK - min(B.Y_GK))/1000 ]';
xw = [D.x_km, D.y_km]';

% rebuild the evaluation grid from the stored cell centres
gx = unique(round(T.x_km, 6));  gy = unique(round(T.y_km, 6));
[GX, GY] = meshgrid(gx, gy);
ix = interp1(gx, 1:numel(gx), round(T.x_km, 6), 'nearest');
iy = interp1(gy, 1:numel(gy), round(T.y_km, 6), 'nearest');
lin = sub2ind(size(GX), iy, ix);
fprintf('grid %d x %d, %d cells carried\n', numel(gy), numel(gx), height(T));

function M = lay(sz, lin, v)
    M = nan(sz); M(lin) = v;
end

KT = lay(size(GX), lin, T.logK_true);
KR = lay(size(GX), lin, S.muK(:));
NT = lay(size(GX), lin, T.n_true);
NR = lay(size(GX), lin, S.muN(:));

fprintf('logK field RMSE %.4f, correlation %.3f\n', ...
        sqrt(mean((S.muK(:)-T.logK_true).^2)), corr(S.muK(:), T.logK_true));
fprintf('n    field RMSE %.4f, correlation %.3f\n', ...
        sqrt(mean((S.muN(:)-T.n_true).^2)), corr(S.muN(:), T.n_true));

fig = figure('Color','w','Units','centimeters','Position',[2 2 W_CM H_CM]);
P  = {KT, KR, NT, NR};
TL = {'(a) log K (truth)', '(b) log K (recovered)', '(c) n (truth)', '(d) n (recovered)'};
LIM = {[min(T.logK_true) max(T.logK_true)], [min(T.logK_true) max(T.logK_true)], ...
       [min(T.n_true) max(T.n_true)],       [min(T.n_true) max(T.n_true)]};

% tiledlayout rather than subplot: a colour bar attached to a tile takes its space from the
% layout instead of shrinking the axis, and these maps are tall and narrow, so shrinking the axis
% collapses the map. The first attempt did exactly that.
tl = tiledlayout(1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:4
    ax = nexttile(tl);
    imagesc(gx, gy, P{k}, 'AlphaData', ~isnan(P{k}));
    set(gca, 'YDir', 'normal'); hold on
    plot(xb(1,:), xb(2,:), 'k-', 'LineWidth', 0.4);
    plot(xw(1,:), xw(2,:), '^', 'MarkerSize', 2.4, 'MarkerFaceColor', [0.15 0.15 0.15], ...
         'MarkerEdgeColor', 'none');
    axis equal tight
    clim(LIM{k});
    title(TL{k}, 'FontWeight', 'normal', 'FontSize', 8);
    % Easting/Northing, not x/y: matches src/make_map.m, which draws Figures 5 and 6.
    xlabel('Easting (km)', 'FontSize', 7.5);
    if k == 1, ylabel('Northing (km)', 'FontSize', 7.5); else, set(gca, 'YTickLabel', []); end
    set(gca, 'FontSize', 7);
    % One colour bar per pair, on the recovered panel, so the two visibly share a scale. The axis
    % is narrowed first: at the default width the bar sits against the next panel and the tick
    % labels are clipped to their first character, which is how the first draft printed "4." for
    % 4.8. Ticks are set explicitly rather than left to the automatic locator.
    if mod(k, 2) == 0
        c = colorbar(ax, 'eastoutside');
        c.FontSize = 6.5;
        nd = 1; if k == 4, nd = 2; end
        c.Ticks = round(linspace(LIM{k}(1), LIM{k}(2), 4), nd);
        c.TickLabels = compose(sprintf('%%.%df', nd), c.Ticks);
    end
end

set(fig, 'PaperUnits', 'centimeters', 'PaperPosition', [0 0 W_CM H_CM], ...
         'PaperSize', [W_CM H_CM]);
print(fig, fullfile('..','figures','fig11_synthetic.png'), '-dpng', '-r600');
fprintf('Written: figures/fig11_synthetic.png at %.1f x %.1f cm\n', W_CM, H_CM);
