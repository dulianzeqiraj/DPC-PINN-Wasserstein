%% Figure S1 - directional semivariograms of log-conductivity   (v2.7)
%
%  Third figure with no generating script in the v1.0.0 package. The PNG lived only inside
%  the supplement, and it carried "ln K" on the axis and in the legend while the text now
%  declares log to be the natural logarithm. Same quantity, two notations; this regenerates
%  it under one.
%
%  It depends on nothing but the well table, so it can be run at any time, before or after
%  the inversion.
%
%  Input : data/inversion_wells_37.csv
%  Output: figures/figS1_variogram.png
%
%  Convention: azimuth is measured clockwise from north, so the unit vector for an azimuth a
%  is [sin(a); cos(a)]. A pair is assigned to a direction when the acute angle between its
%  separation vector and that axis is within TOL. Direction is axial, not vectorial: a pair
%  separated towards 135 degrees and one separated towards 315 degrees lie on the same axis.

clear; clc; close all;
LOGLABEL = 'log K';            % matches make_fig7 and make_fig8
AZ_ALONG  = 135;               % NW-SE, the inferred fabric
AZ_ACROSS = 45;                % perpendicular to it
TOL       = 22.5;              % angular tolerance, degrees
EDGES     = 0:2:14;            % lag bins, km

D = readtable(fullfile('..','data','inversion_wells_37.csv'));
xy = [D.x_km, D.y_km];
z  = log(D.K_mday);
n  = numel(z);
sill = var(z);                 % sample variance, the usual sill reference
fprintf('%d wells, sample variance of %s = %.3f\n', n, LOGLABEL, sill);

% all pairs
[I, J] = find(triu(ones(n), 1));
dxy = xy(J,:) - xy(I,:);
h   = hypot(dxy(:,1), dxy(:,2));
% azimuth of each separation, folded onto [0,180) because direction is axial
az  = mod(atan2d(dxy(:,1), dxy(:,2)), 180);
dz2 = (z(J) - z(I)).^2;

[gA, cA, xA] = dirvario(az, h, dz2, AZ_ALONG,  TOL, EDGES);
[gX, cX, xX] = dirvario(az, h, dz2, AZ_ACROSS, TOL, EDGES);

fig = figure('Color','w','Position',[100 100 900 640]); hold on; box on
yline(sill, ':', 'Color',[.45 .45 .45], 'LineWidth',1.4, ...
      'DisplayName', sprintf('sill (Var %s = %.2f)', LOGLABEL, sill));
plot(xA, gA, '-o', 'Color',[0.15 0.42 0.65], 'LineWidth',2, 'MarkerSize',8, ...
     'MarkerFaceColor',[0.15 0.42 0.65], ...
     'DisplayName', sprintf('along-fabric (NW-SE, %g%s)', AZ_ALONG, char(176)));
plot(xX, gX, '--s', 'Color',[0.75 0.20 0.16], 'LineWidth',2, 'MarkerSize',8, ...
     'MarkerFaceColor',[0.75 0.20 0.16], ...
     'DisplayName', sprintf('across-fabric (%g%s)', AZ_ACROSS, char(176)));

for b = 1:numel(xA)
    if cA(b) > 0, text(xA(b), gA(b)+0.018, sprintf('%d', cA(b)), ...
        'Color',[.4 .4 .4], 'HorizontalAlignment','center', 'FontSize',9); end
    if cX(b) > 0, text(xX(b), gX(b)+0.018, sprintf('%d', cX(b)), ...
        'Color',[.4 .4 .4], 'HorizontalAlignment','center', 'FontSize',9); end
end

xlabel('Lag distance h (km)');
ylabel(sprintf('Semivariance \\gamma(h) of %s', LOGLABEL));
legend('Location','northwest','Box','on');
xlim([0 EDGES(end)]); grid on; set(gca,'GridAlpha',0.12);

out = fullfile('..','figures','figS1_variogram.png');
exportgraphics(fig, out, 'Resolution', 400);
fprintf('Written: %s\n', out);
fprintf('%-6s %8s %6s %8s %6s\n', 'lag', 'along', 'pairs', 'across', 'pairs');
for b = 1:numel(xA)
    fprintf('%-6.1f %8.3f %6d %8.3f %6d\n', xA(b), gA(b), cA(b), gX(b), cX(b));
end

%% ---------------- local function (MATLAB requires these at the end of a script) ----------
function [gam, cnt, ctr] = dirvario(az, h, dz2, azTarget, tol, edges)
a = mod(azTarget, 180);
d = abs(az - a);
d = min(d, 180 - d);                      % acute angle to the axis
sel = d <= tol;
gam = nan(1, numel(edges)-1);
cnt = zeros(1, numel(edges)-1);
ctr = nan(1, numel(edges)-1);
for b = 1:numel(edges)-1
    m = sel & h > edges(b) & h <= edges(b+1);
    cnt(b) = nnz(m);
    ctr(b) = (edges(b) + edges(b+1)) / 2;
    if cnt(b) > 0, gam(b) = 0.5 * mean(dz2(m)); end
end
end
