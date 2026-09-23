%% Figure 9 - the two design-utility panels   (v2.7)
%
%  Fifth figure with no generating script in the v1.0.0 package. What main_oed_w2.m draws is a
%  diagnostic: a scatter of the 400 candidates, no legend, and, in panel (b), no existing wells
%  at all, because that subplot never plots them. The manuscript carries a different, polished
%  pair of images that nothing in the repository produces. This script produces that pair.
%
%  Two panels are written separately because the manuscript places them in two separate frames
%  under one caption. Each is drawn at the aspect ratio of the frame Word reserves for it, so
%  neither is stretched: panel (a) 6.85 x 13.17 cm, panel (b) 6.44 x 13.19 cm. Those two frames
%  are not the same width, which is a layout matter for the author, not something to fix here by
%  distorting one of them.
%
%  Input : results/oed_utility_field.csv, oed_selected_plain.csv, oed_selected_drastic.csv
%          data/inversion_wells_37.csv, data/boundary_349km2.csv
%  Output: figures/fig9a_oed_plain.png, figures/fig9b_oed_drastic.png

clear; clc; close all;

U  = readtable(fullfile('..','results','oed_utility_field.csv'));
Sp = readtable(fullfile('..','results','oed_selected_plain.csv'));
Sd = readtable(fullfile('..','results','oed_selected_drastic.csv'));
D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
B  = readtable(fullfile('..','data','boundary_349km2.csv'));

xb = (B.X_GK - min(B.X_GK))/1000;
yb = (B.Y_GK - min(B.Y_GK))/1000;
xw = D.x_km;  yw = D.y_km;

% one grid for both panels, clipped to the aquifer
ng = 320;
gx = linspace(min(xb), max(xb), ng);
gy = linspace(min(yb), max(yb), round(ng*(max(yb)-min(yb))/(max(xb)-min(xb))));
[GX, GY] = meshgrid(gx, gy);
inside = inpolygon(GX, GY, xb, yb);

panels = struct( ...
  'val',   {U.W2sq, U.W2sq_weighted}, ...
  'sel',   {Sp, Sd}, ...
  'ttl',   {'(a)  plain W_2^2 design utility', '(b)  DRASTIC-weighted utility'}, ...
  'cbl',   {'U = W_2^2', 'U_w'}, ...
  'wcm',   {6.85, 6.44}, ...
  'hcm',   {13.17, 13.19}, ...
  'file',  {'fig9a_oed_plain.png', 'fig9b_oed_drastic.png'});

for p = 1:2
    F = scatteredInterpolant(U.x_km, U.y_km, panels(p).val, 'natural', 'nearest');
    Z = F(GX, GY);
    Z(~inside) = NaN;

    fig = figure('Color','w','Units','centimeters', ...
                 'Position',[2 2 panels(p).wcm panels(p).hcm], ...
                 'PaperUnits','centimeters', ...
                 'PaperPosition',[0 0 panels(p).wcm panels(p).hcm], ...
                 'PaperSize',[panels(p).wcm panels(p).hcm]);
    ax = axes('Units','normalized','Position',[0.155 0.075 0.60 0.855]); hold on; box on

    contourf(GX, GY, Z, 18, 'LineStyle','none');
    colormap(ax, turbo);
    plot(xb, yb, 'k-', 'LineWidth', 0.7);

    hW = plot(xw, yw, '^', 'MarkerSize',3.2, 'MarkerFaceColor',[0.55 0.55 0.55], ...
              'MarkerEdgeColor',[0.15 0.15 0.15], 'LineWidth',0.3);
    hS = plot(panels(p).sel.x_km, panels(p).sel.y_km, 'p', 'MarkerSize',11, ...
              'MarkerFaceColor',[0.85 0.10 0.10], 'MarkerEdgeColor','w', 'LineWidth',0.8);
    for k = 1:height(panels(p).sel)
        text(panels(p).sel.x_km(k)+0.85, panels(p).sel.y_km(k)+0.85, sprintf('%d', k), ...
             'FontSize',8, 'FontWeight','bold', 'Color',[0.85 0.10 0.10]);
    end

    axis equal
    xlim([min(xb)-0.6 max(xb)+0.6]); ylim([min(yb)-0.6 max(yb)+0.6]);
    xlabel('Easting (km)'); ylabel('Northing (km)');
    title(panels(p).ttl, 'FontWeight','normal', 'FontSize',8.5);
    set(ax, 'FontSize', 7.5, 'Layer','top', 'TickDir','out');

    cb = colorbar('Position',[0.795 0.075 0.045 0.855]);
    cb.Label.String = panels(p).cbl;
    cb.Label.FontSize = 8;
    cb.FontSize = 7;

    lg = legend([hW hS], {'existing wells (37)','selected (1-5)'}, ...
                'Location','southoutside', 'Orientation','horizontal', 'FontSize',6.5);
    lg.Box = 'on';
    lg.Position = [0.10 0.005 0.66 0.045];

    out = fullfile('..','figures', panels(p).file);
    exportgraphics(fig, out, 'Resolution', 400);
    info = imfinfo(out);
    fprintf('Written %s : %d x %d px, aspect %.4f (frame %.4f)\n', ...
            panels(p).file, info.Width, info.Height, info.Width/info.Height, ...
            panels(p).wcm/panels(p).hcm);
end

fprintf('\nselected wells, plain:\n'); disp(Sp);
fprintf('selected wells, DRASTIC-weighted:\n'); disp(Sd);
fprintf(['Note for the caption: the utility printed for each rank is the RAW utility at that\n' ...
         'site. Selection order follows the utility re-scored after each posterior update, so\n' ...
         'the printed values need not decrease with rank.\n']);
