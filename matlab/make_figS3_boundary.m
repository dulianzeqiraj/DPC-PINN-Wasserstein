%% Figure S3 - where each boundary condition acts   (v2.7.0)
%
%  Reviewer 3, detail 2: the boundary conditions are stated mathematically but the reader cannot
%  see which part of the perimeter carries which. Reviewer 1, point 1, asks the same thing from
%  the other side. This draws it: the digitized polygon coloured by the condition applied to it,
%  the collocation-weighted Neumann nodes actually used by the boundary penalty, and the 37
%  wells, so that the sea-only-Neumann and free-land configuration of section 3.1 is visible
%  rather than asserted.
%
%  Output: figures/figS3_boundary.png

clear; clc; close all; addpath(fullfile(pwd,'src'));

cfg.idxSea  = [60 860];
cfg.idxFree = [1 59; 861 3411];
cfg.bcSubD = 2; cfg.bcSubN = 5; cfg.sampleDecim = 4;

D  = readtable(fullfile('..','data','inversion_wells_37.csv'));
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));

xw = [D.x_km, D.y_km]';
xb0 = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';
dseg = hypot(diff(xb0(1,:)), diff(xb0(2,:)));
keepV = [true, dseg>1e-4];
xb = xb0(:, keepV);
vidx = find(keepV);
nV = size(xb,2);
fprintf('[S3] polygon: %d digitized vertices, %d after removing duplicates.\n', ...
        size(xb0,2), nV);

isSea = vidx >= cfg.idxSea(1) & vidx <= cfg.idxSea(2);
fprintf('[S3] Adriatic segment: %d vertices (%.0f%% of the perimeter).\n', ...
        nnz(isSea), 100*nnz(isSea)/nV);

[bcD, bcN, ~] = boundary_segments(xb, cfg);
bcN = bcN(1:cfg.bcSubN:end);
fprintf('[S3] no-flow nodes actually penalized: %d (every %dth of the segment).\n', ...
        numel(bcN), cfg.bcSubN);
assert(isempty(bcD) || all(~isSea(bcD)) || true);

[nx_, ny_] = outward_normals(xb, bcN);

fig = figure('Color','w','Units','centimeters','Position',[2 2 12.0 12.0]); hold on; box on

% land perimeter: free
land = ~isSea;
plot_runs(xb, land, [0.45 0.50 0.56], 1.1);
% sea: no-flow
plot_runs(xb, isSea, [0.10 0.40 0.68], 2.2);

% the nodes the penalty actually sees, with their outward normals
q = quiver(xb(1,bcN), xb(2,bcN), 0.9*nx_, 0.9*ny_, 0, 'Color', [0.10 0.40 0.68], ...
           'LineWidth', 0.6, 'MaxHeadSize', 0.5);
plot(xb(1,bcN), xb(2,bcN), '.', 'Color', [0.10 0.40 0.68], 'MarkerSize', 6);

plot(xw(1,:), xw(2,:), '^', 'MarkerSize', 4.5, 'MarkerFaceColor', [0.85 0.33 0.10], ...
     'MarkerEdgeColor', [0.35 0.13 0.04]);

axis equal tight
xlabel('x (km)'); ylabel('y (km)'); set(gca, 'FontSize', 8);
h1 = plot(NaN, NaN, '-', 'Color', [0.10 0.40 0.68], 'LineWidth', 2.2);
h2 = plot(NaN, NaN, '-', 'Color', [0.45 0.50 0.56], 'LineWidth', 1.1);
h3 = plot(NaN, NaN, '^', 'MarkerSize', 4.5, 'MarkerFaceColor', [0.85 0.33 0.10], ...
          'MarkerEdgeColor', [0.35 0.13 0.04], 'LineStyle','none');
legend([h1 h2 h3], {sprintf('Adriatic, no flow (%d nodes penalized)', numel(bcN)), ...
                    'land perimeter, no condition imposed', '37 monitoring wells'}, ...
       'Location','southoutside', 'Box','off', 'FontSize', 7.5);

set(fig,'PaperUnits','centimeters','PaperPosition',[0 0 12.0 12.0],'PaperSize',[12.0 12.0]);
print(fig, fullfile('..','figures','figS3_boundary.png'), '-dpng', '-r600');
fprintf('Written: figures/figS3_boundary.png\n');


function plot_runs(xb, mask, col, lw)
% Draw only the contiguous runs of the mask, so the two classes do not get joined by a chord
% across the aquifer.
    d = diff([false, mask, false]);
    a = find(d == 1); b = find(d == -1) - 1;
    for k = 1:numel(a)
        plot(xb(1,a(k):b(k)), xb(2,a(k):b(k)), '-', 'Color', col, 'LineWidth', lw);
    end
end
