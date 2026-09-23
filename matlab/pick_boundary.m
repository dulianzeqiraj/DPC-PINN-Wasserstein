%% pick_boundary - plots the de-duplicated boundary with vertex-index labels
%  so the coastal Dirichlet and inflow arcs can be fixed by EXACT index ranges.
clear; clc; close all;
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
Bd = readtable(fullfile('..','data','boundary_349km2.csv'));
xb = [ (Bd.X_GK - min(Bd.X_GK))/1000, (Bd.Y_GK - min(Bd.Y_GK))/1000 ]';
dseg = hypot(diff(xb(1,:)), diff(xb(2,:))); xb = xb(:, [true, dseg>1e-4]);
N = size(xb,2);
fig = figure('Units','centimeters','Position',[1 1 22 30]); hold on
plot(xb(1,:), xb(2,:), '-', 'Color',[.7 .7 .7]);
scatter(xb(1,:), xb(2,:), 6, 1:N, 'filled'); colormap(turbo); cb=colorbar;
cb.Label.String = 'vertex index (1..N)';
step = 100;
for i = 1:step:N
    text(xb(1,i), xb(2,i), sprintf('%d',i), 'FontSize',8, 'FontWeight','bold');
end
plot(xb(1,1), xb(2,1), 'kp', 'MarkerSize',14, 'MarkerFaceColor','y');  % start vertex
axis equal; grid on
title(sprintf('Boundary with vertex indices (N=%d). Star = vertex 1. Report: coast = [A..B], inflow arcs = [C..D], ...', N));
xlabel('Easting (km)'); ylabel('Northing (km)');
exportgraphics(fig, fullfile('..','figures','boundary_indices.png'), 'Resolution', 220);
fprintf('Saved figures/boundary_indices.png  (N = %d vertices)\n', N);
