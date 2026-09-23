function make_map(gx, gy, inMask, vals, xb, ttl, fname)
% Single-panel journal figure: one field per figure, no duplicated content.
[GX,GY] = meshgrid(gx,gy);
Z = nan(numel(gy), numel(gx)); Z(inMask) = vals;
fig = figure('Visible','off','Units','centimeters','Position',[1 1 9 16]);
imagesc(gx, gy, Z, 'AlphaData', ~isnan(Z)); set(gca,'YDir','normal'); hold on
plot(xb(1,:), xb(2,:), 'k-', 'LineWidth', 0.7);
axis equal tight; colorbar; title(ttl, 'FontWeight','normal');
xlabel('Easting (km)'); ylabel('Northing (km)');
exportgraphics(fig, fullfile('..','figures',fname), 'Resolution', 300);
close(fig);
end
