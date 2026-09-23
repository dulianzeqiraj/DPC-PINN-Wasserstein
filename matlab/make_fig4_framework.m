%% Figure 4 - the framework diagram   (v2.7)
%
%  Fourth figure with no generating script in the v1.0.0 package.
%
%  What changed against the June drawing:
%    - the L4 DRASTIC auxiliary box is gone, leaving six loss terms and the boundary penalty;
%    - the total-loss box now spells the sum out, term by term, so it matches Eq. (3) exactly
%      instead of writing a sum over i = 1..7 that no longer describes the labels;
%    - the DRASTIC layer no longer feeds the core. It enters at the design stage, as the
%      weighting of the Wasserstein-2 criterion, and its arrow now runs along the bottom to
%      the design box. Four modalities feed the network, not five.
%
%  Geometry: the drawing is sized to the frame Word already reserves for it, 15.49 x 9.47 cm,
%  so the replacement drops in without stretching and without moving anything on the page.
%  The June PNG was 3082 x 2023 px against a 1.636 frame, i.e. it was being stretched.
%
%  Output: figures/fig4_framework.png

clear; clc; close all;

W_CM = 15.49; H_CM = 9.47;                 % the frame Word reserves
fig = figure('Color','w','Units','centimeters', ...
             'Position',[2 2 W_CM H_CM], 'PaperUnits','centimeters', ...
             'PaperPosition',[0 0 W_CM H_CM], 'PaperSize',[W_CM H_CM]);
ax = axes('Position',[0 0 1 1]); axis(ax,[0 1 0 1]); axis off; hold on

CIN   = [1.00 0.96 0.85];  EIN   = [0.72 0.53 0.04];   % inputs
CCORE = [0.90 0.96 0.90];  ECORE = [0.11 0.45 0.16];   % core
CLOSS = [0.90 0.94 0.99];  ELOSS = [0.13 0.35 0.62];   % losses
CUQ   = [0.94 0.91 0.97];  EUQ   = [0.42 0.20 0.60];   % uncertainty
COED  = [0.99 0.91 0.93];  EOED  = [0.65 0.09 0.24];   % design
FS = 6.5;

%% ---- inputs that feed the network
inp = { '37 wells\newlineK, h', ...
        'Channel morphology\newlineaxes + \theta(x)', ...
        'Facies priors\newlinen_{prior}, d (3 classes)', ...
        'Sea boundary \Gamma_{sea}\newline801 vertices' };
yi = [0.80 0.62 0.44 0.26];
for k = 1:numel(inp)
    box(0.015, yi(k)-0.065, 0.175, 0.13, CIN, EIN);
    txt(0.103, yi(k), inp{k}, FS, 'normal');
    arrow(0.190, yi(k), 0.262, 0.62);
end

%% ---- the core
box(0.265, 0.30, 0.245, 0.66, CCORE, ECORE);
txt(0.388, 0.925, '\bfDPC-PINN core', 7.5, 'normal', ECORE);
sub = { 'Shared trunk 4\times64 (tanh, p = 0.1)\newlineFourier features \phi(x)\newlineheads: logK(x) linear, n(x) sigmoid', ...
        'Head network\newlineh(x) = h + s_h N_h(x)', ...
        'Adam \cdot staged LR \cdot early stop\newlineseed 20260610 \cdot FP32 (gated)' };
ys = [0.76 0.56 0.39];
hs = [0.17 0.12 0.13];
for k = 1:numel(sub)
    box(0.280, ys(k)-hs(k)/2, 0.215, hs(k), [1 1 1]*0.99, [0.55 0.75 0.55]);
    txt(0.388, ys(k), sub{k}, FS, 'normal');
end

%% ---- the six loss terms and the boundary penalty
loss = { 'L_1  Darcy residual  \nabla\cdot(Kb\nablah) + N = 0', ...
         'L_2  head misfit (37 wells)', ...
         'L_3  logK misfit (37 wells)', ...
         'L_5  anisotropic smoothness along \theta(x)', ...
         'L_6  Kozeny-Carman K-n coupling', ...
         'L_7  valley-geometry prior (W_0, L_d, \omega(x))', ...
         'L_{\partial\Omega}  sea no-flow (weak Neumann)' };
yl = linspace(0.92, 0.44, numel(loss));
for k = 1:numel(loss)
    box(0.575, yl(k)-0.036, 0.415, 0.072, CLOSS, ELOSS);
    txt(0.783, yl(k), loss{k}, FS, 'normal');
end
arrow(0.512, 0.62, 0.570, 0.68);

% the weighted sum, written out so it matches Eq. (3)
box(0.575, 0.30, 0.415, 0.085, CLOSS, ELOSS);
txt(0.783, 0.3425, ['L = w_1L_1 + w_2L_2 + w_3L_3 + w_5L_5 + w_6L_6 + w_7L_7' ...
                    ' + w_\partial L_{\partial\Omega}'], FS, 'normal');
plot([0.783 0.783], [0.404 0.389], 'k-', 'LineWidth', 0.7);
plot(0.783, 0.386, 'kv', 'MarkerSize', 3, 'MarkerFaceColor','k');

% backprop, dashed, from the sum back into the core: left, then up to the core's lower edge
plot([0.575 0.542 0.542], [0.3425 0.3425 0.345], '--', 'Color',[0.5 0.5 0.5], 'LineWidth',0.7);
plot([0.542 0.542], [0.345 0.474], '--', 'Color',[0.5 0.5 0.5], 'LineWidth',0.7);
plot([0.542 0.512], [0.474 0.474], '--', 'Color',[0.5 0.5 0.5], 'LineWidth',0.7);
plot(0.510, 0.474, '<', 'Color',[0.5 0.5 0.5], 'MarkerSize',3.5, ...
     'MarkerFaceColor',[0.5 0.5 0.5], 'MarkerEdgeColor','none');
text(0.5545, 0.400, 'backprop', 'FontSize',5.8, 'Color',[0.45 0.45 0.45], ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', 'Rotation',90);

%% ---- uncertainty
%  The box is wider than the core: at 6.5 pt the two long lines overflowed a 0.245 box and the
%  arrow to the design block cut straight through "(M = 30)".
box(0.225, 0.112, 0.325, 0.118, CUQ, EUQ);
txt(0.3875, 0.171, ['\bfUncertainty\rm\newlineMC-dropout (T = 50/30) + bootstrap (M = 30)' ...
                    '\newline\sigma^2_{tot} = \sigma^2_{boot} + \sigma^2_{MC}   (coverage-tested)'], FS, 'normal');
arrow(0.388, 0.298, 0.388, 0.234);

%% ---- design, and the DRASTIC layer that only reaches it
box(0.575, 0.085, 0.415, 0.145, COED, EOED);
txt(0.783, 0.1575, ['\bfWasserstein-2 OED\rm\newline\xi^* = argmax_\xi E_{y|\xi} ' ...
                    'W_2^2(\pi_{post}, \pi_{prior})\newlineSinkhorn \epsilon \cdot greedy 5 wells ' ...
                    '\cdot DRASTIC-weighted'], FS, 'normal');
arrow(0.552, 0.171, 0.570, 0.1575);

box(0.015, 0.020, 0.175, 0.085, CIN, EIN);
txt(0.103, 0.0625, 'DRASTIC index\newline180 points', FS, 'normal');
plot([0.190 0.360 0.360], [0.0625 0.0625 0.0625], '-', 'Color',[0.35 0.35 0.35], 'LineWidth',0.7);
plot([0.360 0.570], [0.0625 0.0625], '-', 'Color',[0.35 0.35 0.35], 'LineWidth',0.7);
plot(0.572, 0.0625, '>', 'Color',[0.35 0.35 0.35], 'MarkerSize',3.5, 'MarkerFaceColor',[0.35 0.35 0.35]);
text(0.375, 0.0855, 'design weighting only', 'FontSize',5.8, 'Color',[0.4 0.4 0.4], ...
     'HorizontalAlignment','left', 'VerticalAlignment','middle');

out = fullfile('..','figures','fig4_framework.png');
exportgraphics(fig, out, 'Resolution', 400);
fprintf('Written: %s\n', out);
info = imfinfo(out);
fprintf('   %d x %d px, aspect %.4f (frame is 1.6363)\n', info.Width, info.Height, info.Width/info.Height);

%% ---------------- local functions ----------------
function box(x, y, w, h, face, edge)
rectangle('Position',[x y w h], 'Curvature',0.20, 'FaceColor',face, ...
          'EdgeColor',edge, 'LineWidth',0.8);
end

function txt(x, y, s, fs, weight, col)
if nargin < 6, col = [0 0 0]; end
text(x, y, s, 'FontSize',fs, 'FontWeight',weight, 'Color',col, ...
     'HorizontalAlignment','center', 'VerticalAlignment','middle', 'Interpreter','tex');
end

function arrow(x1, y1, x2, y2)
plot([x1 x2], [y1 y2], '-', 'Color',[0.25 0.25 0.25], 'LineWidth',0.8);
plot(x2, y2, '>', 'Color',[0.25 0.25 0.25], 'MarkerSize',3.5, ...
     'MarkerFaceColor',[0.25 0.25 0.25], 'MarkerEdgeColor','none');
end
