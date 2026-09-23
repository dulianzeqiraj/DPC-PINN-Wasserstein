function [logK, n, h] = forward_all(params, X, cfg, training)
% X: 2xN dlarray (unformatted). Dropout active when training==true (also for MC-dropout).
X = stripdims(X);   % defensive: formats are never needed in this architecture
Z = cfg.Bff * X;  F = [cos(Z); sin(Z)];

A = F;
for i = 1:cfg.nLayers                          % trunk: tanh layers
    A = tanh( params.trunk.("W"+i) * A + params.trunk.("b"+i) );
    if training, A = A .* ( (rand(size(A)) > cfg.dropout) / (1-cfg.dropout) ); end
end
logK = cfg.kMu + ( params.headK.W * A + params.headK.b );   % centred in log-space
n    = cfg.nMin + (cfg.nMax-cfg.nMin) ./ (1 + exp(-(params.headN.W * A + params.headN.b)));

Ah = F;
for i = 1:cfg.nLayers                          % h-net hidden layers
    Ah = tanh( params.hnet.("W"+i) * Ah + params.hnet.("b"+i) );
    if training, Ah = Ah .* ( (rand(size(Ah)) > cfg.dropout) / (1-cfg.dropout) ); end
end
h = cfg.hMu + cfg.hSd * ( params.hnet.("W"+(cfg.nLayers+1)) * Ah + params.hnet.("b"+(cfg.nLayers+1)) );  % scaled to data range
end
