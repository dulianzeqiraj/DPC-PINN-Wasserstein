function [params, Bff] = init_networks(cfg)
% Trainable params as SCALAR structs with dynamic fields (W1,b1,...,Wk,bk) - % the documented pattern for dlgradient/adamupdate. Fourier matrix Bff is FIXED
% (returned separately; never updated).
rngState = rng; rng(7,'twister');
Bff = dlarray(cfg.sigmaF * randn(cfg.mFourier, 2));
din = 2*cfg.mFourier;

params.trunk = mlp_init([din, repmat(cfg.hidden,1,cfg.nLayers)]);          % cfg.nLayers tanh layers
params.headK = struct('W', dlarray(sqrt(1/cfg.hidden)*randn(1,cfg.hidden)), 'b', dlarray(0));
params.headN = struct('W', dlarray(sqrt(1/cfg.hidden)*randn(1,cfg.hidden)), 'b', dlarray(0));
params.hnet  = mlp_init([din, repmat(cfg.hidden,1,cfg.nLayers), 1]);       % last layer linear
rng(rngState);
end

function S = mlp_init(sz)
S = struct();
for i = 1:numel(sz)-1
    S.("W"+i) = dlarray( sqrt(2/sz(i)) * randn(sz(i+1), sz(i)) );
    S.("b"+i) = dlarray( zeros(sz(i+1),1) );
end
S.n = numel(sz)-1;   % layer count (numeric; adamupdate ignores non-dlarray? -> keep OUT)
S = rmfield(S,'n');  % safer: no non-dlarray fields inside learnables
end
