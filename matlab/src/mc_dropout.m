function [muK, sdK, muN, sdN, muH, sdH] = mc_dropout(params, cfg, X, T)
% T stochastic passes with dropout ACTIVE (training=true) - MC-dropout UQ.
N = size(X,2);
if isfield(cfg,'prec'), X = cast(X, cfg.prec); end
if cfg.useGPU, X = gpuArray(X); end
SK = zeros(T,N); SN = zeros(T,N); SH = zeros(T,N);
for t = 1:T
    if mod(t,20)==0, fprintf('  MC-dropout pass %d/%d\n', t, T); end
    [lk, nn, hh] = forward_all(params, X, cfg, true);
    SK(t,:) = double(gather(extractdata(lk)));
    SN(t,:) = double(gather(extractdata(nn)));
    SH(t,:) = double(gather(extractdata(hh)));
end
muK = mean(SK,1); sdK = std(SK,0,1);
muN = mean(SN,1); sdN = std(SN,0,1);
muH = mean(SH,1); sdH = std(SH,0,1);
end
