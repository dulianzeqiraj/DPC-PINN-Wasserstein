function [params, cfgOut, lastParts] = train_dpcpinn(cfg, DAT, wTrain, tag)
% Trains DPC-PINN using ONLY wells wTrain. Returns trained params AND cfgOut
% (carries Bff + per-fold scalings needed for evaluation).
% Honest CV: scalings and KC prefactors come from TRAIN wells only.
Kobs = DAT.Kobs(wTrain); hobs = DAT.hobs(wTrain); fac = DAT.fac(wTrain);

cfg.hMu = mean(hobs); cfg.hSd = std(hobs); cfg.kMu = mean(log(Kobs));

nPrior  = DAT.nPrior(wTrain);
shapeKC = log( nPrior.^3 ./ (1-nPrior).^2 );
betaF = zeros(3,1); fn = ["coarse","medium","fine"];
for k = 1:3
    msk = fac==fn(k);
    if any(msk), betaF(k) = mean( log(Kobs(msk)) - shapeKC(msk) );
    else,        betaF(k) = mean( log(Kobs) - shapeKC ); end
end

[params, cfg.Bff] = init_networks(cfg);
if cfg.w.L4 > 0                       % v2.7: L4 parameters exist only when L4 is on
    params.aux.alpha = dlarray(0.0);
    params.aux.c     = dlarray(0.0);
end
params.aux.betaF = dlarray(betaF);
if ~isfield(cfg,'prec'), cfg.prec = 'double'; end
if ~isfield(cfg,'printEvery'), cfg.printEvery = 300; end
params  = dlupdate(@(x) cast(x, cfg.prec), params);   % FP32/FP64 switch
cfg.Bff = cast(cfg.Bff, cfg.prec);
if cfg.useGPU
    params = dlupdate(@gpuArray, params); cfg.Bff = gpuArray(cfg.Bff);
end

T = DAT.Tfix;
T.xw = dlarray(DAT.xwN(:,wTrain));
T.K  = dlarray(log(Kobs)');
T.h  = dlarray(hobs');
T.hsea = cfg.hsea;
fnT = fieldnames(T);
for s_ = 1:numel(fnT)
    if isa(T.(fnT{s_}),'dlarray'), T.(fnT{s_}) = cast(T.(fnT{s_}), cfg.prec); end
end
if cfg.useGPU
    for s = ["xw","K","h"], T.(s) = gpuArray(T.(s)); end
end

avgG=[]; avgS=[]; lastParts = nan(1,9);
Lh = nan(cfg.nEpochs,1);                     % loss history (early stopping)
for ep = 1:cfg.nEpochs
    idx  = randperm(size(DAT.XnPool,2), cfg.Nc);
    B.Xc = dlarray(cast(DAT.XnPool(:,idx), cfg.prec));
    B.th = dlarray(cast(DAT.thPool(idx)',  cfg.prec));
    if cfg.useGPU, B.Xc = gpuArray(B.Xc); B.th = gpuArray(B.th); end
    B.kc = DAT.kcPool(idx)';
    B.kp = dlarray(cast(DAT.logKpPool(idx)', cfg.prec));
    B.w7 = dlarray(cast(DAT.w7Pool(idx)',    cfg.prec));
    if cfg.useGPU, B.kp = gpuArray(B.kp); B.w7 = gpuArray(B.w7); end
    [loss, grads, parts] = dlfeval(@model_loss, params, cfg, T, B, []);
    grads = clip_grads(grads, cfg.clip);
    lrNow = cfg.lr * 0.3^( (ep > 0.4*cfg.nEpochs) + (ep > 0.7*cfg.nEpochs) + (ep > 0.9*cfg.nEpochs) );
    [params, avgG, avgS] = adamupdate(params, grads, avgG, avgS, ep, lrNow);
    Lh(ep) = double(gather(extractdata(loss)));
    if isnan(Lh(ep))
        error('%s: NaN at epoch %d', tag, ep);
    end
    if mod(ep, cfg.printEvery)==0
        fprintf('%s ep %5d | L=%.3e | L2 %.2e L3 %.2e BCd %.2e BCn %.2e\n', ...
                tag, ep, Lh(ep), parts(2), parts(3), parts(8), parts(9));
    end
    lastParts = parts;
    if isfield(cfg,'patience') && ep > max(2*cfg.patience, cfg.minEp)
        recent = mean(Lh(ep-cfg.patience+1:ep));
        prev   = mean(Lh(ep-2*cfg.patience+1:ep-cfg.patience));
        if recent > prev*(1 - cfg.esTol)
            fprintf('%s early stop at epoch %d (plateau %.2f%% over %d ep)\n', tag, ep, 100*(1-recent/prev), cfg.patience);
            break
        end
    end
end
cfgOut = cfg;
end
