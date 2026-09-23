function L = oed_lib()
%OED_LIB  The Wasserstein-2 design criterion of Eqs (11) to (13), as callable pieces.
%
%  Returned as a struct of handles so that main_oed_w2.m and main_closed_loop.m share one
%  implementation instead of two copies that can drift apart.
%
%    L.cost_matrix(ZK, ZN, w)                  weighted ground cost between field samples
%    L.score_all(CK, Cn, cS, a, sig, o, ...)   expected utility and trace reduction per candidate
%    L.greedy(CK, Cn, cS, sig, o, ..., mode)   sequential placement with a posterior update
%    L.replay(sel, CK, Cn, cS, sig, o, ..., U01, Zn)  score any design, common random numbers
%    L.trace(ZK, ZN, w, p)                     trace of the field covariance under weights p
%    L.sinkhorn(Cn, A, B, eps, nIter)          entropic OT, batched over columns of B
%    L.divergence(Cn, a, B, eps, nIter)        the debiased Sinkhorn divergence
%
%  Conventions used throughout: rows of ZK and ZN are posterior samples of the standardized
%  conductivity and porosity fields on a common support of G nodes; CK(s,c) is the sample-s
%  log-conductivity at candidate c; a is the current measure on the samples; w is the diagonal
%  metric Lambda on the support, with mean one.

L.cost_matrix = @pdist2_sq;
L.score_all   = @score_all;
L.greedy      = @greedy_design;
L.replay      = @replay;
L.trace       = @weighted_trace;
L.sinkhorn    = @sinkhorn_batch;
L.divergence  = @sink_div;
end


function Csq = pdist2_sq(ZK, ZN, w)
% Weighted squared distance between joint field samples: sum_g w_g[(dzK)^2 + (dzN)^2].
    A  = [ZK .* sqrt(w), ZN .* sqrt(w)];
    n2 = sum(A.^2, 2);
    Csq = max(n2 + n2' - 2*(A*A'), 0);
end

function cost = sinkhorn_batch(Cn, A, B, eps, nIter)
% Entropic OT from marginal A to every column of B, sharing one kernel. A may be a single
% column, in which case it applies to every column of B.
    Kk = exp(-Cn/eps);
    KC = Kk .* Cn;
    v = ones(size(B)) ./ size(B,1);
    u = v;
    for it = 1:nIter
        u = A ./ (Kk*v  + 1e-300);
        v = B ./ (Kk'*u + 1e-300);
    end
    cost = sum(u .* (KC*v), 1);
end

function d = sink_div(Cn, a, B, eps, nIter)
% Sinkhorn divergence: the entropic cost minus half of each self-cost. This is the debiasing
% referred to in section 3.5.2; without it the entropic bias inflates every utility by a nearly
% constant amount, which shifts the values without changing the ranking much.
    cAB = sinkhorn_batch(Cn, a, B, eps, nIter);
    cAA = sinkhorn_batch(Cn, a, a, eps, nIter);
    cBB = sinkhorn_batch(Cn, B, B, eps, nIter);
    d = max(cAB - 0.5*cAA - 0.5*cBB, 0);
end

function tr = weighted_trace(ZK, ZN, w, p)
% Trace of the covariance of the joint field under weights p, in the Lambda metric.
    mK = p' * ZK;  mN = p' * ZN;
    vK = p' * (ZK - mK).^2;
    vN = p' * (ZN - mN).^2;
    tr = sum(w .* (vK + vN));
end

function [Uc, dTrace] = score_all(CK, Cnorm, cScale, a, sig, oed, ZK, ZN, w)
% Expected utility (11) and expected fraction of posterior trace removed, per candidate.
    nC = size(CK,2); Sn = numel(a);
    dTrace = zeros(1,nC);
    tr0 = weighted_trace(ZK, ZN, w, a);
    % Common random numbers across candidates: the parent sample and the noise draw are shared,
    % so a difference between two candidates reflects the candidates and not the sampling. Drawn
    % per call, so each greedy step still samples from its own current measure. Without this the
    % per-candidate utility is dominated by Monte Carlo error: two runs of the identical
    % configuration agreed on the ranking only to a Spearman of 0.45.
    parent = randsample_w(a, oed.nY);
    zNoise = randn(oed.nY, 1);
    B = zeros(Sn, nC*oed.nY);
    for c = 1:nC
        pk = CK(:,c);
        y  = pk(parent) + sig*zNoise;
        ll = -((y' - pk).^2) / (2*sig^2);           % Sn x nY, Eq. (12) before normalization
        ll = ll - max(ll,[],1);
        Wt = a .* exp(ll);
        Wt = Wt ./ sum(Wt,1);
        B(:, (c-1)*oed.nY + (1:oed.nY)) = Wt;
        tr = zeros(1,oed.nY);
        for j = 1:oed.nY
            tr(j) = weighted_trace(ZK, ZN, w, Wt(:,j));
        end
        dTrace(c) = 1 - mean(tr)/tr0;
    end
    cost = sink_div(Cnorm, a, B, oed.eps, oed.nIter) * cScale;
    Uc = mean(reshape(cost, oed.nY, nC), 1);
end

function sel = greedy_design(CK, Cnorm, cScale, sig, oed, ZK, ZN, w, mode)
% Sequential placement with a genuine posterior update between wells. The fantasy datum is the
% current predictive mean, so the update is deterministic and the sequence reproducible.
    Sn = size(CK,1);
    a = ones(Sn,1)/Sn;
    sel = zeros(1, oed.kSelect);
    for k = 1:oed.kSelect
        [Uc, dTr] = score_all(CK, Cnorm, cScale, a, sig, oed, ZK, ZN, w);
        obj = Uc;
        if strcmp(mode,'trace'), obj = dTr; end
        obj(sel(1:k-1)) = -inf;
        [~, best] = max(obj);
        sel(k) = best;
        a = bayes_update(a, CK(:,best), sig);
        fprintf('   %-6s well %d: candidate %3d, effective sample size after update %.1f\n', ...
                mode, k, best, 1/sum(a.^2));
    end
end

function [w2cum, trcum] = replay(sel, CK, Cnorm, cScale, sig, oed, ZK, ZN, U01, Zn)
% Replay a design through the sequential update. Randomness comes from the caller, so every
% design sees the same draw at step k. U01: nY x K uniforms, mapped through this design's own
% weights. Zn: nY x K normals for the observation noise.
% v2.7: drawn per call before, i.e. five designs scored under five different samples. Symptom at
% k=1: plain greedy 0.3029 below W2-Lambda 0.3148, which cannot happen on its own measure.
    Sn = size(CK,1);
    a  = ones(Sn,1)/Sn;
    w  = ones(1, size(ZK,2));
    tr0 = weighted_trace(ZK, ZN, w, a);
    K = numel(sel);
    w2cum = zeros(1,K); trcum = zeros(1,K); running = 0;
    for k = 1:K
        pk = CK(:,sel(k));
        parent = invcdf_w(a, U01(:,k));        % same uniforms for every design at this step
        y  = pk(parent) + sig*Zn(:,k);
        ll = -((y' - pk).^2) / (2*sig^2); ll = ll - max(ll,[],1);
        Wt = a .* exp(ll); Wt = Wt ./ sum(Wt,1);
        cost = sink_div(Cnorm, a, Wt, oed.eps, oed.nIter) * cScale;
        running = running + mean(cost);
        w2cum(k) = running;
        a = bayes_update(a, pk, sig);
        trcum(k) = 1 - weighted_trace(ZK, ZN, w, a)/tr0;
    end
end

function a = bayes_update(a, pk, sig)
% Weight update under a fantasy datum equal to the current predictive mean.
    yhat = a' * pk;
    ll = -((yhat - pk).^2) / (2*sig^2); ll = ll - max(ll);
    a = a .* exp(ll); a = a / sum(a);
end

function idx = randsample_w(p, n)
    idx = invcdf_w(p, rand(n,1));
end

function idx = invcdf_w(p, u)
% Inverse-cdf draw from weights p using supplied uniforms. Needed because a is different for
% each design by step k, so the uniforms couple and the indices cannot.
    e = cumsum(p(:)); e = e/e(end);
    idx = arrayfun(@(uu) find(e >= uu, 1, 'first'), u(:));
end
