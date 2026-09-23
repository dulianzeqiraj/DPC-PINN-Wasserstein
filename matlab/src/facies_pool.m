function kc = facies_pool(XcPool, xw, fac, rSupport)
% Facies class for collocation points: nearest-well assignment WITHIN the
% support radius rSupport (km); beyond it the prior reverts to the neutral
% 'medium' class. Rationale (manuscript Sec. 3.3): a point-observation
% facies prior must not extrapolate over multi-km unsampled zones - with
% r = 3 km, 89% of the sampled belt keeps its nearest-well class while 87%
% of the unsampled north (N > 40 km) reverts to neutral.
key = ["coarse","medium","fine"];
[d2, nw] = min( (XcPool(1,:)'-xw(1,:)).^2 + (XcPool(2,:)'-xw(2,:)).^2, [], 2 );
kc  = 2*ones(size(XcPool,2),1);              % neutral default: medium
inS = sqrt(d2) < rSupport;
for k = 1:3
    kc(inS & fac(nw)==key(k)) = k;
end
end
