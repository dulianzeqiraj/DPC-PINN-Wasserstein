function DAT = fold_priors(DAT, XcPool, xw, axesXY, chanXY, wTrain, cfg)
% FOLD_PRIORS  Rebuild the collocation priors that read well data, from the training wells only.
%   DAT = fold_priors(DAT, XcPool, xw, axesXY, chanXY, wTrain, cfg)
% Two priors look at the wells. The facies class of a collocation point (L6) is the class of its
% nearest well, and the geo-prior (L7) is calibrated on well logK and relaxed near well positions.
% Built once from all 37 wells, a held-out well would shape the prior that is then scored at that
% same well. Up to v2.7 the validation scripts did exactly that; from v2.8 they call this per fold
% or per bootstrap member. The orientation field used by L5 comes from the channel network alone
% and is not touched. wTrain may carry repeats (bootstrap): they weight the calibration as they
% weight the data misfit in train_dpcpinn, and change nothing in the nearest-well quantities.
DAT.kcPool = facies_pool(XcPool, xw(:,wTrain), DAT.fac(wTrain), cfg.rFacies);
P7 = geo_prior('calib', axesXY, chanXY, xw(:,wTrain), log(DAT.Kobs(wTrain)), cfg);
G7 = geo_prior('eval', P7, XcPool, xw(:,wTrain), 'noOrientation');
DAT.logKpPool = G7.logKp;  DAT.w7Pool = G7.w7;
fprintf('  geo-prior calib on %d training wells: K_far=%.0f K_chan=%.0f m/day  R2=%.2f\n', ...
        numel(wTrain), exp(P7.a), exp(P7.a+P7.b), P7.R2);
end
