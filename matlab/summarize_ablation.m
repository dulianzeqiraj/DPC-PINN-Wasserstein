%% Aggregate whatever ablation cells exist, and say which are missing
%
%  main_ablation_factorial.m only writes its tables after all 45 cells are on disk. Every cell is
%  seeded independently, rng(SEED0 + 1000*mode + fold), so a mode is a valid result as soon as its
%  own five folds exist, whatever else has or has not run. This reads the checkpoint folder,
%  reports every complete mode, and states plainly which modes are incomplete so that nothing
%  half-finished can reach a table by accident.
%
%  Outputs: results/ablation_factorial_wells.csv   per held-out well, complete modes only
%           results/ablation_factorial_rows.csv    per mode, complete modes only
%           results/ablation_factorial_effects.csv only if all eight subsets are complete

clear; clc;
ckDir = fullfile('..','results','ablation_fac');
assert(exist(ckDir,'dir')==7, 'no checkpoint folder at %s', ckDir);

names = {'base','+L5','+L6','+L7','+L5+L6','+L5+L7','+L6+L7','full','base+L4'};
tags  = {'base','L5','L6','L7','L5L6','L5L7','L6L7','full','baseL4'};
on5   = [0 1 0 0 1 1 0 1 0];
on6   = [0 0 1 0 1 0 1 1 0];
on7   = [0 0 0 1 0 1 1 1 0];
on4   = [0 0 0 0 0 0 0 0 1];

complete = false(1,numel(tags));
for m = 1:numel(tags)
    complete(m) = all(arrayfun(@(f) isfile(fullfile(ckDir, sprintf('%s_f%d.mat', tags{m}, f))), 1:5));
end
fprintf('cells on disk: %d of 45\n', numel(dir(fullfile(ckDir,'*.mat'))));
fprintf('COMPLETE modes  : %s\n', strjoin(names(complete), ', '));
fprintf('INCOMPLETE modes: %s\n', strjoin(names(~complete), ', '));
if ~any(complete)
    error('no mode is complete yet');
end

allRows = {};
for m = find(complete)
    for f = 1:5
        S = load(fullfile(ckDir, sprintf('%s_f%d.mat', tags{m}, f)));
        for j = 1:numel(S.cell_wells)
            allRows(end+1,:) = { string(names{m}), f, S.cell_wells(j), ...
                S.cell_logKobs(j), double(S.muK(j)), double(S.sdK(j)), ...
                S.cell_hobs(j),   double(S.muH(j)), double(S.sdH(j)), ...
                double(abs(S.cell_logKobs(j)-double(S.muK(j))) <= 1.645*double(S.sdK(j))), ...
                double(abs(S.cell_hobs(j)  -double(S.muH(j))) <= 1.645*double(S.sdH(j))) }; %#ok<AGROW>
        end
    end
end
Wt = cell2table(allRows, 'VariableNames', {'mode','fold','Well_ID', ...
     'logK_obs','logK_pred','logK_sd','h_obs','h_pred','h_sd','cov90_K','cov90_h'});
writetable(Wt, fullfile('..','results','ablation_factorial_wells.csv'));

summ = {}; rmseByMode = nan(numel(tags),1);
for m = find(complete)
    sel = Wt.mode == string(names{m});
    eK = Wt.logK_pred(sel) - Wt.logK_obs(sel);
    eH = Wt.h_pred(sel)    - Wt.h_obs(sel);
    r  = sqrt(mean(eK.^2));
    rmseByMode(m) = r;
    summ(end+1,:) = { string(names{m}), on5(m), on6(m), on7(m), on4(m), r, mean(abs(eK)), exp(r), ...
                      100*mean(Wt.cov90_K(sel)), sqrt(mean(eH.^2)), mean(abs(eH)), ...
                      100*mean(Wt.cov90_h(sel)) }; %#ok<AGROW>
end
Rt = cell2table(summ, 'VariableNames', {'mode','L5','L6','L7','L4', ...
     'logK_RMSE','logK_MAE','factor','cov90_K_pct','head_RMSE','head_MAE','cov90_h_pct'});
writetable(Rt, fullfile('..','results','ablation_factorial_rows.csv'));

fprintf('\n============ ABLATION, %d complete modes, n=37 held out each ============\n', nnz(complete));
disp(Rt);

% one-at-a-time contributions, which is what Reviewer 1 point 17 asks for and what the first four
% modes deliver on their own
if complete(1)
    fprintf('\n---- one-at-a-time contribution against the physics-and-data baseline ----\n');
    for m = 2:4
        if complete(m)
            fprintf('  %-6s %+0.4f  (%.4f from %.4f, %+0.1f%%)\n', names{m}, ...
                    rmseByMode(m)-rmseByMode(1), rmseByMode(m), rmseByMode(1), ...
                    100*(rmseByMode(m)-rmseByMode(1))/rmseByMode(1));
        end
    end
    if complete(9)
        fprintf('  %-6s %+0.4f   (the DRASTIC term removed in v2.7)\n', names{9}, ...
                rmseByMode(9)-rmseByMode(1));
    end
    if complete(8)
        fprintf('  %-6s %+0.4f   (all three priors together)\n', names{8}, ...
                rmseByMode(8)-rmseByMode(1));
    end
end

% the factorial effects need every one of the eight subsets
fac = 1:8;
if all(complete(fac))
    sgn = @(v) 2*v(fac)' - 1;
    y = rmseByMode(fac);
    s5 = sgn(on5); s6 = sgn(on6); s7 = sgn(on7);
    eff = {};
    eff(end+1,:) = {'L5 (anisotropy)',       mean(y.*s5)*2};
    eff(end+1,:) = {'L6 (Kozeny-Carman)',    mean(y.*s6)*2};
    eff(end+1,:) = {'L7 (channel geometry)', mean(y.*s7)*2};
    eff(end+1,:) = {'L5 x L6',               mean(y.*s5.*s6)*2};
    eff(end+1,:) = {'L5 x L7',               mean(y.*s5.*s7)*2};
    eff(end+1,:) = {'L6 x L7',               mean(y.*s6.*s7)*2};
    eff(end+1,:) = {'L5 x L6 x L7',          mean(y.*s5.*s6.*s7)*2};
    Et = cell2table(eff, 'VariableNames', {'effect','delta_logK_RMSE'});
    writetable(Et, fullfile('..','results','ablation_factorial_effects.csv'));
    fprintf('\n---- main effects and interactions (negative means the term helps) ----\n');
    for i = 1:height(Et)
        fprintf('  %-24s %+0.4f\n', Et.effect{i}, Et.delta_logK_RMSE(i));
    end
else
    fprintf(['\nThe eight subsets are not all complete, so no main-effects table is written.\n' ...
             'What is available is the one-at-a-time comparison above.\n']);
end

fprintf('\nWritten: results/ablation_factorial_wells.csv, results/ablation_factorial_rows.csv\n');
