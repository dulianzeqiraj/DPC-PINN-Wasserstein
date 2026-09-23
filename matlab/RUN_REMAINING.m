error('Stale since the L5 fix of 22 Sept 2026: this driver skips retraining. Use the RUN_ALL order.');
%% Run only what the revision still needs, most important first
%
%  RUN_ALL.m rebuilds the whole chain from scratch. This does not: the canonical inversion, the
%  cross-validation, the bootstrap and the conformal calibration are already on disk and their
%  numbers are in the manuscript, so re-running them would only risk moving numbers that are
%  already correct and cited.
%
%  The order is by what each stage unblocks, so that stopping early still leaves the paper in a
%  better state than it started:
%
%    1  main_oed_w2        the design criterion the paper defines, which the released code did
%                          not compute. This is the objection that decided the rejection
%                          (referee 1 points 11 to 13, referee 3 general comment 3). No training.
%    2  main_baselines     kriging of heads and porosity from kriged conductivity
%                          (referee 4 points 4 and 5). Minutes.
%    3  make_fig9_oed      Figure 9 from the new design output
%    4  main_synthetic     recovery of known fields, plus the weight and architecture
%                          sensitivities (referee 1 point 19, referee 4 points 3 and 7)
%    5  main_closed_loop   does drilling where the criterion says actually help
%                          (referee 3 general comment 3)
%    6  main_ablation_factorial   completes the 2^3 factorial from 27 cells to 45, which adds the
%                          interactions and the DRASTIC evidence row. The one-at-a-time
%                          separation already in the paper does not depend on this.
%    7  the summaries and the remaining figures
%
%  Every training stage checkpoints per unit of work and skips what is already on disk, so this
%  can be interrupted and relaunched. Run it from matlab/ with the working directory set there.
%
%  After it finishes, regenerate the graphical abstract, which reads the new design points:
%      python ..\tools\make_graphical_abstract.py

function RUN_REMAINING()

stages = {
    'main_oed_w2',              'design criterion, Eqs (11) to (13)'
    'main_baselines',           'baselines including heads and kriged porosity'
    'make_fig9_oed',            'Figure 9'
    'main_synthetic',           'synthetic recovery and the sensitivities'
    'main_closed_loop',         'closed-loop validation of the design'
    'main_ablation_factorial',  'complete the factorial'
    'summarize_ablation',       'ablation tables from whatever cells exist'
    'make_fig7_reliability',    'Figure 7'
    'make_fig8_rmse_bars',      'Figure 8'
    'make_figS1_variogram',     'Figure S1'
    'make_figS3_boundary',      'Figure S3, the boundary conditions'
    'make_fig4_framework',      'Figure 4'
    'report_betaF',             'the fitted Kozeny-Carman prefactors'
    'summarize_all',            'consolidated summary'
    'env_report',               'record the MATLAB version these runs were made under'
};

fprintf('\n================ RUN_REMAINING, %d stages ================\n', size(stages,1));
disp(datetime('now'));
t0 = tic;
failed = {};

for k = 1:size(stages,1)
    name = stages{k,1};
    fprintf('\n---------------- %d/%d  %s\n', k, size(stages,1), name);
    fprintf('                 %s\n', stages{k,2});
    disp(datetime('now'));
    tk = tic;
    try
        % run_isolated gives the stage its own workspace. Every stage script begins with
        % "clear", which would otherwise wipe this loop's own variables and stop the driver
        % after the first stage.
        run_isolated(name);
        fprintf('[ok]   %s in %.1f min\n', name, toc(tk)/60);
    catch err
        failed{end+1} = name; %#ok<AGROW>
        fprintf('[FAIL] %s after %.1f min\n', name, toc(tk)/60);
        disp(getReport(err, 'extended', 'hyperlinks', 'off'));
        fprintf('continuing with the next stage\n');
    end
    close all force
end

fprintf('\n================ done in %.1f min ================\n', toc(t0)/60);
if isempty(failed)
    fprintf('every stage completed.\n');
else
    fprintf('STAGES THAT FAILED: %s\n', strjoin(failed, ', '));
    fprintf('nothing downstream of a failed stage should be trusted until it is fixed.\n');
end
fprintf('\nNow regenerate the graphical abstract, which reads the new design points:\n');
fprintf('    python ..\\tools\\make_graphical_abstract.py\n');
disp(datetime('now'));
end


function run_isolated(name)
% A stage runs here so that its "clear" empties this workspace and not the driver's.
run([name '.m']);
end
