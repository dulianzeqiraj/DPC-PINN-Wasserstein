error('Stale since the L5 fix of 22 Sept 2026: this driver skips retraining. Use the RUN_ALL order.');
%% Everything still outstanding, with a log MATLAB owns
%
%  The first attempt piped MATLAB's output through a shell wrapper. When that wrapper exited,
%  MATLAB blocked on its next write and sat at zero CPU for five hours with the work unfinished.
%  This version opens its own diary, so the log belongs to MATLAB and no shell can break it.
%  Launch it with the shell output sent to nul:
%
%      matlab -batch "RUN_REST" > /dev/null 2>&1
%
%  The factorial is complete and its tables are on disk, so it is not repeated here.

function RUN_REST()

logf = fullfile('..','results','run_rest_log.txt');
if isfile(logf), delete(logf); end
diary(logf); diary on
cleanupObj = onCleanup(@() diary('off'));

stages = {
    'main_oed_w2',            'design criterion on the calibrated posterior'
    'make_fig9_oed',          'Figure 9'
    'summarize_ablation',     'ablation tables from all 45 cells'
    'make_fig7_reliability',  'Figure 7'
    'make_fig8_rmse_bars',    'Figure 8'
    'make_figS1_variogram',   'Figure S1'
    'make_figS3_boundary',    'Figure S3, the boundary conditions'
    'make_fig4_framework',    'Figure 4'
    'report_betaF',           'the fitted Kozeny-Carman prefactors'
    'main_synthetic',         'synthetic recovery and the sensitivities'
    'main_closed_loop',       'closed-loop validation of the design'
    'summarize_all',          'consolidated summary'
    'env_report',             'record the MATLAB version'
};

fprintf('\n================ RUN_REST, %d stages ================\n', size(stages,1));
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
        run_isolated(name);
        fprintf('[ok]   %s in %.1f min\n', name, toc(tk)/60);
    catch err
        failed{end+1} = name; %#ok<AGROW>
        fprintf('[FAIL] %s after %.1f min\n', name, toc(tk)/60);
        disp(getReport(err, 'extended', 'hyperlinks', 'off'));
    end
    close all force
    diary off; diary on      % flush, so the file is readable from outside while this runs
end

fprintf('\n================ RUN_REST done in %.1f min ================\n', toc(t0)/60);
if isempty(failed)
    fprintf('every stage completed.\n');
else
    fprintf('STAGES THAT FAILED: %s\n', strjoin(failed, ', '));
end
disp(datetime('now'));
diary off
end


function run_isolated(name)
% A stage runs here so that its "clear" empties this workspace and not the driver's.
run([name '.m']);
end
