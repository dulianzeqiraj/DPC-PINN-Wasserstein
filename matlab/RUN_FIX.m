error('Stale since the L5 fix of 22 Sept 2026: this driver skips retraining. Use the RUN_ALL order.');
%% The stages that failed, plus the design rerun under coupled sampling
%
%  make_fig8_rmse_bars   read the effects table by column name; R2026a's readtable did not expose
%                        it, so the columns are now taken by position
%  main_oed_w2           rerun because score_all now uses common random numbers across
%                        candidates. Without that coupling the per-candidate ranking did not
%                        reproduce between two runs of the identical configuration, Spearman 0.45
%  main_synthetic        an fprintf escape had become a real line break, so the file did not parse
%  main_closed_loop      only failed because the synthetic did
%
%  The log belongs to MATLAB, as in RUN_REST.m. Launch with the shell output sent to nul:
%      matlab -batch "RUN_FIX" > /dev/null 2>&1

function RUN_FIX()

logf = fullfile('..','results','run_fix_log.txt');
if isfile(logf), delete(logf); end
diary(logf); diary on
cleanupObj = onCleanup(@() diary('off'));

stages = {
    'make_fig8_rmse_bars', 'Figure 8, effects table read by position'
    'main_oed_w2',         'design criterion, now with common random numbers'
    'make_fig9_oed',       'Figure 9 from the new design output'
    'main_synthetic',      'synthetic recovery and the sensitivities'
    'main_closed_loop',    'closed-loop validation of the design'
};

fprintf('\n================ RUN_FIX, %d stages ================\n', size(stages,1));
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
    diary off; diary on
end

fprintf('\n================ RUN_FIX done in %.1f min ================\n', toc(t0)/60);
if isempty(failed)
    fprintf('every stage completed.\n');
else
    fprintf('STAGES THAT FAILED: %s\n', strjoin(failed, ', '));
end
disp(datetime('now'));
diary off
end


function run_isolated(name)
run([name '.m']);
end
