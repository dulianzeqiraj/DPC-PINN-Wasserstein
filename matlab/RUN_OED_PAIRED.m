%% Rerun the design study, head-to-head comparison paired.
%
%  main_oed_w2.m starts with `clear`, so nothing survives across the run() call: diary opened and
%  closed around it, no wrapper variables. Diary owned by MATLAB, shell output discarded; a pipe
%  that closes has hung a run here before.

t0 = tic;
diary(fullfile('..','results','oed_paired_log.txt'));
fprintf('=== main_oed_w2 with common random numbers in replay, %s ===\n', ...
        datetime('now','Format','yyyy-MM-dd HH:mm'));
run('main_oed_w2.m');
fprintf('=== done in %.1f min ===\n', toc(t0)/60);
diary off;
