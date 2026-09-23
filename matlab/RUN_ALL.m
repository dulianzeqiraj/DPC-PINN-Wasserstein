%% ============================================================
%%  DPC-PINN-Wasserstein  -  RUN_ALL (master driver, v2.8.0)
%%  Runs everything from scratch, in order, in a single folder tree.
%%
%%  Every stage that trains writes per-unit checkpoints and skips work already on disk, so the
%%  driver can be stopped and relaunched. Delete the matching folder under results/ to force a
%%  redo. Run it from matlab/ with the working directory set there: run() with an absolute path
%%  changes MATLAB's working directory and every later stage then fails to find its data.
%%
%%  Times measured on the v2.8.0 run (GTX 1050 GPU; results/stage_times.csv). 'shared' means
%%  up to three stages ran on the GPU at once, which lengthens each unit:
%%    1 full inversion             12 min
%%    2 cross-validation           35 min, 7.0 min per fold
%%    3 baselines                  <2 min
%%    4 bootstrap ensemble         6.5 min per member alone, 8.5 shared, 30 members
%%    5 conformal calibration      instant, reads stage 2 and 4
%%    6 factorial ablation         9.4 min per cell shared, 45 cells
%%    7 synthetic recovery         112 min shared, 15 configurations
%%    8 design study               53 min shared (with stage 5), no training
%%   8b closed-loop validation     94 min shared, five refits
%%    9 figures and summary         a few minutes
%% ============================================================
disp('================================================================');
disp(' STAGE 0/9: Environment report (MATLAB, GPU, toolboxes)');
disp(datetime('now'));
run('env_report.m');

disp('================================================================');
disp(' STAGE 1/9: Full inversion, 37 wells, writes results/trained_params.mat');
disp(datetime('now'));
run('main_fushekuqe.m');
run('report_betaF.m');

disp('================================================================');
disp(' STAGE 2/9: 5-fold cross-validation');
disp(datetime('now'));
run('main_cv.m');

disp('================================================================');
disp(' STAGE 3/9: Baselines (ordinary kriging, regression kriging, random forest)');
disp(datetime('now'));
run('main_baselines.m');

disp('================================================================');
disp(' STAGE 4/9: Bootstrap ensemble with out-of-bag coverage');
disp(datetime('now'));
run('uq_bootstrap.m');

disp('================================================================');
disp(' STAGE 5/9: Split-conformal calibration, both protocols');
disp(datetime('now'));
run('conformal_calibration.m');

disp('================================================================');
disp(' STAGE 6/9: Full 2^3 factorial ablation over L5, L6, L7');
disp(datetime('now'));
run('main_ablation_factorial.m');
run('summarize_ablation.m');

disp('================================================================');
disp(' STAGE 7/9: Synthetic recovery of known fields');
disp(datetime('now'));
run('main_synthetic.m');

disp('================================================================');
disp(' STAGE 8/9: Wasserstein-2 design study and the design comparison');
disp(datetime('now'));
run('main_oed_w2.m');

disp('================================================================');
disp(' STAGE 8b/9: Closed-loop validation of the design on the synthetic aquifer');
disp(datetime('now'));
run('main_closed_loop.m');

disp('================================================================');
disp(' STAGE 9/9: Figures and consolidated summary');
disp(datetime('now'));
run('make_fig4_framework.m');
run('make_fig7_reliability.m');
run('make_fig8_rmse_bars.m');
run('make_fig9_oed.m');
run('make_fig11_synthetic.m');
run('make_figS1_variogram.m');
run('make_figS3_boundary.m');
run('summarize_all.m');

disp('================================================================');
disp(' RUN_ALL FINISHED.');
disp(datetime('now'));
