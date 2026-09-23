%% Which results files does readtable mis-parse?
%
%  Panel (b) of Figure 8 drew empty: readtable guessed space as delimiter on
%  ablation_factorial_effects.csv ("L5 x L6" in the effect names). Seven results files have a
%  space inside a text cell; conformal_calibration.csv is one and it feeds the OED calibration.
%  Checks each against the true row count.

diary(fullfile('..','results','diag_reads_log.txt'));
files = { 'ablation_factorial_effects.csv'
          'closed_loop.csv'
          'conformal_calibration.csv'
          'conformal_scores.csv'
          'oed_sensitivity.csv'
          'synthetic_recovery.csv'
          'oed_design_comparison.csv'
          'cv_results.csv' };

for k = 1:numel(files)
    f = fullfile('..','results',files{k});
    if ~isfile(f), fprintf('%-34s MISSING\n', files{k}); continue; end

    % the truth: count the newlines, minus the header
    txt = fileread(f);
    nTrue = numel(strfind(txt, newline));
    if txt(end) ~= newline, nTrue = nTrue + 1; end
    nTrue = nTrue - 1;

    A = readtable(f);                                   % what the scripts do today
    B = readtable(f, 'Delimiter', ',', 'VariableNamingRule', 'preserve');

    ok = height(A) == nTrue;
    fprintf('%-34s rows in file %3d | readtable %3d x %d %s | comma %3d x %d\n', ...
            files{k}, nTrue, height(A), width(A), ...
            string(repmat('  <-- WRONG', 1, ~ok)), height(B), width(B));
end
diary off;
