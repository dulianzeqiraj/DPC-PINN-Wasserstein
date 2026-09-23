%% Report the fitted Kozeny-Carman prefactors  (Reviewer 1, point 7)
%
%  The printed law of Eq. (7) is not the one the code uses. The code fits, per facies, the
%  constant that stands in for the prefactor, and section 3.3 now says so. This prints the
%  fitted values so the claim can be checked against the stored model rather than believed.
%
%  Output: results/kozeny_carman_prefactors.csv

clear; clc; addpath(fullfile(pwd,'src'));
L = load(fullfile('..','results','trained_params.mat'));
D = readtable(fullfile('..','data','inversion_wells_37.csv'));

bF = L.params.aux.betaF;
if isa(bF,'dlarray'), bF = extractdata(bF); end
bF = double(gather(bF(:)));

fac = string(D.facies);
cls = ["coarse"; "medium"; "fine"];   % the order betaF is built in (main_fushekuqe.m:99);
                                      % unique() sorts alphabetically and swapped medium and fine
fprintf('betaF has %d entries; facies classes present: %s\n', numel(bF), strjoin(cls', ', '));

kc = @(nn) log(nn.^3 ./ (1-nn).^2);
init = zeros(numel(cls),1);
for i = 1:numel(cls)
    s = fac == cls(i);
    init(i) = mean(log(D.K_mday(s)) - kc(D.n_mean(s)));
end

n = min(numel(bF), numel(cls));
T = table(cls(1:n), init(1:n), bF(1:n), exp(bF(1:n) + kc(0.30)), ...
    'VariableNames', {'facies','beta_initial','beta_fitted','K_at_n_030_mday'});
writetable(T, fullfile('..','results','kozeny_carman_prefactors.csv'));
disp(T);
fprintf(['\nRead this as: the prefactor is calibrated, not taken from the printed constants.\n' ...
         'The textbook form with a 30 mm diameter and n = 0.32 gives 3e5 m/day, three orders\n' ...
         'of magnitude above the 50 to 200 m/day measured at the wells.\n']);
fprintf('Written: results/kozeny_carman_prefactors.csv\n');
