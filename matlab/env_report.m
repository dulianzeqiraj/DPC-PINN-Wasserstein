%% Environment report - written to results/environment.txt
clear; clc;
L = {};
v = ver('MATLAB');
L{end+1} = sprintf('MATLAB: %s %s', v.Version, v.Release);
ng = 0; try, ng = gpuDeviceCount; catch, end
L{end+1} = sprintf('gpuDeviceCount: %d   (0 = no GPU, or Parallel Computing Toolbox missing)', ng);
if ng > 0
    try, g = gpuDevice; L{end+1} = sprintf('GPU: %s (%.1f GB free)', g.Name, g.AvailableMemory/1e9); catch, end
end
L{end+1} = sprintf('Deep Learning Toolbox (dlarray): %d', ~isempty(which('dlarray')));
L{end+1} = sprintf('Statistics Toolbox (TreeBagger): %d', ~isempty(which('TreeBagger')));
L{end+1} = sprintf('CPU threads: %d', maxNumCompThreads);
try, m = memory; L{end+1} = sprintf('Free RAM: %.1f GB', m.MemAvailableAllArrays/1e9); catch, end
L{end+1} = sprintf('Date: %s', char(datetime('now')));
if ~exist(fullfile('..','results'),'dir'), mkdir(fullfile('..','results')); end
fid = fopen(fullfile('..','results','environment.txt'),'w');
for k = 1:numel(L)
    fprintf('%s\n', L{k});
    fprintf(fid, '%s\n', L{k});
end
fclose(fid);
