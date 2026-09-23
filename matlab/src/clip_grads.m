function g = clip_grads(g, maxNorm)
% Global L2-norm gradient clipping over a nested struct of dlarrays.
s = sumsq_(g);
gn = sqrt(gather(s));
if isfinite(gn) && gn > maxNorm
    g = scale_(g, maxNorm/gn);
elseif ~isfinite(gn)
    g = scale_(g, 0);          % NaN/Inf guard: skip this step's update
end
end
function s = sumsq_(g)
s = 0;
fn = fieldnames(g);
for k = 1:numel(fn)
    v = g.(fn{k});
    if isstruct(v), s = s + sumsq_(v);
    else, s = s + sum(extractdata(v).^2,'all');
    end
end
end
function g = scale_(g, c)
fn = fieldnames(g);
for k = 1:numel(fn)
    v = g.(fn{k});
    if isstruct(v), g.(fn{k}) = scale_(v, c);
    else, g.(fn{k}) = v * c;
    end
end
end
