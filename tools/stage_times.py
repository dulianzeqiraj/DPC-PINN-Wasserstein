#!/usr/bin/env python
"""Wall-clock record of the v2.8.0 run, written to results/stage_times.csv.

    python tools/stage_times.py

Per-unit times come from the modification times of the checkpoints each stage writes (one file per
fold, member, cell, configuration or design), taken only between consecutive units of the same
stage; the first unit of a stage is dropped because it also carries the stage's set-up. Epoch
counts come from the early-stopping line each training prints in its log. Checkpoint times are a
property of the machine that ran the stages: a copy of the repository (git, a zip, a download)
does not preserve them, which is why the result is stored as a CSV rather than recomputed.

Some stages shared the GPU with up to two others, to finish the chain in one day. Their per-unit
times are longer than a unit run alone, and the CSV says which ones ran alone.
"""
import csv
import glob
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
T0 = os.path.getmtime(os.path.join(RES, '_L5_START'))


def units(pattern):
    fs = [f for f in glob.glob(os.path.join(RES, pattern)) if os.path.getmtime(f) > T0]
    return sorted(os.path.getmtime(f) for f in fs)


def gaps(ts, lo=None, hi=None):
    g = [(b - a) / 60 for a, b in zip(ts, ts[1:])]
    if lo is not None:
        g = [x for x, a in zip(g, ts) if lo <= a < hi]
    return g


def epochs(logs, tag):
    out = []
    for lg in logs:
        p = os.path.join(RES, lg)
        if os.path.exists(p):
            txt = open(p, encoding='utf-8', errors='replace').read().replace('\r', '\n')
            out += [int(m) for m in re.findall(r'%s\S* early stop at epoch (\d+)' % tag, txt)]
    return out


rows = []
cv = units('cv_ckpt/fold_*.mat')
g = gaps(cv)
e = epochs(['L5_02_cv.log'], 'F')
rows.append(['5-fold cross-validation', 'GPU', 'alone', len(cv), '%.1f' % st.mean(g),
             '%.2f' % (sum(g) * 60 / sum(e[1:])) if len(e) == len(cv) else '', ''])
bt = units('boot/member_*.mat')
# members 1 and 2 ran before any other stage was started
g = gaps(bt[:2]) if len(bt) >= 2 else []
e = epochs(['L5_04_bootstrap.log'], 'B0')[:2]
rows.append(['bootstrap member', 'GPU', 'alone', 2, '%.1f' % st.mean(g) if g else '',
             '%.2f' % (g[0] * 60 / e[1]) if g and len(e) == 2 else '', ''])
g = gaps(bt)[2:]
rows.append(['bootstrap member', 'GPU', 'shared', len(bt), '%.1f' % st.median(g), '', ''])
fc = units('ablation_fac/*.mat')
rows.append(['factorial ablation cell', 'GPU', 'shared', len(fc), '%.1f' % st.median(gaps(fc)), '', ''])
sy = units('synth/*.mat')
rows.append(['synthetic configuration', 'GPU', 'shared', len(sy), '%.1f' % st.median(gaps(sy)), '', ''])
cl = units('closed_loop/*.mat')
rows.append(['closed-loop refit', 'GPU', 'shared', len(cl), '%.1f' % st.median(gaps(cl)), '', ''])

# stage wall-clock: each stage has its own log, created when MATLAB starts and last written when it
# stops (on Windows getctime is the creation time). Resumed stages have one log per attempt.
stage = {}
for lg in sorted(glob.glob(os.path.join(RES, 'L5_*.log'))):
    stage[os.path.basename(lg)[3:-4]] = (os.path.getctime(lg), os.path.getmtime(lg))
for name, (t0_, t1_) in sorted(stage.items(), key=lambda kv: kv[1][0]):
    rows.append(['stage ' + name, '', '', '', '%.1f' % ((t1_ - t0_) / 60), '', ''])
# Machine time: the union of the stage intervals, so hours in which no stage ran (the chain was
# interrupted overnight once and resumed from its checkpoints) are not counted.
busy, cur = 0.0, None
for a_, b_ in sorted(stage.values()):
    if cur is None or a_ > cur[1]:
        if cur:
            busy += cur[1] - cur[0]
        cur = [a_, b_]
    else:
        cur[1] = max(cur[1], b_)
busy += cur[1] - cur[0]
rows.append(['whole chain, machine time', 'mixed', 'shared', '', '', '', '%.1f' % (busy / 3600)])

out = os.path.join(RES, 'stage_times.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['unit', 'device', 'gpu', 'n_units', 'minutes_per_unit', 's_per_epoch', 'hours_elapsed'])
    w.writerows(rows)
for r in rows:
    print(r)
print('written', out)
