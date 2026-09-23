#!/usr/bin/env python
"""Figure 1, right panel: the aquifer, drawn from data/.

Reviewer 3, detail 1: caption promises red/orange/green symbols for the 37 wells by gravel
facies; the submitted figure has no wells at all, only Albania and a river network.

Redrawn here: 3,411-vertex boundary polygon, 37 wells by facies, extracted channel network,
three named river axes. Left locator panel is the author's cartography, carried over unchanged
and cropped at the frame between the two panels (col 420 of 808; see SPLIT below).

Gauss-Kruger offsets recovered from the data, not assumed: every row of boundary_349km2.csv and
inversion_wells_37.csv carries both coordinate systems, difference constant to the mm. Asserted
for all 37 wells before anything is plotted.

Out: figures/fig1_site.png, 1850 px = 303 dpi at the 15.5 cm placement width.
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
FIGS = os.path.join(HERE, '..', 'figures')
LOCATOR = os.path.join(HERE, 'fig1_locator_from_submitted.png')

W_TOTAL = 1850            # 303 dpi at the 15.5 cm placement width
PLACE_CM = 15.5           # the width the figure is placed at in the document
# The submitted image is 808 px wide. Column 382 is the locator's own right border and the north
# arrow sits at 396 to 418, so the locator crop has to reach past 418; the right panel's rotated
# y-axis labels begin at 432, so it has to stop before that. Its outward tick marks reach 425, so the cut is at 420.
SPLIT = 420
SRC_W = 808

FACIES = {'coarse': ('#d62728', 'Coarse gravel'),
          'medium': ('#ff7f0e', 'Medium gravel'),
          'fine':   ('#2ca02c', 'Fine gravel')}


def rows(name):
    with open(os.path.join(DATA, name), newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    bnd = rows('boundary_349km2.csv')
    wells = rows('inversion_wells_37.csv')
    chan = rows('channel_points_merged.csv')
    axes = rows('river_axes.csv')

    # the projected frame, recovered from the data
    x0 = float(bnd[0]['X_GK']) - float(bnd[0]['x_km']) * 1000.0
    y0 = float(bnd[0]['Y_GK']) - float(bnd[0]['y_km']) * 1000.0
    for r in wells:
        assert abs((float(r['X_GK']) - float(r['x_km']) * 1000.0) - x0) < 0.01
        assert abs((float(r['Y_GK']) - float(r['y_km']) * 1000.0) - y0) < 0.01
    print('Gauss-Kruger origin of the aquifer frame: %.2f, %.2f' % (x0, y0))

    bx = [float(r['x_km']) for r in bnd]
    by = [float(r['y_km']) for r in bnd]

    # geometry of the new panel, matched to the locator crop so the two sit at the same height
    loc = Image.open(LOCATOR)
    h_px = round(loc.size[1] * (W_TOTAL * SPLIT / SRC_W) / loc.size[0])
    w_px = W_TOTAL - round(W_TOTAL * SPLIT / SRC_W)
    print('locator %dx%d -> %dx%d, new panel %dx%d'
          % (loc.size + (round(W_TOTAL * SPLIT / SRC_W), h_px, w_px, h_px)))

    # Draw at the printed size: 1850 px at 15.5 cm = 303 dpi, so figure dpi = 303 makes a point
    # here a point on the page. First attempt drew at 200 dpi over 11.3 cm -> printed 6.6 cm,
    # legend at ~3 pt.
    dpi = W_TOTAL / (PLACE_CM / 2.54)
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    print('panel drawn at %.2f cm, printed at %.2f cm, %.0f dpi'
          % (w_px / dpi * 2.54, w_px / dpi * 2.54, dpi))
    ax = fig.add_axes([0.235, 0.165, 0.735, 0.815])

    ax.plot([float(r['x_km']) for r in chan], [float(r['y_km']) for r in chan],
            marker='.', markersize=0.9, linestyle='none', color='#9ecae1',
            zorder=1, rasterized=False)
    ax.fill(bx, by, facecolor='#fbfbf8', edgecolor='none', zorder=0)
    ax.plot(bx, by, '-', color='black', linewidth=0.8, zorder=3)

    for name, lab in (('MAT', 'River Mat'), ('DROJA', 'River Droja'), ('ISHEM', 'River Ishëm')):
        seg = sorted((r for r in axes if r['river'] == name), key=lambda r: int(r['order']))
        ax.plot([float(r['x_km']) for r in seg], [float(r['y_km']) for r in seg],
                '-', color='#2171b5', linewidth=1.5, zorder=4)
        mid = seg[len(seg) // 2]
        ax.annotate(lab, (float(mid['x_km']), float(mid['y_km'])), fontsize=7.0,
                    style='italic', color='#08306b', zorder=6,
                    xytext=(3, 4), textcoords='offset points')

    for key, (col, lab) in FACIES.items():
        sel = [r for r in wells if r['facies'] == key]
        ax.plot([float(r['x_km']) for r in sel], [float(r['y_km']) for r in sel],
                'o', markersize=4.2, markerfacecolor=col, markeredgecolor='black',
                markeredgewidth=0.4, linestyle='none', zorder=5)
        print('%-7s %2d wells' % (key, len(sel)))

    ax.set_xlim(-1.4, 22.1)
    ax.set_ylim(-1.6, 53.8)
    ax.set_aspect('equal')

    # ticks labelled in the projected system, as the submitted panel had them
    xt = [4384000, 4392000, 4400000]
    yt = [4580000, 4592000, 4604000, 4616000, 4628000]
    ax.set_xticks([(v - x0) / 1000.0 for v in xt])
    ax.set_xticklabels([str(v) for v in xt], fontsize=7.0)
    ax.set_yticks([(v - y0) / 1000.0 for v in yt])
    ax.set_yticklabels([str(v) for v in yt], fontsize=7.0, rotation=90, va='center')
    ax.tick_params(length=2.6, width=0.6, pad=2.2, right=True, top=True,
                   labelright=False, labeltop=False, direction='in')
    for s in ax.spines.values():
        s.set_linewidth(0.7)

    # Scale bar bottom left, where the aquifer is narrow and east of centre. Ticks point down;
    # first attempt had them colliding with the numerals.
    sx, sy = 0.4, 2.6
    ax.plot([sx, sx + 8], [sy, sy], '-', color='black', linewidth=1.5, zorder=7,
            solid_capstyle='butt')
    for d in (0, 4, 8):
        ax.plot([sx + d, sx + d], [sy, sy - 0.7], '-', color='black', linewidth=0.7, zorder=7)
        ax.annotate('%d' % d, (sx + d, sy + 0.5), fontsize=6.8, ha='center', zorder=7)
    ax.annotate('km', (sx + 8.4, sy), fontsize=6.8, va='center', zorder=7)

    # north arrow
    ax.annotate('', xy=(20.4, 51.6), xytext=(20.4, 48.4), zorder=7,
                arrowprops=dict(arrowstyle='-|>', color='black', linewidth=0.8,
                                mutation_scale=9))
    ax.annotate('N', (20.4, 52.0), fontsize=8.5, ha='center', fontweight='bold', zorder=7)

    handles = [Line2D([], [], marker='o', linestyle='none', markersize=4.2, markerfacecolor=c,
                      markeredgecolor='black', markeredgewidth=0.4, label=l)
               for c, l in FACIES.values()]
    handles += [Line2D([], [], color='#2171b5', linewidth=1.5, label='River axis'),
                Line2D([], [], marker='.', linestyle='none', markersize=4,
                       color='#9ecae1', label='Channel network'),
                Line2D([], [], color='black', linewidth=0.8, label='Aquifer boundary')]
    # Below the map, not on it. At 7 pt the six entries are wider than any gap inside the frame;
    # centre left they covered four wells and two river labels. Two columns under the axes cost
    # 7% of the map height and hide nothing.
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.045), ncol=2,
              fontsize=7.0, frameon=False, borderpad=0.3, labelspacing=0.42,
              columnspacing=1.4, handletextpad=0.5, handlelength=1.3)

    tmp = os.path.join(FIGS, '_fig1_right.png')
    fig.savefig(tmp, dpi=dpi, facecolor='white')
    plt.close(fig)

    right = Image.open(tmp).convert('RGB')
    left = loc.convert('RGB').resize((W_TOTAL - right.size[0], h_px), Image.LANCZOS)
    out = Image.new('RGB', (W_TOTAL, h_px), 'white')
    out.paste(left, (0, 0))
    out.paste(right, (left.size[0], 0))
    p = os.path.join(FIGS, 'fig1_site.png')
    out.save(p, dpi=(303, 303))
    os.remove(tmp)
    print('Written: %s  %dx%d px, %.0f dpi at 15.5 cm'
          % (p, out.size[0], out.size[1], out.size[0] / (15.5 / 2.54)))


if __name__ == '__main__':
    main()
