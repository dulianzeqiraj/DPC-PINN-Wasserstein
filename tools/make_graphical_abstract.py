#!/usr/bin/env python
"""Redraw the graphical abstract from the data, so it cannot go stale unnoticed again.

The June version carried three claims that are no longer true: a seven-term loss, a conformal
coverage of 86 and 89 per cent, and the DRASTIC index as an input to the inversion. The loss now
has six terms, the calibrated coverage is 91.9 per cent under both protocols, and DRASTIC survives
only in the design weighting and in one kriging baseline.

The aquifer outline, the 37 wells and the selected design points are read from the repository, so
rerunning this after a new design run updates the picture rather than leaving it behind.

Usage: python tools/make_graphical_abstract.py [output.png]
"""
import csv
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, 'data')
RESULTS = os.path.join(REPO, 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(REPO), 'graphical_abstract.png')

W, H = 2400, 1000
MARGIN, GAP, TOP, BH = 44, 66, 96, 720
BW = (W - 2 * MARGIN - 3 * GAP) // 4
FOOT = TOP + BH - 66

GREEN = (26, 122, 76)
BLUE = (23, 92, 168)
ORANGE = (214, 122, 24)
RED = (186, 46, 40)
INK = (34, 38, 44)
GREY = (110, 116, 124)
FILL = {GREEN: (238, 247, 242), BLUE: (238, 244, 252),
        ORANGE: (253, 245, 234), RED: (253, 240, 239)}

FONTS = 'C:/Windows/Fonts/'


def font(name, size):
    for cand in (name, 'segoeui.ttf', 'arial.ttf'):
        p = FONTS + cand
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = font('segoeuib.ttf', 44)
F_BODY = font('segoeui.ttf', 31)
F_SMALL = font('segoeui.ttf', 26)
F_IT = font('segoeuii.ttf', 27)
F_LBL = font('seguisb.ttf', 30)
F_FOOT = font('segoeui.ttf', 30)


def read_boundary():
    p = os.path.join(DATA, 'boundary_349km2.csv')
    xs, ys = [], []
    with open(p, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            xs.append(float(r['X_GK']))
            ys.append(float(r['Y_GK']))
    x0, y0 = min(xs), min(ys)
    return [((x - x0) / 1000.0, (y - y0) / 1000.0) for x, y in zip(xs, ys)]


def read_wells():
    p = os.path.join(DATA, 'inversion_wells_37.csv')
    out = []
    with open(p, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            out.append((float(r['x_km']), float(r['y_km'])))
    return out


def read_design():
    p = os.path.join(RESULTS, 'oed_selected_plain.csv')
    if not os.path.exists(p):
        return []
    out = []
    with open(p, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            out.append((float(r['x_km']), float(r['y_km'])))
    return out


def read_coverage():
    """Calibrated coverage as conformal_calibration.m wrote it (5-fold CV, logK, leave-one-out)."""
    p = os.path.join(RESULTS, 'conformal_calibration.csv')
    with open(p, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['set'].startswith('5-fold') and r['quantity'] == 'logK':
                return float(r['cov_loo_pct'])
    raise SystemExit('no 5-fold logK row in %s' % p)


def fit(pts, box, pad=16):
    """Map km coordinates into a pixel box, preserving aspect, y up."""
    bx, by, bw, bh = box
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sx = (bw - 2 * pad) / (max(xs) - min(xs))
    sy = (bh - 2 * pad) / (max(ys) - min(ys))
    s = min(sx, sy)
    ox = bx + pad + ((bw - 2 * pad) - s * (max(xs) - min(xs))) / 2
    oy = by + pad + ((bh - 2 * pad) - s * (max(ys) - min(ys))) / 2

    def m(p):
        return (ox + s * (p[0] - min(xs)), oy + (bh - 2 * pad) - s * (p[1] - min(ys))
                - ((bh - 2 * pad) - s * (max(ys) - min(ys))) / 2 + pad - pad)
    return m


def rounded(d, box, col, r=26, wd=4):
    x, y, w, h = box
    d.rounded_rectangle([x, y, x + w, y + h], radius=r, outline=col, width=wd,
                        fill=FILL.get(col, (255, 255, 255)))


def centre(d, text, f, cx, y, col=INK):
    w = d.textlength(text, font=f)
    d.text((cx - w / 2, y), text, font=f, fill=col)


def arrow(d, x, y, ln=44):
    d.line([x, y, x + ln, y], fill=(70, 76, 84), width=8)
    d.polygon([(x + ln + 22, y), (x + ln - 4, y - 15), (x + ln - 4, y + 15)], fill=(70, 76, 84))


def star(d, cx, cy, r, col):
    import math
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=col, outline=(120, 20, 18))


def main():
    bnd = read_boundary()
    wells = read_wells()
    design = read_design()

    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    xs = [MARGIN + i * (BW + GAP) for i in range(4)]
    cols = [GREEN, BLUE, ORANGE, RED]
    titles = ['Multi-modal data', 'DPC-PINN', 'Calibrated joint maps', 'Wasserstein-2 design']
    for i in range(4):
        rounded(d, (xs[i], TOP, BW, BH), cols[i])
        centre(d, titles[i], F_TITLE, xs[i] + BW / 2, TOP + 34, cols[i])
    for i in range(3):
        arrow(d, xs[i] + BW + 8, TOP + BH / 2)

    # ---- box 1: the six information sources, DRASTIC no longer among them
    lines = ['Darcy   div(T grad h) = 0',
             '37 heads + pumping-test K',
             'paleo-channel fabric  \u03b8(x)',
             'channel morphology',
             'from legacy maps',
             'Kozeny-Carman   K \u2194 n']
    y = TOP + 150
    for k, t in enumerate(lines):
        f = F_SMALL if t.startswith('from legacy') else F_BODY
        centre(d, t, f, xs[0] + BW / 2, y, GREY if f is F_SMALL else INK)
        y += 62 if f is F_BODY else 46
    centre(d, 'six sources, one differentiable loss', F_IT, xs[0] + BW / 2, FOOT, GREY)

    # ---- box 2: the network
    cx = xs[1] + BW / 2
    layers = [(cx - 170, 3, (150, 190, 224)), (cx - 20, 4, (46, 116, 182)),
              (cx + 150, 3, (18, 62, 116))]
    coords = []
    for lx, n, col in layers:
        ys_ = [TOP + 300 + (j - (n - 1) / 2) * 78 for j in range(n)]
        coords.append([(lx, yy) for yy in ys_])
    for a, b in zip(coords, coords[1:]):
        for p in a:
            for q in b:
                d.line([p, q], fill=(206, 214, 222), width=2)
    for (lx, n, col), pts in zip(layers, coords):
        for p in pts:
            d.ellipse([p[0] - 21, p[1] - 21, p[0] + 21, p[1] + 21], fill=col)
    for lab, p in zip(['log K', 'n', 'h'], coords[2]):
        d.text((p[0] + 38, p[1] - 20), lab, font=F_LBL, fill=(18, 62, 116))
    centre(d, 'shared trunk  \u00b7  six-term loss', F_BODY, cx, TOP + BH - 190)
    centre(d, 'joint physics-informed inversion', F_IT, cx, FOOT, GREY)

    # ---- box 3: the two fields
    half = (BW - 40) / 2
    for k, (col, lab) in enumerate([((228, 146, 54), 'K(x)'), ((104, 158, 214), 'n(x)')]):
        bx = xs[2] + 20 + k * (half + 4)
        m = fit(bnd, (bx, TOP + 196, half, 352))
        d.polygon([m(p) for p in bnd], fill=col, outline=(90, 90, 90))
        centre(d, lab, F_LBL, bx + half / 2, TOP + 566, INK)
    cov_lab = 'conformal UQ   %.1f%%' % read_coverage()
    bw_ = d.textlength(cov_lab, font=F_BODY) + 40
    d.rounded_rectangle([xs[2] + BW / 2 - bw_ / 2, TOP + 128,
                         xs[2] + BW / 2 + bw_ / 2, TOP + 128 + 54],
                        radius=12, outline=ORANGE, width=3, fill=(255, 255, 255))
    centre(d, cov_lab, F_BODY, xs[2] + BW / 2, TOP + 138)
    centre(d, 'coverage tested, not assumed', F_IT, xs[2] + BW / 2, FOOT, GREY)

    # ---- box 4: the design
    m = fit(bnd, (xs[3] + 20, TOP + 132, BW - 40, 400))
    d.polygon([m(p) for p in bnd], fill=(240, 242, 240), outline=(90, 90, 90))
    for w_ in wells:
        p = m(w_)
        d.polygon([(p[0], p[1] - 8), (p[0] - 7, p[1] + 6), (p[0] + 7, p[1] + 6)],
                  fill=(90, 96, 104))
    for s in design:
        p = m(s)
        star(d, p[0], p[1], 17, (206, 58, 50))
    centre(d, 'argmax  E  W\u2082\u00b2(posterior, prior)', F_BODY, xs[3] + BW / 2, TOP + 556)
    centre(d, 'DRASTIC-weighted variant reported', F_SMALL, xs[3] + BW / 2, TOP + 606, GREY)
    lab = 'next wells to drill' if design else 'design points pending a run'
    centre(d, lab, F_IT, xs[3] + BW / 2, FOOT, GREY)

    centre(d, 'Fush\u00eb-Kuqe alluvial aquifer   \u00b7   349 km\u00b2   \u00b7   '
              '37 monitoring wells   \u00b7   NW Albania',
           F_FOOT, W / 2, TOP + BH + 66, (70, 76, 84))

    img.save(OUT, dpi=(300, 300))
    print('wrote %s  (%d x %d, %d wells, %d design points)'
          % (OUT, W, H, len(wells), len(design)))


if __name__ == '__main__':
    main()
