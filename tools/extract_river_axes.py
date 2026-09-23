#!/usr/bin/env python3
"""Extract Mat & Droja/Ishem river axes from the DRASTIC morphology map.
Pipeline: aquifer-shape georeferencing (similarity fit to boundary_349km2.csv)
          -> blue-channel segmentation -> per-river centerline -> RDP simplify
Output: data/river_axes.csv + figures/river_axes_QA.png
"""
import sys
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The source map image is not part of the repository; give its path as the first argument.
IMG = sys.argv[1] if len(sys.argv) > 1 else 'harta_drastic_mofolo_lume.png'
im  = np.asarray(Image.open(IMG).convert('RGB')).astype(int)
H, W = im.shape[:2]
R, G, B = im[...,0], im[...,1], im[...,2]

# ---------- 1) aquifer body = colored (saturated) region, largest component ----------
mx, mn = im.max(2), im.min(2)
colored = (mx - mn > 35) & (mx > 80)              # saturated, not white/black/grey
colored[:12,:]=0; colored[-55:,:]=0; colored[:,:10]=0; colored[:,-10:]=0   # frame, scale bar, bottom label
colored = ndimage.binary_opening(colored, iterations=2)   # cut thin leader-line bridges to labels
lab, n = ndimage.label(colored)
sizes = ndimage.sum(colored, lab, range(1, n+1))
body = lab == (np.argmax(sizes) + 1)
body = ndimage.binary_closing(body, iterations=2)
body = ndimage.binary_fill_holes(body)
contour = body & ~ndimage.binary_erosion(body)
cr, cc = np.nonzero(contour)                       # row, col of image contour

# ---------- 2) true boundary in km ----------
Bd = pd.read_csv('data/boundary_349km2.csv')
x0, y0 = Bd.X_GK.min(), Bd.Y_GK.min()
bx = (Bd.X_GK - x0).values/1000.0
by = (Bd.Y_GK - y0).values/1000.0
tree = cKDTree(np.c_[bx, by])

# ---------- 3) similarity fit  km = s*Rot(th)·[col, H-row] + (tx,ty) ----------
u, v = cc.astype(float), (H - cr).astype(float)    # image coords, y-up
def to_km(p, uu, vv):
    sx, sy, th, tx, ty = p
    c, sn = np.cos(th), np.sin(th)
    xr = c*uu - sn*vv; yr = sn*uu + c*vv
    return sx*xr + tx, sy*yr + ty
# init from bounding boxes (map may have unequal x/y scales)
sx0 = (bx.max()-bx.min()) / max(np.ptp(u), 1)
sy0 = (by.max()-by.min()) / max(np.ptp(v), 1)
tx0 = bx.min() - sx0*u.min(); ty0 = by.min() - sy0*v.min()
treeC_full = None
def cost(p):
    X, Y = to_km(p, u, v)
    d1, _ = tree.query(np.c_[X, Y], k=1)            # contour -> boundary
    tC = cKDTree(np.c_[X, Y])
    d2, _ = tC.query(np.c_[bx[::4], by[::4]], k=1)  # boundary -> contour (anti-collapse)
    return np.mean(d1**2) + np.mean(d2**2)
res = optimize.minimize(cost, [sx0, sy0, 0.0, tx0, ty0], method='Nelder-Mead',
                        options=dict(maxiter=8000, xatol=1e-6, fatol=1e-8))
sx, sy, th, tx, ty = res.x
X1, Y1 = to_km(res.x, u, v)
rmse_fit = np.sqrt(np.mean(tree.query(np.c_[X1, Y1], 1)[0]**2))
print(f'georef AFFINE: sx={sx*1000:.1f} sy={sy*1000:.1f} m/px rot={np.degrees(th):+.2f} deg  contour-RMSE={rmse_fit*1000:.0f} m')

# ---------- 4) blue river channels ----------
blue = (B > R + 15) & (B > G + 8) & (B > 60) & body
blue = ndimage.binary_opening(blue, iterations=1)
lb, nb = ndimage.label(blue)
rivers = {'MAT': [], 'DROJA': []}
for i in range(1, nb+1):
    ys, xs = np.nonzero(lb == i)
    if len(xs) < 10: continue
    X, Y = to_km(res.x, xs.astype(float), (H - ys).astype(float))
    cxk, cyk = X.mean(), Y.mean()
    elong = max(np.ptp(X), np.ptp(Y))
    if   cyk > 36 and cxk < 4.5:                    continue        # NW coastal lagoons
    elif cyk > 32 and cxk > 4.5 and elong > 0.8:    rivers['MAT'].append((X, Y))
    elif cyk < 13 and elong > 0.6:                  rivers['DROJA'].append((X, Y))
    # west-coast lagoons (cxk<5, cyk>38) and tiny bits -> dropped

# ---------- 4b) full channel network: thin dark lines inside the plain ----------
body_in = ndimage.binary_erosion(body, iterations=3)      # keep away from the black outline
lum = im.mean(2)
bth = ndimage.grey_closing(lum, size=(7,7)) - lum          # black top-hat: thin dark on any background
dark = (bth > 22) & body_in
ld, nd = ndimage.label(dark)
keep = np.zeros_like(dark)
for i in range(1, nd+1):
    m = ld == i
    if m.sum() < 12: continue
    # thickness filter: solid blobs (lagoon hatching) have deep interior
    if ndimage.distance_transform_edt(m).max() > 2.2: continue
    # straightness filter: cartographic section lines are near-perfect lines > ~5 km
    ys_, xs_ = np.nonzero(m)
    P = np.c_[xs_, ys_].astype(float); P -= P.mean(0)
    w_, V_ = np.linalg.eigh(P.T @ P)
    resid = np.sqrt(w_[0]/len(P)); span = np.sqrt(w_[1]/len(P))*2
    if resid < 1.2 and span*60 > 4500: continue            # straight & long -> map line, drop
    keep |= m
chan = keep | (blue & body_in)
ys, xs = np.nonzero(chan)
Xc2, Yc2 = to_km(res.x, xs.astype(float), (H - ys).astype(float))
inside = (Xc2 > -1) & (Xc2 < 25) & (Yc2 > -1) & (Yc2 < 55) & ~((Xc2 < 4.5) & (Yc2 > 36))
pd.DataFrame(dict(x_km=np.round(Xc2[inside],3), y_km=np.round(Yc2[inside],3))).to_csv(
    'data/channel_points_map1.csv', index=False)
print(f'channel network: {inside.sum()} px -> data/channel_points.csv')

# ---------- 5) centerline: PCA-binned medians + RDP ----------
def centerline(pts_list, nbins=22):
    X = np.concatenate([p[0] for p in pts_list]); Y = np.concatenate([p[1] for p in pts_list])
    P = np.c_[X, Y]; mu = P.mean(0); A = P - mu
    w, V = np.linalg.eigh(A.T @ A); ax = V[:, -1]
    t = A @ ax
    edges = np.linspace(t.min(), t.max(), nbins+1)
    path = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t <= b)
        if m.sum() >= 3: path.append([np.median(X[m]), np.median(Y[m])])
    path = np.array(path)
    return path[np.argsort(path @ ax)]             # ordered along major axis
def rdp(P, eps=0.30):
    if len(P) < 3: return P
    e = P[-1]-P[0]; w = P[1:-1]-P[0]
    d = np.abs(e[0]*w[:,1]-e[1]*w[:,0])/np.linalg.norm(e)
    i = np.argmax(d)
    if d[i] > eps:
        L = rdp(P[:i+2], eps); Rr = rdp(P[i+1:], eps)
        return np.vstack([L[:-1], Rr])
    return np.array([P[0], P[-1]])

rows = []
for name, comps in rivers.items():
    if not comps: print(f'WARNING: no blue pixels classified as {name}'); continue
    npx = sum(len(c[0]) for c in comps)
    path = rdp(centerline(comps), 0.30)
    print(f'{name}: {len(comps)} components, {npx} px -> {len(path)} vertices, '
          f'span x[{path[:,0].min():.1f},{path[:,0].max():.1f}] y[{path[:,1].min():.1f},{path[:,1].max():.1f}] km')
    for k, (xk, yk) in enumerate(path, 1):
        rows.append(dict(river=name, x_km=round(xk,3), y_km=round(yk,3), order=k))
pd.DataFrame(rows).to_csv('data/river_axes_map1.csv', index=False)
print('saved data/river_axes.csv')

# ---------- 6) QA figure ----------
Dw = pd.read_csv('data/inversion_wells_37.csv')
fig, axs = plt.subplots(1, 2, figsize=(13, 9))
axs[0].imshow(im); axs[0].contour(blue, colors='cyan', linewidths=0.8)
axs[0].contour(body, colors='k', linewidths=0.5); axs[0].set_title('image: body + blue mask'); axs[0].axis('off')
ax = axs[1]; ax.plot(bx, by, 'k.', ms=1)
ax.scatter(Dw.x_km, Dw.y_km, c='m', s=28, zorder=5, label='wells')
Xc, Yc = to_km(res.x, u, v); ax.plot(Xc, Yc, '.', ms=1, color='0.6', label='image contour (mapped)')
ax.plot(Xc2[inside], Yc2[inside], '.', ms=0.8, color='tab:green', alpha=0.5, label='channel network')
cols = {'MAT':'tab:red', 'DROJA':'tab:blue'}
for name in rivers:
    sub = [r for r in rows if r['river']==name]
    if sub: ax.plot([r['x_km'] for r in sub], [r['y_km'] for r in sub], '-o',
                    lw=2.5, color=cols[name], label=f'{name} axis')
ax.set_aspect('equal'); ax.legend(); ax.set_xlabel('Easting km'); ax.set_ylabel('Northing km')
ax.set_title(f'georef RMSE {rmse_fit*1000:.0f} m | axes over boundary+wells')
plt.tight_layout(); plt.savefig('figures/river_axes_QA_map1.png', dpi=140)
print('saved figures/river_axes_QA.png')
