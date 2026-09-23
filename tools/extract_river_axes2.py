#!/usr/bin/env python3
"""River extraction v2 - clean high-res DRASTIC map (no decorations).
Water = blue/cyan mask; MAT traced as continuous skeleton longest-path."""
import sys
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from collections import deque
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The source map image is not part of the repository; give its path as the first argument.
IMG = sys.argv[1] if len(sys.argv) > 1 else 'Clean_DRASTIC_Display_20260612_164132.png'
im  = np.asarray(Image.open(IMG).convert('RGB')).astype(int)
H, W = im.shape[:2]
R, G, B = im[...,0], im[...,1], im[...,2]
mx, mn = im.max(2), im.min(2)

# ---------- body ----------
colored = (mx - mn > 30) & ~((R>245)&(G>245)&(B>245))
lab, n = ndimage.label(colored)
sizes = ndimage.sum(colored, lab, range(1, n+1))
body = ndimage.binary_fill_holes(lab == (np.argmax(sizes)+1))
contour = body & ~ndimage.binary_erosion(body)
cr, cc = np.nonzero(contour)
step = max(1, len(cr)//4000); cr, cc = cr[::step], cc[::step]

# ---------- georef (affine) ----------
Bd = pd.read_csv('data/boundary_349km2.csv')
x0, y0 = Bd.X_GK.min(), Bd.Y_GK.min()
bx = (Bd.X_GK - x0).values/1000.0; by = (Bd.Y_GK - y0).values/1000.0
tree = cKDTree(np.c_[bx, by])
u, v = cc.astype(float), (H - cr).astype(float)
def to_km(p, uu, vv):
    sx, sy, th, tx, ty = p
    c, sn = np.cos(th), np.sin(th)
    return sx*(c*uu - sn*vv) + tx, sy*(sn*uu + c*vv) + ty
sx0 = (bx.max()-bx.min())/np.ptp(u); sy0 = (by.max()-by.min())/np.ptp(v)
tx0 = bx.min()-sx0*u.min(); ty0 = by.min()-sy0*v.min()
def cost(p):
    X, Y = to_km(p, u, v)
    d1,_ = tree.query(np.c_[X,Y],1)
    d2,_ = cKDTree(np.c_[X,Y]).query(np.c_[bx[::6],by[::6]],1)
    return np.mean(d1**2)+np.mean(d2**2)
res = optimize.minimize(cost,[sx0,sy0,0,tx0,ty0],method='Nelder-Mead',
                        options=dict(maxiter=9000,xatol=1e-6,fatol=1e-8))
X1,Y1 = to_km(res.x,u,v)
rmse = np.sqrt(np.mean(tree.query(np.c_[X1,Y1],1)[0]**2))
print(f'georef: sx={res.x[0]*1000:.1f} sy={res.x[1]*1000:.1f} m/px rot={np.degrees(res.x[2]):+.2f} deg RMSE={rmse*1000:.0f} m')

# ---------- water mask ----------
body_in = ndimage.binary_erosion(body, iterations=10)
water = (B - R > 25) & (B > 90) & body_in
# km coordinates of every water pixel (for filtering by region)
wy, wx = np.nonzero(water)
WX, WY = to_km(res.x, wx.astype(float), (H - wy).astype(float))
lagoon = (WX < 5.0) & (WY > 42)
keepw = np.ones_like(WX, bool) & ~lagoon
keepw_px = keepw.copy()
water2 = np.zeros_like(water); water2[wy[keepw], wx[keepw]] = True
print(f'water px: {water.sum()} -> after lagoon filter: {water2.sum()}')

# probe: km bbox of SE-arm pixel window
pw = (wx>1800)&(wy>1500)&(wy<2600)&keepw_px
if pw.any():
    print('SE-arm probe km bbox: x[%.1f,%.1f] y[%.1f,%.1f] n=%d' % (WX[pw].min(),WX[pw].max(),WY[pw].min(),WY[pw].max(),pw.sum()))
# ---------- river groups ----------
def group_mask(cond):
    m = np.zeros_like(water2); m[wy[keepw][cond(WX[keepw],WY[keepw])], wx[keepw][cond(WX[keepw],WY[keepw])]] = True
    return m
m_mat   = group_mask(lambda x,y: y > 31)
for nm,mm in [('MAT',m_mat)]: pass
print('group px:', 'MAT', m_mat.sum())
m_ishem = group_mask(lambda x,y: (y >= 11.5) & (y <= 18) & (x > 9))
print('group px:', 'ISHEM', m_ishem.sum())
m_droja = group_mask(lambda x,y: y < 11.5)

def longest_path_axis(mask, bridge=14, eps=0.20, min_px=400):
    if mask.sum() < min_px: return None
    mb = ndimage.binary_closing(ndimage.binary_dilation(mask, iterations=bridge), iterations=bridge)
    lab2, n2 = ndimage.label(mb)
    big = np.argmax(ndimage.sum(mb, lab2, range(1, n2+1))) + 1
    sk = skeletonize(lab2 == big)
    ys, xs = np.nonzero(sk)
    idx = {(y,x):i for i,(y,x) in enumerate(zip(ys,xs))}
    nbr = [[] for _ in range(len(ys))]
    for i,(y,x) in enumerate(zip(ys,xs)):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==dx==0: continue
                j = idx.get((y+dy,x+dx))
                if j is not None: nbr[i].append(j)
    def bfs(s):
        dist = {s:0}; par = {s:None}; q = deque([s]); far = s
        while q:
            a = q.popleft()
            for b in nbr[a]:
                if b not in dist:
                    dist[b]=dist[a]+1; par[b]=a; q.append(b)
                    if dist[b]>dist[far]: far=b
        return far, par
    e1,_ = bfs(0); e2, par = bfs(e1)
    path=[]; c=e2
    while c is not None: path.append(c); c=par[c]
    P = np.c_[xs[path], ys[path]].astype(float)
    Xp, Yp = to_km(res.x, P[:,0], H-P[:,1])
    Pk = np.c_[Xp, Yp]
    def rdp(P, eps):
        if len(P)<3: return P
        e=P[-1]-P[0]; w=P[1:-1]-P[0]
        d=np.abs(e[0]*w[:,1]-e[1]*w[:,0])/ (np.linalg.norm(e)+1e-12)
        i=np.argmax(d)
        if d[i]>eps:
            L=rdp(P[:i+2],eps); Rr=rdp(P[i+1:],eps)
            return np.vstack([L[:-1],Rr])
        return np.array([P[0],P[-1]])
    return rdp(Pk, eps)

rows=[]
for name, m in [('MAT',m_mat),('ISHEM',m_ishem),('DROJA',m_droja)]:
    ax = longest_path_axis(m)
    if ax is None: print(f'{name}: too few px, skipped'); continue
    # order: start inland (max x for MAT/ISHEM entering from E; for DROJA keep as is)
    if ax[0,0] < ax[-1,0]: ax = ax[::-1]
    L = np.sum(np.hypot(np.diff(ax[:,0]),np.diff(ax[:,1])))
    print(f'{name}: {len(ax)} vertices, length {L:.1f} km, x[{ax[:,0].min():.1f},{ax[:,0].max():.1f}] y[{ax[:,1].min():.1f},{ax[:,1].max():.1f}]')
    for k,(xk,yk) in enumerate(ax,1):
        rows.append(dict(river=name,x_km=round(xk,3),y_km=round(yk,3),order=k))
pd.DataFrame(rows).to_csv('data/river_axes.csv', index=False)

# ---------- channel points (full net, downsampled) ----------
sel = np.where(keepw)[0][::3]
pd.DataFrame(dict(x_km=np.round(WX[sel],3), y_km=np.round(WY[sel],3))).to_csv('data/channel_points.csv', index=False)
print(f'channel_points.csv: {len(sel)} pts | river_axes.csv: {len(rows)} vertices')

# ---------- QA ----------
Dw = pd.read_csv('data/inversion_wells_37.csv')
fig, axs = plt.subplots(1,2,figsize=(15,10))
axs[0].imshow(im); axs[0].contour(water2, colors='magenta', linewidths=0.4)
axs[0].set_title('water mask over map'); axs[0].axis('off')
ax2=axs[1]; ax2.plot(bx,by,'k.',ms=1)
ax2.plot(WX[sel],WY[sel],'.',ms=0.6,color='tab:green',alpha=0.5,label='channel network')
ax2.scatter(Dw.x_km,Dw.y_km,c='m',s=30,zorder=6,label='wells')
for name,col in [('MAT','tab:red'),('ISHEM','tab:orange'),('DROJA','tab:blue')]:
    s=[r for r in rows if r['river']==name]
    if s: ax2.plot([r['x_km'] for r in s],[r['y_km'] for r in s],'-o',lw=2.5,color=col,label=f'{name} axis',zorder=7)
ax2.set_aspect('equal'); ax2.legend(); ax2.set_title(f'georef RMSE {rmse*1000:.0f} m')
plt.tight_layout(); plt.savefig('figures/river_axes_QA.png', dpi=120)
print('saved figures/river_axes_QA.png')
