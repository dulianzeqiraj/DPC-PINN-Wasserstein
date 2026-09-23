#!/usr/bin/env python3
"""Union of channel morphologies from both maps into one dataset + QA."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = pd.read_csv('data/channel_points_map1.csv'); A['src']='map1_tophat'
B = pd.read_csv('data/channel_points.csv');      B['src']='map2_clean'
M = pd.concat([A,B], ignore_index=True)
# dedup on a 50 m grid
M['gx'] = np.round(M.x_km/0.05).astype(int); M['gy'] = np.round(M.y_km/0.05).astype(int)
M = M.drop_duplicates(['gx','gy']).drop(columns=['gx','gy'])
M[['x_km','y_km','src']].to_csv('data/channel_points_merged.csv', index=False)
print(f'merged: map1={len(A)} + map2={len(B)} -> {len(M)} unique (50 m grid)')

Bd = pd.read_csv('data/boundary_349km2.csv')
x0,y0 = Bd.X_GK.min(), Bd.Y_GK.min()
bx=(Bd.X_GK-x0)/1000; by=(Bd.Y_GK-y0)/1000
D  = pd.read_csv('data/inversion_wells_37.csv')
Ax = pd.read_csv('data/river_axes.csv')
fig,ax = plt.subplots(figsize=(9,12))
ax.plot(bx,by,'k.',ms=1)
for src,col,lab in [('map1_tophat','tab:olive','streams (map 1, top-hat)'),
                    ('map2_clean','tab:green','rivers (map 2, clean)')]:
    s=M[M.src==src]; ax.plot(s.x_km,s.y_km,'.',ms=1.2,color=col,alpha=0.6,label=lab)
ax.scatter(D.x_km,D.y_km,c='m',s=32,zorder=6,label='wells')
for name,col in [('MAT','tab:red'),('ISHEM','tab:orange'),('DROJA','tab:blue')]:
    s=Ax[Ax.river==name]
    if len(s): ax.plot(s.x_km,s.y_km,'-o',lw=2.5,color=col,zorder=7,label=f'{name} axis')
ax.set_aspect('equal'); ax.legend(loc='lower left'); ax.set_xlabel('Easting km'); ax.set_ylabel('Northing km')
ax.set_title('Unified channel morphology (both maps) - %d points' % len(M))
plt.tight_layout(); plt.savefig('figures/morphology_merged_QA.png', dpi=130)
print('saved figures/morphology_merged_QA.png')
