"""Render the actual cutting-space helper; not a native action or policy result."""
import argparse
from pathlib import Path
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Circle
import numpy as np
from terra.dig_direction import boundary_pull_details, boundary_records_from_mask, pull_stroke_details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    shape = (56, 56)
    rr, cc = np.indices(shape)
    slab = (rr >= 15) & (rr < 44) & (cc >= 20) & (cc < 45)
    trench = (rr >= 27) & (rr < 29) & (cc >= 10) & (cc < 47)
    junction = trench | ((cc >= 30) & (cc < 32) & (rr >= 13) & (rr < 29))
    angle = np.deg2rad(27)
    along = np.cos(angle)*(cc-28) + np.sin(angle)*(rr-28)
    across = -np.sin(angle)*(cc-28) + np.cos(angle)*(rr-28)
    oblique = (abs(along) < 20) & (abs(across) < 2)
    cases = [
        ('Bulk: a cross-edge stroke has room', slab, (30,33), False),
        ('Precision: the same edge needs parallel pull', slab, (30,33), True),
        ('Narrow trench: lengthwise strokes', trench, (27,25), False),
        ('Narrow trench: crosswise strokes lack room', trench, (19,30), False),
        ('Junction: the connected branch provides room', junction, (27,22), False),
        ('27° trench: actual cell-to-base directions', oblique, (23,19), False),
    ]
    cutting = jax.jit(pull_stroke_details)
    edge_test = jax.jit(boundary_pull_details)
    colors = ['#f5f5f5', '#bac3ca', '#328b71', '#d36b61']
    fig, axes = plt.subplots(2,3, figsize=(13,9), constrained_layout=True)
    tile = 40/70
    for ax, (title, mask, base, precise) in zip(axes.flat, cases):
        records, count = boundary_records_from_mask(mask)
        allowed, _ = cutting(mask, records, count, base, tile, 4., 6.5, 2.5)
        if precise:
            _, edge_allowed, _ = edge_test(mask, records,count,base,tile,.6,np.deg2rad(25))
            allowed &= edge_allowed
        radius = np.hypot(rr-base[0],cc-base[1])*tile
        candidate = mask & (radius >= 4) & (radius <= 6.5)
        view = np.zeros(shape,np.int8)
        view[mask] = 1
        view[candidate & np.asarray(allowed)] = 2
        view[candidate & ~np.asarray(allowed)] = 3
        ax.imshow(view,cmap=ListedColormap(colors),vmin=0,vmax=3,origin='lower',interpolation='nearest')
        for record in records[:count]:
            ax.plot(record[[4,6]],record[[3,5]],color='#34404b',lw=.6)
        ax.plot(base[1],base[0],'s',color='#183054',ms=6)
        for radius_m in (4,6.5):
            ax.add_patch(Circle((base[1],base[0]),radius_m/tile,fill=False,ls='--',lw=.8,color='#708397'))
        candidates=np.argwhere(candidate)
        for y,x in candidates[::max(1,len(candidates)//5)][:5]:
            delta=np.array([base[1]-x,base[0]-y],float)
            delta*=2/max(np.linalg.norm(delta),1e-6)
            ax.arrow(x,y,*delta,color='#183054',head_width=.6,length_includes_head=True,lw=.6)
        ax.set_title(title,loc='left',fontsize=10)
        ax.set_aspect('equal');ax.set_xlim(6,50);ax.set_ylim(8,48)
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('2.5 m continuous cutting room within 4–6.5 m reach\nOptional precision: 0.6 m edge band, 25° tangent tolerance',fontsize=13)
    fig.legend(handles=[Patch(color=colors[1],label='Target outside reach'),Patch(color=colors[2],label='Geometrically permitted'),Patch(color=colors[3],label='Needs another base pose')],loc='outside lower center',ncol=3,frameon=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=170)
    plt.close(fig)

if __name__ == '__main__':
    main()
