"""65 - Spatial smoothing of the predictions. What is missing is next month's rain (27: regional weather oracle
0.690 -> 0.918), and rain is shared by a whole basin. Every HUC's own model sees its own state plus a few basin
aggregates of the CURRENT anomaly; it does not see what the model concluded for its basin mates. Theory: pulling each
logit towards the mean logit of its basin (same origin) averages out per-HUC noise in the estimate of the regional
outlook. Risk: basins mix wet and dry sub-basins, so smoothing can erase real within-basin contrasts.
Post-hoc, two weights (basin, downstream), chosen NESTED: on years 0-6 judged on 7-13 and vice versa.
Input: the leave-one-year-out + basin-holdout OOF of 63 (seed 42)."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import WORK, FIG
from cvkit import BASIN, HOLDOUTS, t, test_like, tr, y, year

uns = HOLDOUTS["basin"]
O = pd.read_parquet(WORK / "63_oof_s42.parquet")
to = tr.to_id.astype(str).str.strip().str.lower().values
key = pd.DataFrame({"id": tr.id.values, "to": to, "t": t, "basin": tr.id.map(BASIN).values})


def neighbours(lg):
    d = key.assign(lg=lg)
    g = d.groupby(["basin", "t"]).lg
    s, n = g.transform("sum").to_numpy(), g.transform("size").to_numpy()
    basin = np.where(n > 1, (s - lg) / np.maximum(n - 1, 1), lg)  # leave-self-out basin mean
    own = d.set_index(["id", "t"]).lg
    down = own.reindex(pd.MultiIndex.from_arrays([d.to, d.t])).to_numpy()
    down = np.where(np.isfinite(down), down, lg)
    return basin, down


yearly = lambda p, yrs: np.array([ap(y[year == k], p[year == k]) for k in yrs])
W = np.round(np.arange(-0.4, 0.61, 0.1), 2)  # negative = sharpen away from the basin mean
for member in ["lgb V13 tuned", "lgb V8"]:
    p = O[member].to_numpy()
    lg = np.log(p / (1 - p))
    basin, down = neighbours(lg)
    smooth = lambda wb, wd: lg + wb * (basin - lg) + wd * (down - lg)
    grid = {(wb, wd): yearly(smooth(wb, wd), range(14)) for wb, wd in itertools.product(W, W) if wb + wd <= 0.8}
    b = grid[(0.0, 0.0)]
    print(f"\n[65] {member}: base mean AP {b.mean():.4f}")
    top = sorted(grid, key=lambda k: -grid[k].mean())[:5]
    for k in top:
        print(f"  basin w {k[0]:.1f} down w {k[1]:.1f}: {grid[k].mean():.4f} ({(grid[k] - b).mean():+.4f}, up {(grid[k] > b).sum()}/14)  [in-sample choice]")
    for sel, held in [(range(0, 7), range(7, 14)), (range(7, 14), range(0, 7))]:
        best = max(grid, key=lambda k: grid[k][list(sel)].mean())
        d = grid[best][list(held)] - b[list(held)]
        print(f"  NESTED select years {sel.start}-{sel.stop - 1} -> w={best}; judged on {held.start}-{held.stop - 1}: "
              f"{d.mean():+.4f} (up {(d > 0).sum()}/7)")
    for wb in (-0.2, -0.1, 0.1, 0.2):
        ps = smooth(wb, 0.0)
        tl, tlb = test_like(ps, n=500), test_like(lg, n=500)
        n_new = np.array([ap(y[(year == k) & uns], ps[(year == k) & uns]) - ap(y[(year == k) & uns], lg[(year == k) & uns]) for k in range(14)])
        print(f"  fixed basin w {wb}: test-like {(tl - tlb).mean():+.4f} ({(tl > tlb).mean():.0%} up), new basins {n_new.mean():+.4f}")

    if member == "lgb V13 tuned":
        G = pd.Series({k: (v - b).mean() * 1000 for k, v in grid.items()}).unstack()
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        im = ax[0].imshow(G.to_numpy(), cmap="RdBu", vmin=-5, vmax=5, origin="lower")
        ax[0].set_xticks(range(len(G.columns)), G.columns); ax[0].set_yticks(range(len(G.index)), G.index)
        ax[0].set_xlabel("downstream weight"); ax[0].set_ylabel("basin weight")
        ax[0].set_title("AP gain x1000 (all 14 years, in-sample grid)")
        plt.colorbar(im, ax=ax[0])
        dry = y == 1
        ax[1].hist(lg[dry] - basin[dry], bins=80, alpha=.6, density=True, label="positives")
        ax[1].hist(lg[~dry] - basin[~dry], bins=80, alpha=.6, density=True, label="negatives")
        ax[1].set_xlim(-6, 6)
        ax[1].set_title("own logit minus basin-mates mean logit")
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(FIG / "65_spatial_smoothing.png", dpi=120)
        print(f"[fig] {FIG / '65_spatial_smoothing.png'}")
