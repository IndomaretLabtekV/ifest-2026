"""79 - Gap-free use of the other test windows: the HUC's climate OFFSET over the test period.
78: 16.7 percent of the test anomaly variance is a per-HUC constant across the four windows (valid view 10-12 percent,
max 16.5): a HUC that stays wetter/drier than its climatology through 2015-2020, plus HUC-specific errors of the test
climatology. The label compares next month's supply with 0.6 x climatology, so a HUC that sits persistently below its
climatology crosses the threshold more often. Its offset can be read from the other three windows WITHOUT knowing
their order or distance (the flaw of V16).
Features (mean over 3 other windows of the same HUC, each |offset| >= 13 months):
  oth_mean     : mean of their 36 monthly anomalies
  oth_tcal     : mean anomaly at the TARGET's calendar month (each window holds every calendar month once)
  oth_below    : share of their 36 months below the 0.6 threshold
Emulation of the test: 3 blocks of 56 months (the test spans ~4-5 years). Validation rows draw their 3 other windows
from the same block; training rows draw them from their own training segment only, never within 12 months of the
validation block (purge 12). Whole-basin holdout and the rescaled valid view as always."""
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from cvkit import HOLDOUTS, PARAMS, base_cols, t, tr, views, y, year
from common import FIG

THR = np.log(0.6)
uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
V13 = {**PARAMS, "learning_rate": 0.0273, "num_leaves": 53, "min_data_in_leaf": 91, "feature_fraction": 0.328,
       "bagging_fraction": 0.822, "lambda_l2": 9.27, "extra_trees": True}
ANOM = [f"anom{k}" for k in range(12)]
ids = pd.Index(np.unique(tr.id.values))
hi = ids.get_indexer(tr.id.values)
NT = int(t.max()) + 1
OTH = ["oth_mean", "oth_tcal", "oth_below"]


def cube(F):
    """HUC x t x 12 anomaly array of one view (NaN where the row is absent from the view)."""
    C = np.full((len(ids), NT, 12), np.nan, np.float32)
    idx = F.index.to_numpy()
    C[hi[idx], t[idx]] = F[ANOM].to_numpy(np.float32)
    return C


def other_features(C, rows, lo, hi_, rng, n_other=3):
    """for each row index in `rows`, draw n_other other windows of the same HUC at |offset| >= 13 inside [lo, hi_)."""
    tt, hh = t[rows], hi[rows]
    cand = np.concatenate([np.arange(-56, -12), np.arange(13, 57)])
    out = np.full((len(rows), 3), np.nan, np.float32)
    draws = rng.choice(cand, size=(len(rows), 24))
    for j in range(len(rows)):
        u = tt[j] + draws[j]
        u = u[(u >= lo[j]) & (u < hi_[j])]
        u = u[np.isfinite(C[hh[j], u, 0])][:n_other] if len(u) else u
        u = np.unique(u)[:n_other]
        if len(u) < n_other:
            continue
        W = C[hh[j], u]                                   # n_other x 12
        k = (u - (tt[j] + 1)) % 12                        # lag of the target's calendar month in each window
        out[j] = [np.nanmean(W), np.nanmean(W[np.arange(len(u)), k]), np.nanmean(W < THR)]
    return out


CT, CV_ = cube(TV), cube(VV)
rng = np.random.default_rng(0)
BLOCKS = [(0, 56), (56, 112), (112, NT)]
res = []
oof = {"base": np.full(len(tr), np.nan), "offset": np.full(len(tr), np.nan)}
for b, (lo, hi_b) in enumerate(BLOCKS):
    va_rows = np.flatnonzero((t >= lo) & (t < hi_b))
    fit_rows = np.flatnonzero(((t < lo - 12) | (t >= hi_b + 12)) & ~uns)
    fit_rows = fit_rows[np.isin(fit_rows, TV.index.to_numpy())]
    # training segment of each fit row (below or above the block), never touching block +- 12
    seg_lo = np.where(t[fit_rows] < lo, 0, hi_b + 12)
    seg_hi = np.where(t[fit_rows] < lo, lo - 12, NT)
    Of = other_features(CT, fit_rows, seg_lo, seg_hi, rng)
    Ov = other_features(CV_, va_rows, np.full(len(va_rows), lo), np.full(len(va_rows), hi_b), rng)
    Xf = TV.loc[fit_rows, cols].to_numpy("float32")
    Xv = VV.loc[va_rows, cols].to_numpy("float32")
    ok_f = np.isfinite(Of).all(1)
    mb = lgb.train(V13, lgb.Dataset(Xf, y[fit_rows]), 549)
    mo = lgb.train(V13, lgb.Dataset(np.hstack([Xf, Of])[ok_f], y[fit_rows][ok_f]), 549)
    oof["base"][va_rows] = mb.predict(Xv)
    ok_v = np.isfinite(Ov).all(1)
    po = mb.predict(Xv)
    po[ok_v] = mo.predict(np.hstack([Xv, Ov])[ok_v])
    oof["offset"][va_rows] = po
    print(f"  block {b}: fit {len(fit_rows)} (with offset {ok_f.mean():.2f}), valid {len(va_rows)} (with offset {ok_v.mean():.2f})", flush=True)

yrs = sorted(set(year[~np.isnan(oof['base'])]))
per = {k: np.array([ap(y[year == yy], oof[k][year == yy]) for yy in yrs]) for k in oof}
pn = {k: np.array([ap(y[(year == yy) & uns], oof[k][(year == yy) & uns]) for yy in yrs]) for k in oof}
d, dn = per["offset"] - per["base"], pn["offset"] - pn["base"]
print(f"\n[79] per-year AP (3 blocks, basin holdout): base {per['base'].mean():.4f} -> offset {per['offset'].mean():.4f} "
      f"({d.mean():+.4f}, up {(d > 0).sum()}/{len(yrs)}) | new basins {dn.mean():+.4f} (up {(dn > 0).sum()}/{len(yrs)})")
# univariate check on validation rows
Fv = pd.DataFrame(np.vstack([other_features(CV_, np.flatnonzero((t >= lo) & (t < hb)), np.full(((t >= lo) & (t < hb)).sum(), lo),
                                            np.full(((t >= lo) & (t < hb)).sum(), hb), np.random.default_rng(1))
                             for lo, hb in BLOCKS]), columns=OTH)
yy = np.concatenate([y[(t >= lo) & (t < hb)] for lo, hb in BLOCKS])
for c in OTH:
    ok = np.isfinite(Fv[c].to_numpy())
    print(f"  univariate AP of -{c}: {ap(yy[ok], -Fv[c].to_numpy()[ok]) if c != 'oth_below' else ap(yy[ok], Fv[c].to_numpy()[ok]):.3f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(np.array(yrs) - .2, per["base"], .4, label="base")
ax.bar(np.array(yrs) + .2, per["offset"], .4, label="+ HUC offset from 3 other windows")
ax.set_xlabel("validation year"); ax.set_ylabel("AP"); ax.legend()
ax.set_title("Gap-free offset features, 3-block emulation of the test period")
plt.tight_layout()
plt.savefig(FIG / "79_huc_offset_feature.png", dpi=120)
print(f"[fig] {FIG / '79_huc_offset_feature.png'}")
