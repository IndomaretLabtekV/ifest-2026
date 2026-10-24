"""68 - Value of the NEXT test origin's window (67: the test origins form one chain 169 -> 94 -> 81 -> 14, 14 months
apart, so for three origins the following origin's 12 lags cover months target+2 .. target+13).
Emulation on the train chain: for row (HUC, t) the "next window" is the row (HUC, t+14) of the same view, i.e. its
anomalies anom0..anom11 = months t+14 .. t+3. They are test INPUTS (no label, no leaderboard).
Leakage control: those features reach 14 months ahead, so the purge below a validation year grows from 12 to 26 months
(a training row's next window must not contain the validation year's target months), and above it to 26.
Judged on leave-one-year-out + whole-basin holdout (rescaled valid view), plus an exact test emulation: 4 origins at
t, t+14, t+28, t+42 with months 10, 12, 2, 4; the first three get the next-window model, the last the base model."""
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import HOLDOUTS, PARAMS, base_cols, t, tr, views, y, year

GAP = 14
uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
V13 = {**PARAMS, "learning_rate": 0.0273, "num_leaves": 53, "min_data_in_leaf": 91, "feature_fraction": 0.328,
       "bagging_fraction": 0.822, "lambda_l2": 9.27, "extra_trees": True}
NX = [f"nx_anom{k}" for k in range(12)]


def next_window(F, idx):
    """anom0..11 of the same HUC GAP months later, within the same view."""
    key = pd.MultiIndex.from_arrays([tr.id.values[idx], t[idx]])
    src = pd.DataFrame(F.loc[idx, [f"anom{k}" for k in range(12)]].to_numpy(), index=key)
    nxt = src.reindex(pd.MultiIndex.from_arrays([tr.id.values[idx], t[idx] + GAP])).to_numpy()
    return pd.DataFrame(nxt, index=idx, columns=NX)


TVn = pd.concat([TV, next_window(TV, TV.index.to_numpy())], axis=1)
VVn = pd.concat([VV, next_window(VV, VV.index.to_numpy())], axis=1)
has = np.isfinite(VVn.reindex(np.arange(len(tr)))["nx_anom11"].to_numpy())
print(f"[68] next window available for {has.mean():.3f} of train rows (missing = last {GAP} months of the chain)")

oof = {"base": np.full(len(tr), np.nan), "next window": np.full(len(tr), np.nan)}
for k in range(14):
    fit = ((t < 12 * k - 12 - GAP) | (t >= 12 * k + 12 + GAP)) & ~uns
    va = year == k
    f, v = np.flatnonzero(fit), np.flatnonzero(va)
    fh = f[np.isfinite(TVn.loc[f, "nx_anom11"].to_numpy())]
    for name, c, ff in [("base", cols, f), ("next window", cols + NX, fh)]:
        m = lgb.train(V13, lgb.Dataset(TVn.loc[ff, c].to_numpy("float32"), y[ff]), 549)
        oof[name][va] = m.predict(VVn.loc[v, c].to_numpy("float32"))
    m_ = va & has
    print(f"  year {k:2d}: rows with next window {m_.sum():6d} | base {ap(y[m_], oof['base'][m_]) if m_.any() else np.nan:.4f} "
          f"next {ap(y[m_], oof['next window'][m_]) if m_.any() else np.nan:.4f}", flush=True)
pd.DataFrame(oof).to_parquet(WORK / "68_oof.parquet")

yrs = [k for k in range(14) if (has & (year == k)).sum() > 1000]
b = np.array([ap(y[(year == k) & has], oof["base"][(year == k) & has]) for k in yrs])
n = np.array([ap(y[(year == k) & has], oof["next window"][(year == k) & has]) for k in yrs])
bn = np.array([ap(y[(year == k) & has & uns], oof["base"][(year == k) & has & uns]) for k in yrs])
nn = np.array([ap(y[(year == k) & has & uns], oof["next window"][(year == k) & has & uns]) for k in yrs])
print(f"\n[68] rows that have a next window, {len(yrs)} years: base {b.mean():.4f} -> next window {n.mean():.4f} "
      f"({(n - b).mean():+.4f}, up {(n > b).sum()}/{len(yrs)}); new basins {bn.mean():.4f} -> {nn.mean():.4f}")

# exact emulation of the test layout
month = tr.month.values
starts = [s for s in np.unique(t[month == 10]) if s + 3 * GAP <= t.max()]
emu = []
for s in starts:
    ts = [s, s + GAP, s + 2 * GAP, s + 3 * GAP]
    rows = [np.flatnonzero(t == u) for u in ts]
    yy = np.concatenate([y[r] for r in rows])
    pb = np.concatenate([oof["base"][r] for r in rows])
    pn = np.concatenate([oof["next window"][r] for r in rows[:3]] + [oof["base"][rows[3]]])
    if np.isnan(pn).any() or np.isnan(pb).any():
        continue
    emu.append(dict(start=s, prev=yy.mean(), base=ap(yy, pb), nextw=ap(yy, pn)))
E = pd.DataFrame(emu)
print(f"[68] test-layout emulation ({len(E)} chains of 4 origins 14 months apart): base {E.base.mean():.4f} -> "
      f"with next windows {E.nextw.mean():.4f} ({(E.nextw - E.base).mean():+.4f}, up {(E.nextw > E.base).sum()}/{len(E)})")
print(E.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].bar(np.array(yrs) - .2, b, .4, label="base")
ax[0].bar(np.array(yrs) + .2, n, .4, label="+ next origin's window")
ax[0].set_xlabel("validation year"); ax[0].set_ylabel("AP (rows with a next window)")
ax[0].set_title("Leave-one-year-out + basin holdout, purge 26 months")
ax[0].legend()
ax[1].scatter(E.base, E.nextw)
lim = [E[["base", "nextw"]].min().min() - .02, E[["base", "nextw"]].max().max() + .02]
ax[1].plot(lim, lim, "k--", lw=.8)
ax[1].set_xlabel("pooled AP, base"); ax[1].set_ylabel("pooled AP, next window for 3 of 4 origins")
ax[1].set_title("Exact test layout: Oct, Dec, Feb, Apr, 14 months apart")
plt.tight_layout()
plt.savefig(FIG / "68_next_window_cv.png", dpi=120)
print(f"[fig] {FIG / '68_next_window_cv.png'}")
