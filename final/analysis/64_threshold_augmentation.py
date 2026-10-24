"""64 - Threshold augmentation. The label is one cut of a continuous quantity, label = z <= log 0.6 with
z = log(ws(t+1) / clim(t+1)) (03). Neighbouring cuts (0.45, 0.8) obey the same physics: the same recession, the same
next-month rain. Theory: training on several cuts at once, with every threshold-relative feature re-expressed as a
margin to ITS cut, multiplies the examples near a decision boundary (the onset / recovery zones where the AP is lost,
18) and should smooth the ranking at the 0.6 cut. Risk: the other cuts have different base rates and seasonality, so
they can dilute the signal. Judged on leave-one-year-out + whole-basin holdout, scored only at the real 0.6 cut."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score as ap

from common import FIG
from cvkit import HOLDOUTS, PARAMS, REPAIRS, base_cols, t, test_like, tr, views, y, year

uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
V13 = {**PARAMS, "learning_rate": 0.0273, "num_leaves": 53, "min_data_in_leaf": 91, "feature_fraction": 0.328,
       "bagging_fraction": 0.822, "lambda_l2": 9.27, "extra_trees": True}
MARGINS = ["margin_anom_persist", "margin_raw_persist", "margin_lastyear", "margin_dry"]

# continuous target (as in 61): ws(t+1) = repaired lag0 of the same HUC one origin later
rep = REPAIRS["v7"](tr)
nxt = pd.Series(rep.q_ws_cum_m3_lag0.abs().values, index=pd.MultiIndex.from_arrays([tr.id.values, tr.t.values]))
ws_next = nxt.reindex(pd.MultiIndex.from_arrays([tr.id.values, tr.t.values + 1])).to_numpy()
FT = TV.reindex(np.arange(len(tr)))
clim_next = rep.q_ws_cum_m3_lag0.abs().values / np.exp(FT.margin_raw_persist.values + np.log(0.6))
z = np.log((ws_next + 1) / (clim_next + 1))
ok = np.isfinite(z) & ~uns
agree = ((z[ok] <= np.log(0.6)) == y[ok]).mean()
print(f"[64] z available for {np.isfinite(z[~uns]).mean():.3f} of train-view rows; (z <= log 0.6) == label on {agree:.4f}")
for c in (0.45, 0.6, 0.8):
    print(f"     base rate at cut {c}: {(z[ok] <= np.log(c)).mean():.3f}")


def shifted(X, c):
    X = X.copy()
    X[MARGINS] = X[MARGINS] + np.log(0.6) - np.log(c)
    return X


def fit_aug(f, cuts, w_other, with_thr):
    parts, ys, ws = [], [], []
    for c in cuts:
        X = TV.loc[f, cols]
        if c == 0.6:
            yy, keep = y[f], np.ones(len(f), bool)
        else:
            keep = np.isfinite(z[f])
            yy = (z[f] <= np.log(c)).astype(int)
            X = shifted(X, c)
        if with_thr:
            X = X.assign(thr=np.log(c))
        parts.append(X[keep]); ys.append(yy[keep]); ws.append(np.full(keep.sum(), 1.0 if c == 0.6 else w_other))
    Xf = pd.concat(parts).to_numpy("float32")
    return lgb.train(V13, lgb.Dataset(Xf, np.concatenate(ys), weight=np.concatenate(ws)), 549)


VARIANTS = {"base (0.6 only)": ((0.6,), 1.0, False),
            "aug 0.45/0.6/0.8": ((0.45, 0.6, 0.8), 1.0, False),
            "aug 0.45/0.6/0.8 + thr feature": ((0.45, 0.6, 0.8), 1.0, True),
            "aug 0.5/0.6/0.7 w0.5 + thr": ((0.5, 0.6, 0.7), 0.5, True)}
oof = {n: np.full(len(tr), np.nan) for n in VARIANTS}
for k in range(14):
    fit, va = (t < 12 * k - 12) | (t >= 12 * k + 24), year == k
    f, v = np.flatnonzero(fit & ~uns), np.flatnonzero(va)
    for n, (cuts, w, thr) in VARIANTS.items():
        Xv = VV.loc[v, cols]
        Xv = Xv.assign(thr=np.log(0.6)) if thr else Xv
        oof[n][va] = fit_aug(f, cuts, w, thr).predict(Xv.to_numpy("float32"))
    print(f"  year {k}: " + " | ".join(f"{n[:14]} {ap(y[va], oof[n][va]):.4f}" for n in VARIANTS), flush=True)

yearly = lambda p, m=None: np.array([ap(y[(year == k) & (m if m is not None else True)],
                                        p[(year == k) & (m if m is not None else True)]) for k in range(14)])
b, bn = yearly(oof["base (0.6 only)"]), yearly(oof["base (0.6 only)"], uns)
tlb = test_like(oof["base (0.6 only)"], n=500)
print("\n[64] vs base, leave-one-year-out + basin holdout:")
res = {}
for n in VARIANTS:
    a, an = yearly(oof[n]), yearly(oof[n], uns)
    tl = test_like(oof[n], n=500)
    res[n] = a - b
    print(f"  {n:34s} AP {a.mean():.4f} ({(a - b).mean():+.4f}, up {(a > b).sum()}/14) | new basins {(an - bn).mean():+.4f} "
          f"| test-like {(tl - tlb).mean():+.4f} ({(tl > tlb).mean():.0%} draws up)")

fig, ax = plt.subplots(figsize=(10, 4.5))
for i, n in enumerate(list(VARIANTS)[1:]):
    ax.bar(np.arange(14) + (i - 1) * 0.27, res[n] * 1000, width=0.27, label=n)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("validation year")
ax.set_ylabel("AP gain vs base (x1000)")
ax.set_title("Threshold augmentation: per-year gain at the real 0.6 cut")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "64_threshold_augmentation.png", dpi=120)
print(f"[fig] {FIG / '64_threshold_augmentation.png'}")
