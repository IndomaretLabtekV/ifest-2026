"""18 - Where is the remaining AP lost, and what is worth fixing?
OOF LightGBM on the honest V2 harness (rescaled validation). Rows are split into zones by the current
margin m0 = log(ws0/clim_now) - log(0.6):
  dry      m0 < 0         (already below threshold -> question is recovery)
  onset    0 <= m0 < 0.5  (just above -> question is whether it drops)
  safe     m0 >= 0.5
Oracle test: make the ranking inside one zone perfect (keep the others), recompute global AP.
The gain is an upper bound on what fixing that zone can buy."""
import importlib

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import FIG, WORK

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y = h.FOLDS, h.DROP, h.PARAMS, h.y
THR = np.log(0.6)

rows = []
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    cols = [c for c in F.columns if c not in DROP]
    X = F[cols].to_numpy("float32")
    fit = trn & ~uns & ~va
    m = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    d = F[va].copy()
    d["p"], d["y"], d["fold"] = m.predict(X[va], num_iteration=m.best_iteration), y[va], i
    rows.append(d)
D = pd.concat(rows)
D.to_parquet(WORK / "18_oof.parquet")
m0 = D.anom0 - THR
D["zone"] = np.select([m0 < 0, m0 < 0.5], ["dry", "onset"], "safe")

print("[zones] share of rows / share of positives / positive rate / AP inside zone:")
for z, g in D.groupby("zone"):
    print(f"  {z:6s} rows {len(g) / len(D):.3f}  positives {g.y.sum() / D.y.sum():.3f}  rate {g.y.mean():.3f}"
          f"  AP-in-zone {ap(g.y, g.p):.3f}  AUC-in-zone {auc(g.y, g.p):.3f}")


def fold_ap(D, p):
    return np.mean([ap(g.y, p[g.index_pos]) for _, g in D.assign(index_pos=np.arange(len(D))).groupby("fold")])


base = fold_ap(D, D.p.values)
print(f"\n[oracle] baseline mean fold AP {base:.4f}")
for z in ["dry", "onset", "safe"]:
    p = D.p.values.copy()
    inz = (D.zone == z).values
    # perfect ordering inside the zone while keeping the zone's score range (positives on top of that range)
    for f in D.fold.unique():
        sel = inz & (D.fold == f).values
        vals = np.sort(p[sel])
        order = np.argsort(-(D.y.values[sel] + 1e-6 * rankdata(p[sel])))
        new = np.empty(sel.sum()); new[order] = vals[::-1]
        p[sel] = new
    print(f"[oracle] perfect ranking inside '{z}': mean fold AP {fold_ap(D, p):.4f}  (gain {fold_ap(D, p) - base:+.4f})")

# which features carry signal inside the onset zone?
on = D[D.zone == "onset"]
feats = [c for c in D.columns if c not in ("p", "y", "fold", "zone") + tuple(DROP)]
uni = sorted(((c, abs(auc(on.y, on[c].fillna(on[c].median())) - .5) + .5) for c in feats), key=lambda x: -x[1])
print("\n[onset] top univariate AUC inside onset zone:", [(c, round(a, 3)) for c, a in uni[:12]])
print(f"[onset] model AUC inside onset zone: {auc(on.y, on.p):.3f}")
dry = D[D.zone == "dry"]
uni_d = sorted(((c, abs(auc(dry.y, dry[c].fillna(dry[c].median())) - .5) + .5) for c in feats), key=lambda x: -x[1])
print("[dry] top univariate AUC inside dry zone (recovery):", [(c, round(a, 3)) for c, a in uni_d[:8]])

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
b = pd.cut(m0, np.arange(-2, 2.01, .1))
D.groupby(b, observed=True).agg(rate=("y", "mean"), pred=("p", "mean")).plot(ax=ax[0], marker=".")
ax[0].set_title("positive rate and mean prediction vs current margin"); ax[0].tick_params(axis="x", rotation=90)
pd.Series(dict(uni[:12])).sort_values().plot.barh(ax=ax[1], color="#C44E52"); ax[1].set_title("univariate AUC inside onset zone")
plt.tight_layout(); plt.savefig(FIG / "18_error_zones.png", dpi=100); plt.close()
