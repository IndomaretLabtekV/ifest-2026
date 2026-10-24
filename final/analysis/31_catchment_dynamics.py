"""31 - Final audit of unused signal: catchment dynamics from the row's own 12 anomalies.
Everything here is row-internal and scale-free, so unlike the history features of V3 it also applies to
test-only HUCs, which is exactly where the pipeline still relies on the plain model.
  ar1           lag-1 autocorrelation of the 12-month anomaly series (catchment memory)
  flashiness    mean |a_k - a_{k+1}| divided by the spread (Richards-Baker style)
  slope         linear trend over the 12 months
  since_below   months since the series was last below the 0.6 threshold
  amp_ratio     (max - min) of the anomaly series
  use_trend     log(use now / use 3 months ago), th and ps shares (the only unused column family)
Screened by univariate AUC per zone, then tested in the honest V2/V3 harness."""
import importlib

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import FIG, WORK

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y, tr = h.FOLDS, h.DROP, h.PARAMS, h.y, h.tr
THR = np.log(0.6)


from fe import dynamics

print("[screen] univariate AUC per zone (fold 1 validation rows)")
trn, va, uns = FOLDS[1]
F1 = h.fold_features(1)
D1 = dynamics(F1, pd.concat([tr[~va], h.rescale(tr[va], seed=101)]).sort_index())
m0 = F1.anom0.values - THR
for zone, mask in [("dry", m0 < 0), ("onset", (m0 >= 0) & (m0 < .5)), ("all", np.ones(len(m0), bool))]:
    sel = mask & va
    line = []
    for c in D1.columns:
        v = D1[c].values[sel]
        ok = np.isfinite(v)
        a = auc(y[sel][ok], v[ok])
        line.append(f"{c} {max(a, 1 - a):.3f}")
    print(f"  [{zone:5s}] n={sel.sum():6d} " + " | ".join(line))

res = {"base": [], "dynamics": []}
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    TF = pd.read_parquet(WORK / f"v2tr5_fold{i}.parquet")
    D = dynamics(F, pd.concat([tr[~va], h.rescale(tr[va], seed=100 + i)]).sort_index())
    cols = [c for c in F.columns if c not in DROP]
    X0 = np.hstack([F[cols].to_numpy("float32"), TF.to_numpy("float32")])
    fit = trn & ~va & ~uns
    for tag, X in [("base", X0), ("dynamics", np.hstack([X0, D.to_numpy("float32")]))]:
        m = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = m.predict(X[va], num_iteration=m.best_iteration)
        ph = np.where(uns[va], p, p)
        dz = (F.anom0.values - THR)[va] < 0
        res[tag].append((ap(y[va], ph), ap(y[va][uns[va]], ph[uns[va]]), auc(y[va][dz], ph[dz])))
    print(f"fold {i}: base {res['base'][-1][0]:.4f} | +dynamics {res['dynamics'][-1][0]:.4f}"
          f" | unseen {res['base'][-1][1]:.4f} -> {res['dynamics'][-1][1]:.4f}"
          f" | dry AUC {res['base'][-1][2]:.4f} -> {res['dynamics'][-1][2]:.4f}", flush=True)

for k, v in res.items():
    v = np.array(v)
    print(f"{k:9s} AP {v[:, 0].mean():.4f} | unseen {v[:, 1].mean():.4f} | dry-zone AUC {v[:, 2].mean():.4f}")

imp = lgb.train(PARAMS, lgb.Dataset(np.hstack([F1[[c for c in F1.columns if c not in DROP]].to_numpy("float32"),
                                               D1.to_numpy("float32")])[FOLDS[1][0] & ~FOLDS[1][1] & ~FOLDS[1][2]],
                                    y[FOLDS[1][0] & ~FOLDS[1][1] & ~FOLDS[1][2]]), 300)
gain = pd.Series(imp.feature_importance("gain"))
names = [c for c in F1.columns if c not in DROP] + list(D1.columns)
share = (gain / gain.sum()).set_axis(names).sort_values(ascending=False)
print("\n[importance] share of gain taken by the new dynamics features:",
      round(share[list(D1.columns)].sum(), 4), "| individually:", share[list(D1.columns)].round(4).to_dict())
plt.figure(figsize=(7, 4)); share.head(15).sort_values().plot.barh(color="#4C72B0")
plt.title("gain share, top features with dynamics added"); plt.tight_layout()
plt.savefig(FIG / "31_catchment_dynamics.png", dpi=100); plt.close()
