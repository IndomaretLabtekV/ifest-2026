"""27 - How much of the remaining error needs information the dataset does not contain?
The label is a function of the HUC's own supply anomaly next month. Give the model oracle features
built from OTHER HUCs' anomalies in the TARGET month (leave-one-out basin mean, leave-one-out
nationwide mean, upstream mean). Those stand in for 'next month's regional weather', which a real
forecaster would get from a meteorological forecast and which this dataset does not provide.
The AP gap between the normal model and the oracle model is the headroom that no feature engineering
on these columns can reach."""
import importlib

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import FIG, WORK
from fe import network

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y, tr = h.FOLDS, h.DROP, h.PARAMS, h.y, h.tr
net = network(tr, tr)
basin = tr.id.map(net["basin"]).values
t_all, id_all = tr.t.values, tr.id.values


def oracle_block(F):
    """anomaly of each HUC in the target month, aggregated over other HUCs only"""
    a0 = pd.Series(F.anom0.values, index=pd.MultiIndex.from_arrays([id_all, t_all]))
    a0 = a0[~a0.index.duplicated()]
    nxt = a0.reindex(pd.MultiIndex.from_arrays([id_all, t_all + 1])).to_numpy()
    d = pd.DataFrame({"basin": basin, "t": t_all, "a": nxt, "id": id_all, "to_id": tr.to_id.values})
    g = d.groupby(["basin", "t"]).a
    s, n = g.transform("sum").to_numpy(), g.transform("count").to_numpy()
    loo_basin = (s - np.nan_to_num(nxt)) / np.maximum(n - np.isfinite(nxt), 1)
    gt = d.groupby("t").a
    s2, n2 = gt.transform("sum").to_numpy(), gt.transform("count").to_numpy()
    loo_all = (s2 - np.nan_to_num(nxt)) / np.maximum(n2 - np.isfinite(nxt), 1)
    up = d.groupby(["to_id", "t"]).a.mean()
    up_next = up.reindex(pd.MultiIndex.from_arrays([d.id, d.t])).to_numpy()
    return np.column_stack([loo_basin, loo_all, up_next]), nxt


res = {"model": [], "oracle": [], "oracle_only": []}
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    TF = pd.read_parquet(WORK / f"v2tr5_fold{i}.parquet")
    cols = [c for c in F.columns if c not in DROP]
    X0 = np.hstack([F[cols].to_numpy("float32"), TF.to_numpy("float32")])
    O, nxt = oracle_block(F)
    fit = trn & ~va & ~uns
    for tag, X in [("model", X0), ("oracle", np.hstack([X0, O])), ("oracle_only", O)]:
        m = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = m.predict(X[va], num_iteration=m.best_iteration)
        res[tag].append(ap(y[va], p))
    ok = np.isfinite(O[:, 0]) & va
    print(f"fold {i}: model {res['model'][-1]:.4f}  +oracle {res['oracle'][-1]:.4f}  oracle-only {res['oracle_only'][-1]:.4f}"
          f"  | corr(own next anomaly, basin LOO next anomaly) "
          f"{np.corrcoef(nxt[ok & np.isfinite(O[:, 0]) & np.isfinite(nxt)], O[ok & np.isfinite(O[:, 0]) & np.isfinite(nxt), 0])[0, 1]:.3f}", flush=True)

for k, v in res.items():
    print(f"{k:12s} mean AP {np.mean(v):.4f}  per-fold {np.round(v, 4).tolist()}")
gap = np.mean(res["oracle"]) - np.mean(res["model"])
print(f"\n[headroom] knowing other HUCs' target-month anomalies is worth {gap:+.4f} AP")
print(f"[headroom] remaining gap to a perfect ranker: {1 - np.mean(res['oracle']):.4f}")

plt.figure(figsize=(6, 3.5))
pd.Series({k: np.mean(v) for k, v in res.items()}).plot.bar(color=["#4C72B0", "#C44E52", "#999999"])
plt.axhline(np.mean(res["model"]), ls="--", c="k", lw=.8); plt.ylabel("CV AP")
plt.title("Headroom from next-month regional weather"); plt.tight_layout()
plt.savefig(FIG / "27_oracle_headroom.png", dpi=100); plt.close()
