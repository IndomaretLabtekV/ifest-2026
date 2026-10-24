"""19 - Seasonal analog features (hydrological analog forecasting), leakage-safe, on the V2 harness.
For a query row (HUC h, origin month m, time t): candidates are train-part rows of the SAME HUC and SAME
origin month from other years (|t - t'| >= 12). Distance = RMS over the 6-month anomaly trajectory
anom0..anom5. Features: label rate of the K nearest analogs (K=3, 7), their mean distance, and the
analog count. Unseen HUCs get NaN, exactly like test-only HUCs.
18 showed the dry zone (recovery vs persistence) holds the largest headroom; analogs encode how this
particular catchment recovered in similar past situations."""
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import WORK

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y, tr = h.FOLDS, h.DROP, h.PARAMS, h.y, h.tr
STATE = [f"anom{k}" for k in range(6)]
t_all, id_all, mon_all = tr.t.values, tr.id.values, tr.month.values


def analog_feats(F, source, query):
    """source/query: boolean masks. Returns (n_rows, 4) array, NaN where not computed."""
    out = np.full((len(F), 4), np.nan, dtype="float32")
    S = F[STATE].to_numpy("float32")
    src = pd.DataFrame({"i": np.where(source)[0], "id": id_all[source], "m": mon_all[source]})
    qry = pd.DataFrame({"i": np.where(query)[0], "id": id_all[query], "m": mon_all[query]})
    groups = src.groupby(["id", "m"]).i.apply(np.array).to_dict()
    for (hid, m), qi in qry.groupby(["id", "m"]).i:
        si = groups.get((hid, m))
        if si is None:
            continue
        qi = qi.to_numpy()
        d = np.sqrt(np.nanmean((S[qi, None, :] - S[None, si, :]) ** 2, axis=2))
        d[np.abs(t_all[qi, None] - t_all[None, si]) < 12] = np.inf
        order = np.argsort(d, axis=1)
        ds = np.take_along_axis(d, order, 1)
        ys = y[si][order]
        valid = np.isfinite(ds)
        n = valid.sum(1)
        for j, k in enumerate((3, 7)):
            kk = np.minimum(k, n)
            w = valid[:, :k]
            out[qi, j] = np.where(kk > 0, (ys[:, :k] * w).sum(1) / np.maximum(w.sum(1), 1), np.nan)
        out[qi, 2] = np.where(n > 0, np.where(valid[:, :3], ds[:, :3], 0).sum(1) / np.maximum(valid[:, :3].sum(1), 1), np.nan)
        out[qi, 3] = n
    return out


names = ["analog_rate3", "analog_rate7", "analog_dist3", "analog_n"]
res = {"base": [], "analog": []}
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    part = ~va & ~uns
    A = analog_feats(F, source=part, query=part | va)
    cols = [c for c in F.columns if c not in DROP]
    fit = trn & part
    zone_dry = (F.anom0.values - np.log(0.6)) < 0
    for tag, X in [("base", F[cols].to_numpy("float32")), ("analog", np.hstack([F[cols].to_numpy("float32"), A]))]:
        m = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = m.predict(X[va], num_iteration=m.best_iteration)
        dz = zone_dry[va]
        res[tag].append((ap(y[va], p), ap(y[va][~uns[va]], p[~uns[va]]), ap(y[va][uns[va]], p[uns[va]]),
                         auc(y[va][dz], p[dz]), h.test_like_ap(p, va)[0]))
        print(f"fold {i} {tag:7s} AP {res[tag][-1][0]:.4f} seen {res[tag][-1][1]:.4f} unseen {res[tag][-1][2]:.4f}"
              f" dry-zone AUC {res[tag][-1][3]:.4f} test-like {res[tag][-1][4]:.4f}", flush=True)
    a = A[va][:, 1]
    ok = ~np.isnan(a)
    print(f"   analog_rate7 coverage on valid {ok.mean():.3f}; univariate AUC {auc(y[va][ok], a[ok]):.3f};"
          f" inside dry zone {auc(y[va][ok & zone_dry[va]], a[ok & zone_dry[va]]):.3f}", flush=True)

for tag, r in res.items():
    r = np.array(r)
    print(f"{tag:7s} AP {r[:, 0].mean():.4f} | seen {r[:, 1].mean():.4f} unseen {r[:, 2].mean():.4f} | "
          f"dry-zone AUC {r[:, 3].mean():.4f} | test-like {r[:, 4].mean():.4f}")
