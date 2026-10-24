"""33 - Confirm the catchment-dynamics features in the real V5 setup (hybrid + blend).
31 tested them with the history model applied to every row. Here they are added to BOTH members of the
hybrid: the plain model that scores test-only HUCs and the history model that scores known HUCs."""
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import WORK

h = importlib.import_module("13_v2_cv")
from fe import dynamics
FOLDS, DROP, PARAMS, y, tr = h.FOLDS, h.DROP, h.PARAMS, h.y, h.tr
HIST = {**PARAMS, "learning_rate": 0.0469, "num_leaves": 187, "min_data_in_leaf": 39,
        "feature_fraction": 0.316, "bagging_fraction": 0.539, "lambda_l2": 4.21, "max_bin": 255}
THR = np.log(0.6)

res = {"V5 hybrid": [], "V6 hybrid (+dynamics)": []}
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    TF = pd.read_parquet(WORK / f"v2tr5_fold{i}.parquet")
    D = dynamics(F, pd.concat([tr[~va], h.rescale(tr[va], seed=100 + i)]).sort_index())
    cols = [c for c in F.columns if c not in DROP]
    B = F[cols].to_numpy("float32")
    Dn = D.to_numpy("float32")
    T = TF.to_numpy("float32")
    fit = trn & ~va & ~uns

    def run(Xb, Xh):
        mb = lgb.train(PARAMS, lgb.Dataset(Xb[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(Xb[va], y[va])],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
        mh = lgb.train(HIST, lgb.Dataset(Xh[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(Xh[va], y[va])],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
        pb = mb.predict(Xb[va], num_iteration=mb.best_iteration)
        ph = mh.predict(Xh[va], num_iteration=mh.best_iteration)
        return np.where(uns[va], pb, ph)

    p5 = run(B, np.hstack([B, T]))
    p6 = run(np.hstack([B, Dn]), np.hstack([B, T, Dn]))
    dz = (F.anom0.values - THR)[va] < 0
    for tag, p in [("V5 hybrid", p5), ("V6 hybrid (+dynamics)", p6)]:
        res[tag].append((ap(y[va], p), ap(y[va][~uns[va]], p[~uns[va]]), ap(y[va][uns[va]], p[uns[va]]),
                         auc(y[va][dz], p[dz]), h.test_like_ap(p, va)[0]))
    print(f"fold {i}: V5 {res['V5 hybrid'][-1][0]:.4f} -> V6 {res['V6 hybrid (+dynamics)'][-1][0]:.4f}"
          f" | unseen {res['V5 hybrid'][-1][2]:.4f} -> {res['V6 hybrid (+dynamics)'][-1][2]:.4f}", flush=True)

for k, v in res.items():
    v = np.array(v)
    print(f"{k:24s} AP {v[:, 0].mean():.4f} | seen {v[:, 1].mean():.4f} unseen {v[:, 2].mean():.4f} | "
          f"dry AUC {v[:, 3].mean():.4f} | test-like {v[:, 4].mean():.4f} | per-fold {np.round(v[:, 0], 4).tolist()}")
d = np.array(res["V6 hybrid (+dynamics)"])[:, 0] - np.array(res["V5 hybrid"])[:, 0]
print(f"[verdict] gain {d.mean():+.4f}, positive in {int((d > 0).sum())}/4 folds (seed noise from 26 is 0.0005)")
