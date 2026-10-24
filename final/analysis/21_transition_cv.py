"""21 - Transition-probability features (fe.transition_features) on the honest V2 harness.
Fit rows: history = train part. Validation rows: history = train part + validation rows (other years
only), mirroring the test where every HUC also has its own 4 test rows as history."""
import importlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import WORK
from fe import clim_table_split, transition_features
from prep import repair

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y, tr = h.FOLDS, h.DROP, h.PARAMS, h.y, h.tr
THR = np.log(0.6)
MIN_HIST = int(sys.argv[1]) if len(sys.argv) > 1 else 3
res = {"base": [], "trans": [], "trans_drop": [], "hybrid": []}
PRED = {}
rng = np.random.default_rng(42)
for i, (trn, va, uns) in enumerate(FOLDS):
    path = WORK / f"v2tr{MIN_HIST}_fold{i}.parquet"
    F = h.fold_features(i)
    part = ~va & ~uns
    if path.exists():
        TF = pd.read_parquet(path)
    else:
        tp = repair(tr[part])
        # clim_next for the query = ws0 / exp(margin_raw_persist + THR) recovers the scale-consistent clim_next
        tp["_clim_next"] = tp.q_ws_cum_m3_lag0.abs().values / np.exp(F.margin_raw_persist.values[part] + THR)
        sv = repair(h.rescale(tr[va], seed=100 + i))
        sv["_clim_next"] = sv.q_ws_cum_m3_lag0.abs().values / np.exp(F.margin_raw_persist.values[va] + THR)
        TF = pd.concat([transition_features(tp, tp, MIN_HIST), transition_features(sv, pd.concat([tp, sv]), MIN_HIST)]).reindex(tr.index)
        TF.to_parquet(path)
    cols = [c for c in F.columns if c not in DROP]
    fit = trn & part
    dry = (F.anom0.values - THR) < 0
    Xt = np.hstack([F[cols].to_numpy("float32"), TF.to_numpy("float32")])
    Xd = Xt.copy()
    # feature dropout: hide the history features on 30% of fit rows so the model also learns the no-history regime
    drop_rows = fit & (rng.random(len(Xd)) < 0.3)
    Xd[np.ix_(drop_rows, range(len(cols), Xd.shape[1]))] = np.nan
    for tag, X in [("base", F[cols].to_numpy("float32")), ("trans", Xt), ("trans_drop", Xd)]:
        m = lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        p = m.predict((Xt if tag == "trans_drop" else X)[va], num_iteration=m.best_iteration)
        dz = dry[va]
        PRED[tag] = p
        res[tag].append((ap(y[va], p), ap(y[va][~uns[va]], p[~uns[va]]), ap(y[va][uns[va]], p[uns[va]]),
                         auc(y[va][dz], p[dz]), h.test_like_ap(p, va)[0]))
        print(f"fold {i} {tag:5s} AP {res[tag][-1][0]:.4f} seen {res[tag][-1][1]:.4f} unseen {res[tag][-1][2]:.4f}"
              f" dry AUC {res[tag][-1][3]:.4f} test-like {res[tag][-1][4]:.4f}", flush=True)
    # hybrid: history model for seen HUCs, plain V2 model for unseen HUCs, pooled into one ranking
    u = uns[va]
    p = np.where(u, PRED["base"], PRED["trans"])
    dz = dry[va]
    res["hybrid"].append((ap(y[va], p), ap(y[va][~u], p[~u]), ap(y[va][u], p[u]), auc(y[va][dz], p[dz]), h.test_like_ap(p, va)[0]))
    print(f"fold {i} hybrid AP {res['hybrid'][-1][0]:.4f} test-like {res['hybrid'][-1][4]:.4f}", flush=True)
    print(f"   coverage valid: seen {TF.trans_p_exceed.notna().values[va & ~uns].mean():.3f}"
          f" unseen {TF.trans_p_exceed.notna().values[va & uns].mean():.3f}", flush=True)
for tag, r in res.items():
    r = np.array(r)
    print(f"{tag:5s} AP {r[:, 0].mean():.4f} | seen {r[:, 1].mean():.4f} unseen {r[:, 2].mean():.4f} | "
          f"dry AUC {r[:, 3].mean():.4f} | test-like {r[:, 4].mean():.4f} | per-fold {np.round(r[:, 0], 4).tolist()}")
