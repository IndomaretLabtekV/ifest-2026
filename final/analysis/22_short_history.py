"""22 - Can the history model serve test-only HUCs, which have exactly 3 years of history (their other
3 test rows)? V3 sends them to the base model because MIN_HIST=5.
Emulation: held-out HUCs are never trained on. Their rows OUTSIDE the validation block act as their
'own other rows'; history is subsampled to exactly K years (K = 2, 3, 5) like the test. Compare on
validation rows of held-out HUCs: base model vs history model fed K-year features, and a history
model trained without trans_n (n=3 never occurs in training)."""
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import WORK
from fe import transition_features
from prep import repair

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y, tr = h.FOLDS, h.DROP, h.PARAMS, h.y, h.tr
THR = np.log(0.6)
R = []
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    TF = pd.read_parquet(WORK / f"v2tr5_fold{i}.parquet")
    cols = [c for c in F.columns if c not in DROP]
    part = ~va & ~uns
    fit = trn & part
    X0 = F[cols].to_numpy("float32")
    Xh = np.hstack([X0, TF.to_numpy("float32")])
    Xh_non = np.hstack([X0, TF.drop(columns="trans_n").to_numpy("float32")])
    train_ = lambda X: lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                                 callbacks=[lgb.early_stopping(200, verbose=False)])
    mb, mh, mhn = train_(X0), train_(Xh), train_(Xh_non)
    # held-out HUCs: history = their own rows outside the validation block (never trained on)
    q_rows = va & uns
    src_rows = uns & ~va
    sv = repair(h.rescale(tr[va], seed=100 + i))
    src = repair(tr[src_rows])
    q = sv[uns[va]].copy()
    q["_clim_next"] = q.q_ws_cum_m3_lag0.abs().values / np.exp(F.margin_raw_persist.values[q_rows] + THR)
    pb = mb.predict(X0[q_rows], num_iteration=mb.best_iteration)
    yq = y[q_rows]
    line = {"fold": i, "base": ap(yq, pb)}
    for K in (2, 3, 5):
        T = transition_features(q, src, min_hist=K, max_years=K, seed=i).to_numpy("float32")
        cov = ~np.isnan(T[:, 0])
        ph = mh.predict(np.hstack([X0[q_rows], T]), num_iteration=mh.best_iteration)
        phn = mhn.predict(np.hstack([X0[q_rows], T[:, :3]]), num_iteration=mhn.best_iteration)
        # rows without enough history fall back to base, as in the notebook
        line[f"hist K={K}"] = ap(yq, np.where(cov, ph, pb))
        line[f"hist-no-n K={K}"] = ap(yq, np.where(cov, phn, pb))
        line[f"cov K={K}"] = cov.mean()
    R.append(line)
    print({k: round(v, 4) for k, v in line.items()}, flush=True)
R = pd.DataFrame(R)
print(R.drop(columns="fold").mean().round(4).to_string())
R.to_csv(WORK / "22_short_history.csv", index=False)
