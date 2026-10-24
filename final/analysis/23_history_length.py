"""23 - Does the history model's gain grow with the number of history years?
V3 CV predicted +0.3 pt but the public score moved +3.2 pt (0.720 -> 0.752). In CV, seen validation HUCs
have ~8-9 usable history years (validation block + purge removed); in the test they have all 14 train
years. Validation rows of seen HUCs are re-featurised with history capped at K years and scored."""
import importlib

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from fe import transition_features
from prep import repair

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y, tr = h.FOLDS, h.DROP, h.PARAMS, h.y, h.tr
THR = np.log(0.6)
KS = [2, 3, 5, 8, None]
R = []
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    TF = pd.read_parquet(WORK / f"v2tr5_fold{i}.parquet")
    cols = [c for c in F.columns if c not in DROP]
    part = ~va & ~uns
    fit = trn & part
    X0 = F[cols].to_numpy("float32")
    Xh = np.hstack([X0, TF.to_numpy("float32")])
    train_ = lambda X: lgb.train(PARAMS, lgb.Dataset(X[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                                 callbacks=[lgb.early_stopping(200, verbose=False)])
    mb, mh = train_(X0), train_(Xh)
    q_rows = va & ~uns
    tp = repair(tr[part])
    sv = repair(h.rescale(tr[va], seed=100 + i))
    q = sv[~uns[va]].copy()
    q["_clim_next"] = q.q_ws_cum_m3_lag0.abs().values / np.exp(F.margin_raw_persist.values[q_rows] + THR)
    hist = pd.concat([tp, sv])
    pb = mb.predict(X0[q_rows], num_iteration=mb.best_iteration)
    yq = y[q_rows]
    line = {"fold": i, "base": ap(yq, pb)}
    for K in KS:
        T = transition_features(q, hist, min_hist=K or 5, max_years=K, seed=i).to_numpy("float32")
        cov = ~np.isnan(T[:, 0])
        ph = mh.predict(np.hstack([X0[q_rows], T]), num_iteration=mh.best_iteration)
        line[f"K={K or 'all'}"] = ap(yq, np.where(cov, ph, pb))
        if K is None:
            line["median years (all)"] = float(np.nanmedian(T[:, 3]))
    R.append(line)
    print({k: round(v, 4) for k, v in line.items()}, flush=True)
R = pd.DataFrame(R)
m = R.drop(columns="fold").mean()
print(m.round(4).to_string())
gain = pd.Series({k: m[k] - m["base"] for k in m.index if k.startswith("K=")})
plt.figure(figsize=(6, 3.5)); gain.plot(marker="o"); plt.axhline(0, c="k", lw=.8)
plt.title("AP gain of history model on seen HUCs vs history years"); plt.ylabel("AP gain"); plt.tight_layout()
plt.savefig(FIG / "23_history_length.png", dpi=100); plt.close()
