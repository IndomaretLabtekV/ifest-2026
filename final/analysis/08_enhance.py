"""08 - Enhancements on top of the best 07 feature set, same folds.
E7  regression on the continuous target z = log(ws(t+1)/clim(t+1)) (recovered via the timeline),
    converted to P(z < log 0.6) with a Gaussian residual -> uses magnitude, not just the 0/1 side.
E8  classifier + regression prediction as a stacked feature (inner OOF, so no leakage).
E9  fold-aware HUC history encodings (label rate per HUC / per HUC x target month from the train part only).
E10 blend of classifier and regression probabilities."""
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score as ap

from common import LAG, WORK, load_clean
from prep import clim_scale

cv = importlib.import_module("07_cv_ablation")
F, y, t, ids, FOLDS, PARAMS = cv.F, cv.y, cv.t, cv.ids, cv.FOLDS, cv.PARAMS
F = F.drop(columns=["origin_anom0", "origin_below0"])
THR = np.log(0.6)

# continuous target: consensus ws of month t+1 across all rows containing it
tr = cv.tr.assign(t=t)
long = pd.concat([pd.DataFrame({"id": tr.id, "m": tr.t - k, "v": tr[f"q_ws_cum_m3_lag{k}"]}) for k in LAG])
long = long[long.v >= 0]
ws_next = long.groupby(["id", "m"]).v.median().reindex(pd.MultiIndex.from_arrays([tr.id, tr.t + 1])).to_numpy()
clim_next = (tr.clim_median_next * 10.0 ** clim_scale(tr)).to_numpy()
z = np.log((ws_next + 1) / (clim_next + 1))
has_z = ~np.isnan(z)
print(f"[target z] available {has_z.mean():.3f}; agreement (z<log0.6)==label: {((z[has_z] < THR) == y[has_z]).mean():.4f}")

REG = {**PARAMS, "objective": "huber", "alpha": 1.0, "metric": "l2"}
X = F.to_numpy("float32")


def reg_fit_predict(Xa, za, Xb, rounds=1500):
    m = lgb.train(REG, lgb.Dataset(Xa, za), rounds)
    resid = np.std(za - m.predict(Xa))
    return m.predict(Xb), resid


def clf_fit_predict(Xa, ya, Xb, Xv, yv):
    m = lgb.train(PARAMS, lgb.Dataset(Xa, ya), 3000, valid_sets=[lgb.Dataset(Xv, yv)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(Xb, num_iteration=m.best_iteration)


def hist_enc(trn, rows):
    d = pd.DataFrame({"id": ids[trn], "tm": F.month.values[trn] % 12 + 1, "y": y[trn]})
    g1 = d.groupby("id").y.agg(["mean", "size"])
    g2 = d.groupby(["id", "tm"]).y.mean()
    q = pd.DataFrame({"id": ids[rows], "tm": F.month.values[rows] % 12 + 1})
    prior = y[trn].mean()
    # smoothed with 20 pseudo-observations towards the global rate
    e1 = q.id.map((g1["mean"] * g1["size"] + prior * 20) / (g1["size"] + 20))
    e2 = g2.reindex(pd.MultiIndex.from_arrays([q.id, q.tm])).to_numpy()
    return np.column_stack([e1.to_numpy(), e2]).astype("float32")


rows = []
for i, (trn, va, uns) in enumerate(FOLDS):
    zt = trn & has_z
    # E7 regression -> probability
    mu, s = reg_fit_predict(X[zt], z[zt], X[va])
    p_reg = norm.cdf((THR - mu) / s)
    # E3 baseline classifier (re-fit here so blends use identical models)
    p_clf = clf_fit_predict(X[trn], y[trn], X[va], X[va], y[va])
    # E8 stacked: inner 2-fold OOF regression prediction as a feature
    idx = np.where(zt)[0]
    half = np.isin(ids[idx], np.unique(ids[idx])[::2])
    oof = np.full(len(y), np.nan, dtype="float32")
    for a, b in [(idx[half], idx[~half]), (idx[~half], idx[half])]:
        oof[b], _ = reg_fit_predict(X[a], z[a], X[b])
    trn_nz = np.where(trn)[0]
    oof_trn = oof[trn_nz]
    Xs_tr = np.column_stack([X[trn_nz], oof_trn])
    Xs_va = np.column_stack([X[va], mu])
    p_stk = clf_fit_predict(Xs_tr, y[trn_nz], Xs_va, Xs_va, y[va])
    # E9 HUC history encodings
    Xh_tr = np.column_stack([X[trn], hist_enc(trn, trn)])
    Xh_va = np.column_stack([X[va], hist_enc(trn, va)])
    p_hist = clf_fit_predict(Xh_tr, y[trn], Xh_va, Xh_va, y[va])
    r = {"fold": i}
    for n, p in [("E3 clf", p_clf), ("E7 reg->prob", p_reg), ("E8 clf+reg feature", p_stk),
                 ("E9 clf+HUC history", p_hist), ("E10 blend clf/reg 50/50", (p_clf + p_reg) / 2),
                 ("E10b blend stk/reg", (p_stk + p_reg) / 2)]:
        v = y[va]
        r[n] = ap(v, p); r[n + " |seen"] = ap(v[~uns[va]], p[~uns[va]]); r[n + " |unseen"] = ap(v[uns[va]], p[uns[va]])
    rows.append(r)
    print({k: round(v, 4) for k, v in r.items() if "|" not in k}, flush=True)

R = pd.DataFrame(rows).drop(columns="fold")
out = pd.DataFrame({"mean": R.mean(), "std": R.std()}).round(4)
print(out.to_string())
out.to_csv(WORK / "08_enhance.csv")
