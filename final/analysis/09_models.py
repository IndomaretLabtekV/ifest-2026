"""09 - Model comparison on the E4 feature set, identical folds.
LightGBM configs (ours / stronger regularisation / Frontiers-2025 streamflow-drought config),
XGBoost, CatBoost, and rank blends. Local run is CPU; the notebook uses GPU for XGB/CatBoost."""
import importlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap

cv = importlib.import_module("07_cv_ablation")
F, y, FOLDS, BASE = cv.F, cv.y, cv.FOLDS, cv.PARAMS
X = F.drop(columns=["origin_anom0", "origin_below0"]).to_numpy("float32")

LGB = {
    "lgb base (07)": BASE,
    "lgb regularised": {**BASE, "learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 500,
                        "feature_fraction": 0.5, "lambda_l2": 10.0},
    "lgb frontiers25": {**BASE, "num_leaves": 63, "min_data_in_leaf": 100, "max_depth": 7,
                        "bagging_fraction": 0.1, "bagging_freq": 10, "max_bin": 127},
}


def fit_lgb(p, trn, va):
    m = lgb.train(p, lgb.Dataset(X[trn], y[trn]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(X[va], num_iteration=m.best_iteration), m.best_iteration


def fit_xgb(trn, va):
    m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.03, max_depth=7, min_child_weight=20, subsample=0.8,
                          colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist", eval_metric="aucpr",
                          early_stopping_rounds=200, random_state=42, n_jobs=20)
    m.fit(X[trn], y[trn], eval_set=[(X[va], y[va])], verbose=False)
    return m.predict_proba(X[va])[:, 1], m.best_iteration


def fit_cat(trn, va):
    m = CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=8, l2_leaf_reg=5, eval_metric="PRAUC",
                           od_type="Iter", od_wait=200, random_seed=42, thread_count=20, verbose=False)
    m.fit(X[trn], y[trn], eval_set=(X[va], y[va]), use_best_model=True)
    return m.predict_proba(X[va])[:, 1], m.best_iteration_


which = sys.argv[1:] or list(LGB) + ["xgb", "cat"]
P = {n: [] for n in which}
for i, (trn, va, uns) in enumerate(FOLDS):
    for n in which:
        if n in LGB:
            p, it = fit_lgb(LGB[n], trn, va)
        elif n == "xgb":
            p, it = fit_xgb(trn, va)
        else:
            p, it = fit_cat(trn, va)
        P[n].append(p)
        print(f"fold {i} {n:18s} AP {ap(y[va], p):.4f} iters {it}", flush=True)


def score(preds):
    r = []
    for (trn, va, uns), p in zip(FOLDS, preds):
        r.append((ap(y[va], p), ap(y[va][~uns[va]], p[~uns[va]]), ap(y[va][uns[va]], p[uns[va]])))
    return np.array(r)


rank = lambda p: rankdata(p) / len(p)
res = {n: score(P[n]) for n in which}
lgb_names = [n for n in which if n in LGB]
if {"xgb", "cat"} <= set(which) and lgb_names:
    best_lgb = max(lgb_names, key=lambda n: res[n][:, 0].mean())
    trio = [best_lgb, "xgb", "cat"]
    res["blend trio (rank mean)"] = score([np.mean([rank(P[n][i]) for n in trio], 0) for i in range(len(FOLDS))])
    res["blend all (rank mean)"] = score([np.mean([rank(P[n][i]) for n in which], 0) for i in range(len(FOLDS))])
tab = pd.DataFrame({n: {"AP": r[:, 0].mean(), "std": r[:, 0].std(), "seen": r[:, 1].mean(), "unseen": r[:, 2].mean(),
                        "per-fold": np.round(r[:, 0], 4).tolist()} for n, r in res.items()}).T
print(tab.to_string())
np.save(cv.WORK / "09_oof.npy", {n: P[n] for n in which}, allow_pickle=True)
