"""16 - Big Trio + rank blends on the honest V2 harness.
Usage: python 16_v2_models.py [extra]   ('extra' uses the X1 feature set from 15)"""
import importlib
import sys

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y = h.FOLDS, h.DROP, h.PARAMS, h.y
EXTRA = "extra" in sys.argv[1:]


def data(i):
    F = h.fold_features(i, EXTRA)
    cols = [c for c in F.columns if c not in DROP]
    trn, va, uns = FOLDS[i]
    fit = trn & ~uns & ~va
    X = F[cols].to_numpy("float32")
    return X[fit], y[fit], X[va], y[va]


def run_lgb(Xa, ya, Xv, yv):
    m = lgb.train(PARAMS, lgb.Dataset(Xa, ya), 5000, valid_sets=[lgb.Dataset(Xv, yv)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xv, num_iteration=m.best_iteration)


def run_xgb(Xa, ya, Xv, yv):
    m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.03, max_depth=7, min_child_weight=20, subsample=0.8,
                          colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist", eval_metric="aucpr",
                          early_stopping_rounds=200, random_state=42, n_jobs=20)
    m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
    return m.predict_proba(Xv)[:, 1]


def run_cat(Xa, ya, Xv, yv):
    m = CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=8, l2_leaf_reg=5, od_type="Iter", od_wait=200,
                           random_seed=42, thread_count=20, verbose=False, allow_writing_files=False)
    m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
    return m.predict_proba(Xv)[:, 1]


P = {"lgb": [], "xgb": [], "cat": []}
for i in range(len(FOLDS)):
    d = data(i)
    for n, f in [("lgb", run_lgb), ("xgb", run_xgb), ("cat", run_cat)]:
        P[n].append(f(*d))
    print(f"fold {i} done", flush=True)

rank = lambda p: rankdata(p) / len(p)
tag = "X1" if EXTRA else "E4"
for n in P:
    h.evaluate(f"V2 {tag} {n}", P[n])
h.evaluate(f"V2 {tag} blend lgb+xgb+cat", [np.mean([rank(P[n][i]) for n in P], 0) for i in range(len(FOLDS))])
h.evaluate(f"V2 {tag} blend lgb+xgb", [(rank(P["lgb"][i]) + rank(P["xgb"][i])) / 2 for i in range(len(FOLDS))])
