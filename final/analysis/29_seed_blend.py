"""29 - Does seed averaging still pay inside the 3-model blend?
26 showed +0.23 pt for a 5-seed average of one LightGBM. The V4 submission already averages three model
families, which removes part of the variance, so the gain has to be re-measured in that setting:
  blend-1seed  = rank mean of (lgb, xgb, cat) hybrids, one seed each  (this is V4)
  blend-3seed  = rank mean of the same three families with 3 seeds each"""
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap

from common import WORK

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y = h.FOLDS, h.DROP, h.PARAMS, h.y
HIST_LGB = {**PARAMS, "learning_rate": 0.0469, "num_leaves": 187, "min_data_in_leaf": 39,
            "feature_fraction": 0.316, "bagging_fraction": 0.539, "lambda_l2": 4.21, "max_bin": 255}
SEEDS = [42, 1, 7]


def fit_lgb(Xa, ya, Xv, yv, seed, params):
    p = {**params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
    m = lgb.train(p, lgb.Dataset(Xa, ya), 5000, valid_sets=[lgb.Dataset(Xv, yv)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xv, num_iteration=m.best_iteration)


def fit_xgb(Xa, ya, Xv, yv, seed, _):
    m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.03, max_depth=7, min_child_weight=20, subsample=0.8,
                          colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist", eval_metric="aucpr",
                          early_stopping_rounds=200, random_state=seed, n_jobs=20)
    m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
    return m.predict_proba(Xv)[:, 1]


def fit_cat(Xa, ya, Xv, yv, seed, _):
    m = CatBoostClassifier(iterations=3000, learning_rate=0.06, depth=8, l2_leaf_reg=5, od_type="Iter", od_wait=200,
                           random_seed=seed, thread_count=20, verbose=False, allow_writing_files=False)
    m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
    return m.predict_proba(Xv)[:, 1]


FAM = {"lgb": (fit_lgb, HIST_LGB), "xgb": (fit_xgb, None), "cat": (fit_cat, None)}
rank = lambda p: rankdata(p) / len(p)
P = {f: {s: [] for s in SEEDS} for f in FAM}
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    TF = pd.read_parquet(WORK / f"v2tr5_fold{i}.parquet")
    cols = [c for c in F.columns if c not in DROP]
    X0 = F[cols].to_numpy("float32")
    Xh = np.hstack([X0, TF.to_numpy("float32")])
    fit = trn & ~va & ~uns
    pb = fit_lgb(X0[fit], y[fit], X0[va], y[va], 42, PARAMS)  # base model for held-out HUCs
    for f, (fn, pr) in FAM.items():
        for s in SEEDS:
            ph = fn(Xh[fit], y[fit], Xh[va], y[va], s, pr)
            P[f][s].append(np.where(uns[va], pb, ph))
    print(f"fold {i} done", flush=True)

one = [np.mean([rank(P[f][42][i]) for f in FAM], 0) for i in range(len(FOLDS))]
three = [np.mean([rank(P[f][s][i]) for f in FAM for s in SEEDS], 0) for i in range(len(FOLDS))]
for name, preds in [("blend-1seed (V4)", one), ("blend-3seed", three)]:
    per = [ap(y[va], p) for (_, va, _), p in zip(FOLDS, preds)]
    tl = np.mean([h.test_like_ap(p, va)[0] for (_, va, _), p in zip(FOLDS, preds)])
    print(f"{name:18s} AP {np.mean(per):.4f} | test-like {tl:.4f} | per-fold {np.round(per, 4).tolist()}")
for f in FAM:
    per_seed = [np.mean([ap(y[va], P[f][s][i]) for i, (_, va, _) in enumerate(FOLDS)]) for s in SEEDS]
    print(f"  {f}: per-seed AP {np.round(per_seed, 4).tolist()} (spread {max(per_seed) - min(per_seed):.4f})")
