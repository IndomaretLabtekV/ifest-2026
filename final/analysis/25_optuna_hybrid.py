"""25 - Optuna HPO for the V3 hybrid (allowed by rule 5).
Objective: mean AP of the HYBRID on folds 1 and 2 (the two hardest), where the history model scores
seen-HUC rows and the base model (fixed V2 params) scores held-out HUCs. Only the history model is tuned,
since it is the new V3 component still running the base-model parameters.
The best trial is then re-validated on all 4 folds against the V3 default."""
import importlib

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import WORK

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y = h.FOLDS, h.DROP, h.PARAMS, h.y
DATA = {}
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    TF = pd.read_parquet(WORK / f"v2tr5_fold{i}.parquet")
    cols = [c for c in F.columns if c not in DROP]
    X0 = F[cols].to_numpy("float32")
    fit = trn & ~va & ~uns
    DATA[i] = (X0, np.hstack([X0, TF.to_numpy("float32")]), fit, va, uns)


def train_pred(params, X, fit, va):
    m = lgb.train(params, lgb.Dataset(X[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(X[va], num_iteration=m.best_iteration)


BASE_PRED = {i: train_pred(PARAMS, DATA[i][0], DATA[i][2], DATA[i][3]) for i in DATA}


def hybrid_ap(params, folds):
    out = []
    for i in folds:
        X0, Xh, fit, va, uns = DATA[i]
        ph = train_pred(params, Xh, fit, va)
        p = np.where(uns[va], BASE_PRED[i], ph)
        out.append(ap(y[va], p))
    return float(np.mean(out))


def objective(trial):
    p = {**PARAMS,
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
         "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
         "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
         "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 50, log=True),
         "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255])}
    return hybrid_ap(p, [1, 2])


if __name__ == "__main__":
    default_12 = hybrid_ap(PARAMS, [1, 2])
    print(f"[default] hybrid AP folds 1-2: {default_12:.4f}", flush=True)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({"learning_rate": PARAMS["learning_rate"], "num_leaves": PARAMS["num_leaves"],
                         "min_data_in_leaf": PARAMS["min_data_in_leaf"], "feature_fraction": PARAMS["feature_fraction"],
                         "bagging_fraction": PARAMS["bagging_fraction"], "lambda_l2": PARAMS["lambda_l2"],
                         "max_bin": PARAMS["max_bin"]})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=30, callbacks=[lambda s, t: print(f"trial {t.number} {t.value:.4f} best {s.best_value:.4f}", flush=True)])
    print("[best params]", study.best_params)
    best = {**PARAMS, **study.best_params}
    d03, b03 = hybrid_ap(PARAMS, [0, 3]), hybrid_ap(best, [0, 3])
    d4, b4 = (d03 + default_12) / 2, (b03 + study.best_value) / 2
    print(f"[held-out folds 0,3 never used by tuning] hybrid AP default {d03:.4f} | tuned {b03:.4f} | gain {b03 - d03:+.4f}")
    print(f"[4-fold] hybrid AP default {d4:.4f} | tuned {b4:.4f} | gain {b4 - d4:+.4f}")
    pd.Series(study.best_params).to_json(WORK / "25_best_params.json")
