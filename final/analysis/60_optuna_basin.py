"""60 - Honest hyper-parameter tuning of the extra-trees LightGBM on the harness that matches the test
(leave-one-year-out + whole-basin holdout). Optuna tunes on validation years 0-6 only; the chosen parameters are then
judged against the V9 defaults on years 7-13, which the search never saw (the V4 tuning used the old random-HUC
harness)."""
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import WORK
from cvkit import HOLDOUTS, PARAMS, base_cols, fit_predict, t, test_like, views, y, year

optuna.logging.set_verbosity(optuna.logging.WARNING)
uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
BASE = {**PARAMS, "extra_trees": True}
fold = lambda k: ((t < 12 * k - 12) | (t >= 12 * k + 24), year == k)
TUNE, HOLD = range(0, 7), range(7, 14)


def score(params, rounds, years):
    oof = np.full(len(t), np.nan)
    aps = []
    for k in years:
        fit, va = fold(k)
        oof[va] = fit_predict(TV, VV, fit, va, cols, params=params, rounds=rounds, uns=uns)
        aps.append(ap(y[va], oof[va]))
    return np.array(aps), oof


def objective(trial):
    p = {**BASE, "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
         "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
         "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 50, log=True),
         "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True)}
    rounds = int(15 / p["learning_rate"])
    return score(p, rounds, TUNE)[0].mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30)
best = {**BASE, **study.best_params}
rounds = int(15 / best["learning_rate"])
print(f"[60] best on tuning years {study.best_value:.4f}; params {study.best_params}, rounds {rounds}")
b_tune, _ = score(BASE, 300, TUNE)
print(f"[60] default on tuning years {b_tune.mean():.4f} -> tuned {study.best_value:.4f} (optimistic: selected here)")
b_hold, ob = score(BASE, 300, HOLD)
t_hold, ot = score(best, rounds, HOLD)
d = t_hold - b_hold
mask = np.isin(year, list(HOLD))
tl_b, tl_t = test_like(ob, n=600, mask=mask), test_like(ot, n=600, mask=mask)
print(f"[60] UNSEEN years 7-13: default {b_hold.mean():.4f} -> tuned {t_hold.mean():.4f} ({d.mean():+.4f}, up {(d > 0).sum()}/7) | "
      f"test-like {tl_b.mean():.4f} -> {tl_t.mean():.4f} ({(tl_t - tl_b).mean():+.4f}, {(tl_t > tl_b).mean():.0%} draws up)")
pd.Series(study.best_params).to_json(WORK / "60_best_params.json")
