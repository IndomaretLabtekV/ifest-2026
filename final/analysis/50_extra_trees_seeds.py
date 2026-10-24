"""50 - Is the extra_trees gain of 48 (+0.0035 test-like, new basins up in 10/14 folds) real or one lucky seed?
Three seeds for both the base LightGBM and the extra-trees LightGBM, same folds and holdout as 48, plus the
learning-rate/rounds setting the notebook actually uses (lr 0.03, more rounds) because extra trees need more
trees to reach the same fit."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import WORK
from cvkit import HOLDOUTS, PARAMS, base_cols, fit_predict, t, test_like, views, y, year

uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
folds = [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]
path = WORK / "50_oof.parquet"
if path.exists():
    O = pd.read_parquet(path)
else:
    O = {}
    for seed in (1, 7, 2024):
        for name, extra, rounds, lr in [("base", False, 300, .05), ("extra", True, 300, .05), ("extra lr.03 x600", True, 600, .03)]:
            p = {**PARAMS, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "extra_seed": seed,
                 "extra_trees": extra, "learning_rate": lr}
            oof = np.full(len(t), np.nan)
            for fit, va in folds:
                oof[va] = fit_predict(TV, VV, fit, va, cols, params=p, rounds=rounds, uns=uns)
            O[f"{name} | seed {seed}"] = oof
            print(f"  done {name} seed {seed}", flush=True)
    O = pd.DataFrame(O)
    O.to_parquet(path)

rows = []
for seed in (1, 7, 2024):
    b = O[f"base | seed {seed}"].to_numpy()
    tl_b = test_like(b, n=600, seed=seed)
    for name in ("extra", "extra lr.03 x600"):
        p = O[f"{name} | seed {seed}"].to_numpy()
        d_new = np.array([ap(y[va & uns], p[va & uns]) - ap(y[va & uns], b[va & uns]) for _, va in folds])
        d_all = np.array([ap(y[va], p[va]) - ap(y[va], b[va]) for _, va in folds])
        tl = test_like(p, n=600, seed=seed)
        rows.append(dict(seed=seed, variant=name, fold_gain=d_all.mean(), folds_up=(d_all > 0).sum(),
                         new_gain=d_new.mean(), new_up=(d_new > 0).sum(), tl_gain=(tl - tl_b).mean(),
                         draws_up=(tl > tl_b).mean()))
R = pd.DataFrame(rows)
print(R.round(4).to_string(index=False))
print("\n[mean over seeds]")
print(R.groupby("variant")[["fold_gain", "new_gain", "tl_gain", "draws_up"]].mean().round(4).to_string())
