"""51 - Does extra_trees also help the HISTORY LightGBM (transition features, Optuna params of V4) that scores the
seen HUCs (74% of test rows)? 48/50 only tested the plain model. Seen HUCs keep their own 14-year history, so the
comparison is made on seen rows only, leave-one-year-out, with the transition features cached in 43."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import WORK
from cvkit import PARAMS, UNS, base_cols, fit_predict, t, test_like, views, y, year

HIST = {**PARAMS, "learning_rate": 0.0469, "num_leaves": 187, "min_data_in_leaf": 39, "feature_fraction": 0.316,
        "bagging_fraction": 0.539, "lambda_l2": 4.21}
TV, VV = views("v7")
TF = pd.read_parquet(WORK / "43_trans.parquet")
TVh, VVh = pd.concat([TV, TF.loc[TV.index]], axis=1), pd.concat([VV, TF], axis=1)
cols = base_cols(TV) + list(TF.columns)
folds = [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]
res = {}
for name, extra in [("history model", False), ("history model + extra trees", True)]:
    oof = np.full(len(t), np.nan)
    for fit, va in folds:
        oof[va & ~UNS] = fit_predict(TVh, VVh, fit, va & ~UNS, cols, params={**HIST, "extra_trees": extra}, rounds=200)
    res[name] = oof
    print(f"  done {name}", flush=True)
b, e = res["history model"], res["history model + extra trees"]
d = np.array([ap(y[va & ~UNS], e[va & ~UNS]) - ap(y[va & ~UNS], b[va & ~UNS]) for _, va in folds])
tl_b, tl_e = test_like(b, n=600), test_like(e, n=600)
print(f"[51] seen HUCs, history model: extra trees gain {d.mean():+.4f} per fold, up in {(d > 0).sum()}/14 | "
      f"test-like {tl_b.mean():.4f} -> {tl_e.mean():.4f} ({(tl_e - tl_b).mean():+.4f}, {(tl_e > tl_b).mean():.0%} draws up)")
