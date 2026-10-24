"""62 - DART as a blend member for the HISTORY model (seen HUCs, 74% of test rows). 61 showed DART on the plain model
matches the tuned extra-trees LightGBM alone and improves it in a blend (+0.0026, up 12/14). Here the same check on
seen HUCs with the transition features (V4 history params + extra trees, 51): history LightGBM vs history DART vs their
rank average, leave-one-year-out."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import WORK
from cvkit import PARAMS, UNS, base_cols, fit_predict, t, test_like, views, y, year

HIST = {**PARAMS, "learning_rate": 0.0469, "num_leaves": 187, "min_data_in_leaf": 39, "feature_fraction": 0.316,
        "bagging_fraction": 0.539, "lambda_l2": 4.21, "extra_trees": True}
DART = {**PARAMS, "boosting": "dart", "drop_rate": 0.1, "learning_rate": 0.1, "feature_fraction": 0.5}
TV, VV = views("v7")
TF = pd.read_parquet(WORK / "43_trans.parquet")
TVh, VVh = pd.concat([TV, TF.loc[TV.index]], axis=1), pd.concat([VV, TF], axis=1)
cols = base_cols(TV) + list(TF.columns)
seen = ~UNS
res = {"history lgb": np.full(len(t), np.nan), "history dart": np.full(len(t), np.nan)}
for k in range(14):
    fit, va = (t < 12 * k - 12) | (t >= 12 * k + 24), (year == k) & seen
    res["history lgb"][va] = fit_predict(TVh, VVh, fit, va, cols, params=HIST, rounds=200)
    res["history dart"][va] = fit_predict(TVh, VVh, fit, va, cols, params=DART, rounds=300)
    print(f"  year {k}: lgb {ap(y[va], res['history lgb'][va]):.4f} dart {ap(y[va], res['history dart'][va]):.4f}", flush=True)
blend = np.full(len(t), np.nan)
for k in range(14):
    va = (year == k) & seen
    blend[va] = (pd.Series(res["history lgb"][va]).rank(pct=True).values + pd.Series(res["history dart"][va]).rank(pct=True).values) / 2
yearly = lambda p: np.array([ap(y[(year == k) & seen], p[(year == k) & seen]) for k in range(14)])
b = yearly(res["history lgb"])
tlb = test_like(res["history lgb"], n=600, mask=seen)
for name, p in [("history dart", res["history dart"]), ("rank blend lgb + dart", blend)]:
    d = yearly(p) - b
    tl = test_like(p, n=600, mask=seen)
    print(f"[62] {name:22s} {d.mean():+.4f} vs history lgb (up {(d > 0).sum()}/14) | test-like {(tl - tlb).mean():+.4f} "
          f"({(tl > tlb).mean():.0%} draws up)")
