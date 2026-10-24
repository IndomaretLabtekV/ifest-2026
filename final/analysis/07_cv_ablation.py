"""07 - Validation design + ablation. Every preprocessing / FE decision is judged here by AP.
CV mimics the test: (a) contiguous time block held out with a 12-month purge on both sides
(lag windows are 12 months long, so closer rows share raw values with the validation block);
(b) 25% of HUCs removed from training entirely (26% of test HUCs never appear in train)."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import LAG_COLS, NUM_COLS, WORK, load_clean
from prep import clim_scale, repair

F = pd.read_parquet(WORK / "feat_train.parquet")
y, t, ids = F.pop("label").values, F.pop("t").values, F.pop("id").values
F = F.drop(columns=["origin_id"])

tr, _ = load_clean()
tr = tr.drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
assert (tr.id.values == ids).all()
raw = tr[NUM_COLS + ["month"]].copy()
rep = repair(tr)[NUM_COLS + ["month"]]
e = 10.0 ** clim_scale(tr)
rep["clim_median_now"] *= e; rep["clim_median_next"] *= e


def folds(n_blocks=4, purge=12, huc_frac=0.25, seed=42):
    edges = np.linspace(t.min(), t.max() + 1, n_blocks + 1).astype(int)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    for i in range(n_blocks):
        lo, hi = edges[i], edges[i + 1]
        unseen = set(rng.choice(uniq, int(huc_frac * len(uniq)), replace=False))
        is_unseen = np.array([h in unseen for h in ids])
        va = (t >= lo) & (t < hi)
        trn = ((t < lo - purge) | (t >= hi + purge)) & ~is_unseen
        yield trn, va, is_unseen


FOLDS = list(folds())
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, max_depth=-1,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
              verbose=-1, seed=42, num_threads=20)


def run(name, X, rounds=2000):
    res, best = [], []
    for trn, va, uns in FOLDS:
        dtr, dva = lgb.Dataset(X[trn], y[trn]), lgb.Dataset(X[va], y[va])
        m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(X[va], num_iteration=m.best_iteration)
        res.append((ap(y[va], p), ap(y[va][~uns[va]], p[~uns[va]]), ap(y[va][uns[va]], p[uns[va]])))
        best.append(m.best_iteration)
    r = np.array(res)
    print(f"{name:38s} AP {r[:, 0].mean():.4f} ±{r[:, 0].std():.4f} | seen {r[:, 1].mean():.4f} unseen {r[:, 2].mean():.4f}"
          f" | per-fold {np.round(r[:, 0], 4).tolist()} | iters {best}", flush=True)
    return m, r


if __name__ == "__main__":
    print(f"rows {len(y)}, folds: " + ", ".join(f"trn {a.sum()} va {b.sum()}" for a, b, _ in FOLDS))
    for i, (trn, va, uns) in enumerate(FOLDS):
        p = -F.margin_anom_persist.fillna(0).values[va]
        print(f"[baseline persistence] fold {i}: AP {ap(y[va], p):.4f}  pos-rate {y[va].mean():.3f}")
    which = sys.argv[1:] or ["raw", "repaired", "fe", "fe-origin", "fe-basin", "fe+repaired"]
    origin = ["origin_anom0", "origin_below0"]
    basin = [c for c in F.columns if c.startswith(("basin_", "down_", "up_anom", "anom0_vs"))]
    if "raw" in which: run("E1 raw parsed columns", raw.to_numpy("float32"))
    if "repaired" in which: run("E2 repaired lags + clim scale fix", rep.to_numpy("float32"))
    if "fe" in which: run("E3 engineered features (all)", F.to_numpy("float32"))
    if "fe-origin" in which: run("E4 FE - origin-level", F.drop(columns=origin).to_numpy("float32"))
    if "fe-basin" in which: run("E5 FE - origin - basin/network", F.drop(columns=origin + basin).to_numpy("float32"))
    if "fe-final" in which:
        run("E4b FE - origin - basin_size", F.drop(columns=origin + ["basin_size"]).to_numpy("float32"))
    if "fe+repaired" in which:
        run("E6 FE - origin + repaired raw", np.hstack([F.drop(columns=origin).to_numpy("float32"), rep.to_numpy("float32")]))
