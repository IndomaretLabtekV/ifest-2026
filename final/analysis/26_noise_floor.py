"""26 - Noise floor of the pipeline and of the score itself.
Three numbers decide whether any further work is measurable:
  (a) seed noise: same data, same params, different LightGBM seeds -> spread of CV AP;
  (b) seed bagging: does averaging S seeds beat a single seed by more than that spread?
  (c) evaluation noise: bootstrap of a 4-origin test-like draw, and the PAIRED difference between two
      models on the same rows (the relevant quantity when comparing two public scores).
"""
import importlib

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, y, tr = h.FOLDS, h.DROP, h.PARAMS, h.y, h.tr
HIST = {**PARAMS, "learning_rate": 0.0469, "num_leaves": 187, "min_data_in_leaf": 39,
        "feature_fraction": 0.316, "bagging_fraction": 0.539, "lambda_l2": 4.21, "max_bin": 255}
SEEDS = [42, 1, 7, 99, 2024]

P = {s: [] for s in SEEDS}
BASE = []
for i, (trn, va, uns) in enumerate(FOLDS):
    F = h.fold_features(i)
    TF = pd.read_parquet(WORK / f"v2tr5_fold{i}.parquet")
    cols = [c for c in F.columns if c not in DROP]
    X0 = F[cols].to_numpy("float32")
    Xh = np.hstack([X0, TF.to_numpy("float32")])
    fit = trn & ~va & ~uns
    mb = lgb.train(PARAMS, lgb.Dataset(X0[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(X0[va], y[va])],
                   callbacks=[lgb.early_stopping(200, verbose=False)])
    pb = mb.predict(X0[va], num_iteration=mb.best_iteration)
    BASE.append(pb)
    for s in SEEDS:
        p = {**HIST, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}
        m = lgb.train(p, lgb.Dataset(Xh[fit], y[fit]), 5000, valid_sets=[lgb.Dataset(Xh[va], y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        P[s].append(np.where(uns[va], pb, m.predict(Xh[va], num_iteration=m.best_iteration)))
    print(f"fold {i} done", flush=True)

per_seed = np.array([[ap(y[va], P[s][i]) for i, (_, va, _) in enumerate(FOLDS)] for s in SEEDS])
means = per_seed.mean(1)
print(f"\n[a] CV AP per seed: {np.round(means, 4).tolist()}")
print(f"[a] seed noise: mean {means.mean():.4f}, std {means.std(ddof=1):.4f}, range {means.max() - means.min():.4f}")

rank = lambda p: rankdata(p) / len(p)
bag = [np.mean([rank(P[s][i]) for s in SEEDS], 0) for i in range(len(FOLDS))]
bag_ap = np.mean([ap(y[va], bag[i]) for i, (_, va, _) in enumerate(FOLDS)])
print(f"[b] 5-seed rank average: {bag_ap:.4f}  (best single seed {means.max():.4f}, mean single {means.mean():.4f},"
      f" gain over mean single {bag_ap - means.mean():+.4f})")

# (c) evaluation noise on a test-like 4-origin draw, and paired difference between two models
rng = np.random.default_rng(0)
draws, paired = [], []
for i, (trn, va, uns) in enumerate(FOLDS):
    d = pd.DataFrame({"o": tr.origin_id.values[va], "m": tr.month.values[va], "y": y[va],
                      "p1": bag[i], "p2": rank(P[42][i])})
    by_m = [d.o[d.m == mm].unique() for mm in (2, 4, 10, 12)]
    for _ in range(200):
        s = d[d.o.isin([rng.choice(o) for o in by_m if len(o)])]
        if s.y.nunique() < 2:
            continue
        draws.append(ap(s.y, s.p1))
        paired.append(ap(s.y, s.p1) - ap(s.y, s.p2))
draws, paired = np.array(draws), np.array(paired)
print(f"[c] single 4-origin draw: mean {draws.mean():.4f}, std {draws.std():.4f}, p10-p90 "
      f"{np.percentile(draws, 10):.3f}-{np.percentile(draws, 90):.3f}")
print(f"[c] paired difference (same rows, two models): mean {paired.mean():+.4f}, std {paired.std():.4f}, "
      f"|diff| > 0.01 in {np.mean(np.abs(paired) > 0.01):.1%} of draws")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].bar(range(len(SEEDS)), means, color="#4C72B0"); ax[0].set_xticks(range(len(SEEDS)))
ax[0].set_xticklabels(SEEDS); ax[0].set_ylim(means.min() - .01, means.max() + .01)
ax[0].axhline(bag_ap, c="#C44E52", ls="--", label="5-seed average"); ax[0].legend(); ax[0].set_title("CV AP per seed")
ax[1].hist(draws, bins=40, color="#55A868"); ax[1].set_title("AP of one 4-origin test-like draw")
plt.tight_layout(); plt.savefig(FIG / "26_noise_floor.png", dpi=100); plt.close()
