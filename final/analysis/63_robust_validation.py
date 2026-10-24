"""63 - Robust validation: do the verdicts between the candidate final configurations survive a DIFFERENT spatial split?
Every decision since V8 was taken on ONE whole-basin holdout (seed 42, 25 basins). The 47 new test basins are a
different draw, so here the whole leave-one-year-out + basin-holdout harness is repeated with three basin splits
(seeds 42, 1, 7) and the members of every candidate final blend (V8, V13, V14 style) are re-judged per year and per
split. A verdict we can trust must hold in most years AND in all three spatial splits."""
import sys

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import HOLDOUTS, PARAMS, base_cols, basin_holdout, t, test_like, tr, views, y, year

SEEDS = [int(s) for s in sys.argv[1:]] or [42, 1, 7]
for s in SEEDS:
    if s != 42:
        HOLDOUTS[f"basin{s}"] = basin_holdout(seed=s)
hname = lambda s: "basin" if s == 42 else f"basin{s}"

V8 = {**PARAMS, "learning_rate": 0.03}
V9 = {**V8, "extra_trees": True}
V13 = {**PARAMS, "learning_rate": 0.0273, "num_leaves": 53, "min_data_in_leaf": 91, "feature_fraction": 0.328,
       "bagging_fraction": 0.822, "lambda_l2": 9.27, "extra_trees": True}
DART = {**PARAMS, "boosting": "dart", "drop_rate": 0.1, "learning_rate": 0.1, "feature_fraction": 0.5}
NAMES = ["lgb V8", "lgb V9 extra", "lgb V13 tuned", "lgb dart", "xgboost", "catboost"]
folds = [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]


def member(name, Xf, yf, Xv):
    if name == "lgb V8":
        return lgb.train(V8, lgb.Dataset(Xf, yf), 500).predict(Xv)
    if name == "lgb V9 extra":
        return lgb.train(V9, lgb.Dataset(Xf, yf), 500).predict(Xv)
    if name == "lgb V13 tuned":
        return lgb.train(V13, lgb.Dataset(Xf, yf), 549).predict(Xv)
    if name == "lgb dart":
        return lgb.train(DART, lgb.Dataset(Xf, yf), 300).predict(Xv)
    if name == "xgboost":
        m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, min_child_weight=20, subsample=0.8,
                              colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist", n_jobs=20, random_state=42)
        return m.fit(Xf, yf).predict_proba(Xv)[:, 1]
    if name == "catboost":
        m = cb.CatBoostClassifier(iterations=400, learning_rate=0.1, depth=8, l2_leaf_reg=5, random_seed=42,
                                  thread_count=20, verbose=False, allow_writing_files=False)
        return m.fit(Xf, yf).predict_proba(Xv)[:, 1]
    raise ValueError(name)


def oof_for(seed):
    uns = HOLDOUTS[hname(seed)]
    path = WORK / f"63_oof_s{seed}.parquet"
    O = pd.read_parquet(path) if path.exists() else pd.DataFrame(index=np.arange(len(tr)))
    todo = [n for n in NAMES if n not in O]
    if not todo:
        return O, uns
    TV, VV = views("v7", hname(seed))
    cols = base_cols(TV)
    for n in todo:
        oof = np.full(len(tr), np.nan)
        for fit, va in folds:
            f, v = np.flatnonzero(fit & ~uns), np.flatnonzero(va)
            oof[va] = member(n, TV.loc[f, cols].to_numpy("float32"), y[f], VV.loc[v, cols].to_numpy("float32"))
        O[n] = oof
        O.to_parquet(path)
        print(f"  seed {seed}: done {n}", flush=True)
    return O, uns


BLENDS = {"V8 final: lgbV8+xgb+cat": ["lgb V8", "xgboost", "catboost"],
          "V13 final: tuned+xgb+cat": ["lgb V13 tuned", "xgboost", "catboost"],
          "V14 final: tuned+dart": ["lgb V13 tuned", "lgb dart"],
          "tuned+dart+xgb+cat": ["lgb V13 tuned", "lgb dart", "xgboost", "catboost"],
          "lgbV8+tuned+dart": ["lgb V8", "lgb V13 tuned", "lgb dart"],
          "all 6": NAMES}
REF = "V8 final: lgbV8+xgb+cat"


def ranked(O):
    R = O.copy()
    for _, va in folds:
        R.loc[va, :] = O.loc[va].rank(pct=True).values
    for b, m in BLENDS.items():
        R[b] = R[m].mean(1)
    return R


def yearly(p, mask=None):
    mask = np.ones(len(y), bool) if mask is None else mask
    return np.array([ap(y[(year == k) & mask], p[(year == k) & mask]) for k in range(14)])


rows, per_year = [], {}
for s in SEEDS:
    O, uns = oof_for(s)
    R = ranked(O)
    ref, ref_new = yearly(R[REF].to_numpy()), yearly(R[REF].to_numpy(), uns)
    tl_ref = test_like(R[REF].to_numpy(), n=500)
    for c in NAMES + list(BLENDS):
        p = R[c].to_numpy()
        a, an = yearly(p), yearly(p, uns)
        tl = test_like(p, n=500)
        per_year[(s, c)] = a
        rows.append(dict(seed=s, config=c, ap=a.mean(), d_vs_V8=(a - ref).mean(), up=int((a > ref).sum()),
                         ap_newbasin=an.mean(), d_new=(an - ref_new).mean(), testlike=tl.mean(),
                         tl_up=(tl > tl_ref).mean()))
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n[63] per spatial split, per-year mean AP, gain vs the V8 final blend (up = years out of 14), new-basin rows, "
      "test-like 4-origin draws:")
print(res.round(4).to_string(index=False))
piv = res.pivot(index="config", columns="seed", values="d_vs_V8")
piv["mean"] = piv.mean(1)
piv["splits_up"] = (res.pivot(index="config", columns="seed", values="d_vs_V8") > 0).sum(1)
print("\n[63] gain vs V8 final blend, by spatial split:")
print(piv.sort_values("mean", ascending=False).round(4).to_string())
res.to_csv(WORK / "63_robust_validation.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
P = piv.drop(columns=["mean", "splits_up"]).sort_values(SEEDS[0])
im = ax[0].imshow(P.to_numpy() * 1000, cmap="RdBu", vmin=-8, vmax=8, aspect="auto")
ax[0].set_yticks(range(len(P)), P.index, fontsize=8)
ax[0].set_xticks(range(len(P.columns)), [f"basin seed {c}" for c in P.columns])
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        ax[0].text(j, i, f"{P.iat[i, j] * 1000:+.1f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax[0], label="AP gain vs V8 final (x1000)")
ax[0].set_title("Verdicts across three whole-basin splits")
for c in [REF, "V14 final: tuned+dart", "tuned+dart+xgb+cat", "lgb V13 tuned"]:
    ax[1].plot(range(14), per_year[(SEEDS[0], c)], marker="o", label=c)
ax[1].set_xlabel("validation year (0 = Sep 2000-Aug 2001)")
ax[1].set_ylabel("AP")
ax[1].set_title(f"Per-year AP (basin seed {SEEDS[0]}): the year dominates, configs move together")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "63_robust_validation.png", dpi=120)
print(f"[fig] {FIG / '63_robust_validation.png'}")
