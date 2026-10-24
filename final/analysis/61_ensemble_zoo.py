"""61 - Ensemble zoo, all models trained from scratch (rules 2-3), judged on the harness that agrees with the public
(leave-one-year-out + whole-basin holdout + rescaled test view, 44).
Members: tuned LightGBM + extra trees (V13 base), LightGBM default, LightGBM DART, XGBoost, CatBoost, sklearn
HistGradientBoosting, sklearn ExtraTrees, LightGBM LambdaRank grouped by origin, LightGBM regression on the continuous
target z = log(ws_next / clim_next) (score = -z), and the base LightGBM on the v6-repair view (preprocessing diversity).
Ensembles are chosen by greedy forward selection with replacement (Caruana 2004) on rank-averaged OOF, NESTED: selected
on years 0-6 and judged on 7-13, and the other way round, so the reported gain never saw its own selection years."""
import json

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import HOLDOUTS, PARAMS, REPAIRS, base_cols, t, test_like, tr, views, y, year

uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
TV6, VV6 = views("v6", "basin")
cols = base_cols(TV)
best = json.loads((WORK / "60_best_params.json").read_text())
best = {k: int(v) if k in ("num_leaves", "min_data_in_leaf") else v for k, v in best.items()}
TUNED = {**PARAMS, **best, "extra_trees": True}
R_TUNED = int(15 / best["learning_rate"])
folds = [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]

# continuous target: z = log(ws(t+1) / clim(t+1)), ws(t+1) = repaired lag0 of the same HUC one origin later
rep = REPAIRS["v7"](tr)
nxt = pd.Series(rep.q_ws_cum_m3_lag0.abs().values, index=pd.MultiIndex.from_arrays([tr.id.values, tr.t.values]))
ws_next = nxt.reindex(pd.MultiIndex.from_arrays([tr.id.values, tr.t.values + 1])).to_numpy()
clim_next = rep.q_ws_cum_m3_lag0.abs().values / np.exp(TV.margin_raw_persist.reindex(np.arange(len(tr))).values + np.log(0.6))
z = np.log((ws_next + 1) / (clim_next + 1))


def X(view, idx, c=cols):
    return view.loc[idx, c].to_numpy("float32")


def member(name, fit, va):
    f, v = np.flatnonzero(fit & ~uns), np.flatnonzero(va)
    Xf, Xv, yf = X(TV, f), X(VV, v), y[f]
    if name == "lgb tuned+extra (V13)":
        return lgb.train(TUNED, lgb.Dataset(Xf, yf), R_TUNED).predict(Xv)
    if name == "lgb default":
        return lgb.train(PARAMS, lgb.Dataset(Xf, yf), 300).predict(Xv)
    if name == "lgb dart":
        return lgb.train({**PARAMS, "boosting": "dart", "drop_rate": 0.1, "learning_rate": 0.1}, lgb.Dataset(Xf, yf), 300).predict(Xv)
    if name == "lgb v6 repair":
        return lgb.train(TUNED, lgb.Dataset(X(TV6, f), yf), R_TUNED).predict(X(VV6, v))
    if name == "xgboost":
        m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7, min_child_weight=20, subsample=0.8,
                              colsample_bytree=0.6, reg_lambda=5.0, tree_method="hist", n_jobs=20, random_state=42)
        return m.fit(Xf, yf).predict_proba(Xv)[:, 1]
    if name == "catboost":
        m = cb.CatBoostClassifier(iterations=500, learning_rate=0.08, depth=8, l2_leaf_reg=5, random_seed=42,
                                  thread_count=20, verbose=False, allow_writing_files=False)
        return m.fit(Xf, yf).predict_proba(Xv)[:, 1]
    if name == "sklearn hist gbdt":
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
                                           l2_regularization=1.0, random_state=42)
        return m.fit(Xf, yf).predict_proba(Xv)[:, 1]
    if name == "extra trees (bagging)":
        med = np.nanmedian(Xf, 0)
        fill = lambda A: np.where(np.isfinite(A), A, med)
        m = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=50, max_features=0.5, n_jobs=20, random_state=42)
        return m.fit(fill(Xf), yf).predict_proba(fill(Xv))[:, 1]
    if name == "lgb lambdarank":
        order = np.argsort(t[f], kind="stable")
        grp = np.bincount(pd.Series(t[f][order]).astype("category").cat.codes)
        m = lgb.train({**PARAMS, "objective": "lambdarank", "lambdarank_truncation_level": 2000, "metric": "map"},
                      lgb.Dataset(Xf[order], yf[order], group=grp), 300)
        return m.predict(Xv)
    if name == "lgb regression on z":
        ok = np.isfinite(z[f])
        m = lgb.train({**PARAMS, "objective": "regression"}, lgb.Dataset(Xf[ok], z[f][ok]), 300)
        return -m.predict(Xv)
    raise ValueError(name)


NAMES = ["lgb tuned+extra (V13)", "lgb default", "lgb dart", "lgb v6 repair", "xgboost", "catboost",
         "sklearn hist gbdt", "extra trees (bagging)", "lgb lambdarank", "lgb regression on z"]
path = WORK / "61_oof.parquet"
O = pd.read_parquet(path) if path.exists() else pd.DataFrame(index=np.arange(len(tr)))
for n in NAMES:
    if n in O:
        continue
    oof = np.full(len(tr), np.nan)
    for fit, va in folds:
        oof[va] = member(n, fit, va)
    O[n] = oof
    O.to_parquet(path)
    print(f"  done {n}", flush=True)

# per-fold rank (like ranking one test set), then the scoring helpers
Rk = O.copy()
for _, va in folds:
    Rk.loc[va, :] = O.loc[va].rank(pct=True).values


def yearly(p, years):
    return np.array([ap(y[year == k], p[year == k]) for k in years])


base = Rk["lgb tuned+extra (V13)"].to_numpy()
print("\n[61] single members (per-fold rank), mean AP over 14 years, and correlation with the V13 base:")
for n in NAMES:
    a = yearly(Rk[n].to_numpy(), range(14))
    d = a - yearly(base, range(14))
    print(f"  {n:24s} AP {a.mean():.4f} ({d.mean():+.4f} vs base, up {(d > 0).sum()}/14) | "
          f"spearman with base {pd.Series(Rk[n]).corr(pd.Series(base), method='spearman'):.3f}")


def greedy(years, steps=15):
    """Caruana forward selection with replacement on the mean yearly AP of the chosen years."""
    chosen = ["lgb tuned+extra (V13)"]
    for _ in range(steps):
        cur = Rk[chosen].mean(1).to_numpy()
        best_n, best_v = None, yearly(cur, years).mean()
        for n in NAMES:
            v = yearly(Rk[chosen + [n]].mean(1).to_numpy(), years).mean()
            if v > best_v + 1e-5:
                best_n, best_v = n, v
        if best_n is None:
            break
        chosen.append(best_n)
    return chosen


print("\n[61] nested greedy ensembles (select on one half of the years, judge on the other):")
res = {}
for sel, held in [(range(0, 7), range(7, 14)), (range(7, 14), range(0, 7))]:
    ch = greedy(sel)
    ens = Rk[ch].mean(1).to_numpy()
    d = yearly(ens, held) - yearly(base, held)
    w = pd.Series(ch).value_counts().to_dict()
    print(f"  select {sel.start}-{sel.stop - 1} -> weights {w} | held-out years {held.start}-{held.stop - 1}: "
          f"{d.mean():+.4f} vs base (up {(d > 0).sum()}/7)")
    res[f"{sel.start}-{sel.stop - 1}"] = (ch, d)
simple = {"base + xgb + cat (V13 style)": ["lgb tuned+extra (V13)", "xgboost", "catboost"],
          "+ extra trees": ["lgb tuned+extra (V13)", "xgboost", "catboost", "extra trees (bagging)"],
          "+ v6 repair": ["lgb tuned+extra (V13)", "xgboost", "catboost", "lgb v6 repair"],
          "+ lambdarank": ["lgb tuned+extra (V13)", "xgboost", "catboost", "lgb lambdarank"],
          "all 10 equal": NAMES}
print("\n[61] fixed (not selected) blends, all 14 years, and test-like 4-origin draws:")
tl_b = test_like(base, n=600)
for n, ch in simple.items():
    ens = Rk[ch].mean(1).to_numpy()
    d = yearly(ens, range(14)) - yearly(base, range(14))
    tl = test_like(ens, n=600)
    print(f"  {n:30s} {d.mean():+.4f} (up {(d > 0).sum()}/14) | test-like {(tl - tl_b).mean():+.4f} ({(tl > tl_b).mean():.0%} draws up)")

fig, ax = plt.subplots(figsize=(10, 5))
C = Rk[NAMES].corr(method="spearman")
im = ax.imshow(C, vmin=.8, vmax=1, cmap="viridis")
ax.set_xticks(range(len(NAMES)), NAMES, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(NAMES)), NAMES, fontsize=8)
plt.colorbar(im)
ax.set_title("Spearman correlation of OOF ranks (lower = more diverse)")
plt.tight_layout()
plt.savefig(FIG / "61_ensemble_zoo.png", dpi=120)
print(f"[fig] {FIG / '61_ensemble_zoo.png'}")
