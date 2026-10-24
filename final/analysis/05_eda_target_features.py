"""05 - EDA on target + engineered features.
Target behaviour (imbalance, seasonality, persistence, HUC heterogeneity), univariate AP/AUC of
every feature, clustered correlation matrix, adversarial validation train vs test."""
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.model_selection import cross_val_predict

from common import FIG, WORK, load_clean
from fe import build, clim_table, network

tr, te = load_clean()
tr = tr.drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
tl = pd.read_csv(WORK / "timeline.csv").dropna(subset=["t"])
tr = tr.merge(tl[["origin_id", "t"]], on="origin_id", how="left")
both = pd.concat([tr, te], ignore_index=True)
C, net = clim_table(both), network(tr, te)
F = build(both, C, net)
Ftr, Fte = F.iloc[: len(tr)].reset_index(drop=True), F.iloc[len(tr):].reset_index(drop=True)
Ftr.assign(label=tr.label.values, t=tr.t.values, id=tr.id.values, origin_id=tr.origin_id.values).to_parquet(WORK / "feat_train.parquet")
Fte.assign(row_id=te.row_id.values, id=te.id.values, origin_id=te.origin_id.values).to_parquet(WORK / "feat_test.parquet")
y = tr.label.values
print(f"[features] {F.shape[1]} features, train {Ftr.shape} test {Fte.shape}")

# 1. target ---------------------------------------------------------------------------
print(f"\n[target] positive rate {y.mean():.4f} -> AP of a random ranker = {y.mean():.3f}")
print("[target] rate by origin month:", tr.groupby("month").label.mean().round(3).to_dict())
yr = tr.groupby(tr.t // 12).label.mean()
print("[target] rate by chain-year:", yr.round(3).to_dict())
hr = tr.groupby("id").label.mean()
print(f"[target] per-HUC positive rate: quantiles {hr.quantile([.05, .25, .5, .75, .95]).round(3).tolist()}")
cur = (Ftr.margin_anom_persist < 0).astype(int)
print(f"[persistence] P(label=1 | currently below 0.6 clim) = {y[cur == 1].mean():.3f};"
      f" P(label=1 | above) = {y[cur == 0].mean():.3f}")
print(f"[persistence] AP of the single feature -margin_anom_persist = {ap(y, -Ftr.margin_anom_persist.fillna(0)):.4f}")

# 2. univariate ranking ------------------------------------------------------------------
rows = []
for c in Ftr.columns:
    v = Ftr[c].fillna(Ftr[c].median())
    a = auc(y, v)
    s = -v if a < 0.5 else v
    rows.append((c, max(a, 1 - a), ap(y, s), Ftr[c].isna().mean(), Fte[c].isna().mean()))
uni = pd.DataFrame(rows, columns=["feature", "AUC", "AP", "nan_train", "nan_test"]).sort_values("AP", ascending=False)
print("\n[univariate] top 20 by AP:"); print(uni.head(20).round(4).to_string(index=False))
print("[univariate] bottom 8:"); print(uni.tail(8).round(4).to_string(index=False))

# 3. clustered correlation matrix -------------------------------------------------------
top = uni.feature.head(35).tolist()
corr = Ftr[top].corr(method="spearman").fillna(0)
order = hierarchy.leaves_list(hierarchy.linkage(1 - corr.abs(), "average"))
corr = corr.iloc[order, order]
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()
print("\n[corr] most redundant pairs (|spearman|>0.95):")
print(pairs[pairs.abs() > 0.95].sort_values(ascending=False).round(3).head(12).to_string())
plt.figure(figsize=(14, 12)); sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.title("Spearman correlation, top-35 features (clustered)"); plt.tight_layout()
plt.savefig(FIG / "05_corr_matrix.png", dpi=100); plt.close()

# 4. adversarial validation -----------------------------------------------------------------
X = pd.concat([Ftr.sample(60000, random_state=0), Fte]).drop(columns=["month"])
z = np.r_[np.zeros(60000), np.ones(len(Fte))]
pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0),
                       X, z, cv=5, method="predict_proba")[:, 1]
print(f"\n[adversarial] train-vs-test AUC = {auc(z, pa):.3f}")
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0).fit(X, z)
imp = pd.Series(m.booster_.feature_importance("gain"), X.columns).sort_values(ascending=False)
print("[adversarial] top drifting features:", (imp / imp.sum()).head(8).round(3).to_dict())
# origin-level features take only 4 distinct values in test and month only 4 -> trivially separable.
# Re-run without them, on train rows from the same 4 calendar months, to measure real covariate drift.
drop = ["month", "m_sin", "m_cos", "origin_anom0", "origin_below0"]
Xs = pd.concat([Ftr[tr.month.isin(te.month.unique()).values].sample(60000, random_state=0), Fte]).drop(columns=drop)
pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0),
                       Xs, z, cv=5, method="predict_proba")[:, 1]
print(f"[adversarial] without origin/month features, same months: AUC = {auc(z, pa):.3f}")
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0).fit(Xs, z)
imp = pd.Series(m.booster_.feature_importance("gain"), Xs.columns).sort_values(ascending=False)
print("[adversarial] top drifting features:", (imp / imp.sum()).head(8).round(3).to_dict())
# third run: row-level features only (no cross-sectional basin/network aggregates)
xsec = [c for c in Xs.columns if c.startswith(("basin_", "down_", "up_anom", "anom0_vs"))]
pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0),
                       Xs.drop(columns=xsec), z, cv=5, method="predict_proba")[:, 1]
print(f"[adversarial] row-level features only: AUC = {auc(z, pa):.3f}")
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0).fit(Xs.drop(columns=xsec), z)
imp = pd.Series(m.booster_.feature_importance("gain"), Xs.drop(columns=xsec).columns).sort_values(ascending=False)
print("[adversarial] top drifting features:", (imp / imp.sum()).head(8).round(3).to_dict())

# 5. figures -----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
tr.groupby("month").label.mean().plot.bar(ax=ax[0, 0], color="#4C72B0"); ax[0, 0].set_title("Positive rate by origin month")
yr.plot(ax=ax[0, 1], marker="o"); ax[0, 1].set_title("Positive rate by year of recovered timeline")
b = pd.cut(Ftr.margin_anom_persist, np.arange(-4, 4.25, .25))
tr.groupby(b, observed=True).label.mean().plot(ax=ax[1, 0], marker=".")
ax[1, 0].axvline(15.5, ls="--", c="k"); ax[1, 0].set_title("P(label) vs current anomaly margin (persistence)")
ax[1, 0].tick_params(axis="x", rotation=90)
uni.head(20).set_index("feature").AP.sort_values().plot.barh(ax=ax[1, 1], color="#55A868")
ax[1, 1].axvline(y.mean(), ls="--", c="k"); ax[1, 1].set_title("Univariate AP (top 20)")
plt.tight_layout(); plt.savefig(FIG / "05_target_eda.png", dpi=100); plt.close()

fig, ax = plt.subplots(2, 4, figsize=(18, 7))
for a, c in zip(ax.ravel(), uni.feature.head(8)):
    lo, hi = Ftr[c].quantile([.01, .99])
    for lab, col in [(0, "#4C72B0"), (1, "#C44E52")]:
        a.hist(Ftr.loc[y == lab, c].clip(lo, hi), bins=60, alpha=.55, density=True, color=col, label=f"y={lab}")
    a.set_title(c); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "05_top_feature_dists.png", dpi=100); plt.close()
