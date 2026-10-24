"""48 - Generalising to NEW basins (26% of test rows are 47 unseen basins, 44). Theory says four remedies:
  importance weighting  covariate-shift correction (Shimodaira 2000): weight train rows by p(new)/p(train) from an
                        adversarial classifier on static/climatology features of the held-out basins (labels unused)
  monotone constraints  wetter now -> never more risk; constrained trees extrapolate more safely
  strong regularisation fewer, larger leaves
  physics logistic      a 9-feature logistic model on the threshold margins, rank-blended 80/20 with the trees
All judged with leave-one-year-out + whole-basin holdout (the harness that agreed with the public in 44)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from common import FIG, WORK
from cvkit import HOLDOUTS, PARAMS, base_cols, fit_predict, t, test_like, tr, views, y, year

uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
folds = [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]
STATIC = ["log_inc_area", "log_cum_area", "headwater", "log_pop_dens", "pop_growth", "clim_cv", "seas_step"]

# importance weights: which fit HUCs look like the held-out basins (static features only, no labels)
huc = pd.DataFrame(VV[STATIC].to_numpy(), columns=STATIC)
huc["id"], huc["new"] = tr.id.values, uns
H = huc.groupby("id").median()
p_new = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, num_leaves=15, verbose=-1, random_state=0),
                          H[STATIC], H.new.astype(int), cv=5, method="predict_proba")[:, 1]
odds = pd.Series(np.clip(p_new / (1 - p_new) * (1 - H.new.mean()) / H.new.mean(), 0.2, 5.0), H.index)
W = tr.id.map(odds).to_numpy()
print(f"[weights] adversarial AUC held-out basins vs fit HUCs on static features: "
      f"{roc_auc_score(H.new, p_new):.3f}; "
      f"weight quantiles {np.round(np.quantile(W[~uns], [.05, .5, .95]), 2).tolist()}")

DEC = ["anom0", "anom_mean3", "anom_mean6", "margin_anom_persist", "margin_raw_persist", "margin_dry", "basin_anom0",
       "down_anom0", "up_anom0"]
INC = ["n_below12", "drought_run", "basin_below0"]
mono = [(-1 if c in DEC else 1 if c in INC else 0) for c in cols]
CONFIGS = {
    "base (V8-like)": {},
    "importance weights": dict(weight=W),
    "monotone": dict(params={**PARAMS, "monotone_constraints": mono, "monotone_constraints_method": "advanced"}),
    "strong regularisation": dict(params={**PARAMS, "num_leaves": 31, "min_data_in_leaf": 1000, "lambda_l2": 10.0,
                                          "feature_fraction": 0.5}),
    "extra trees": dict(params={**PARAMS, "extra_trees": True}),
}
LOGI = ["margin_anom_persist", "margin_raw_persist", "anom_mean3", "margin_dry", "seas_step", "basin_anom0",
        "drought_run", "m_sin", "m_cos"]

path = WORK / "48_oof.parquet"
if path.exists():
    O = pd.read_parquet(path)
else:
    O = {}
    for name, kw in CONFIGS.items():
        oof = np.full(len(t), np.nan)
        for fit, va in folds:
            oof[va] = fit_predict(TV, VV, fit, va, cols, uns=uns, **kw)
        O[name] = oof
        print(f"  done {name}", flush=True)
    oof = np.full(len(t), np.nan)
    for fit, va in folds:
        f = np.flatnonzero(fit & ~uns)
        v = np.flatnonzero(va)
        Xf = TV.loc[f, LOGI].fillna(0).to_numpy()
        sc = StandardScaler().fit(Xf)
        m = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(Xf), y[f])
        oof[va] = m.predict_proba(sc.transform(VV.loc[v, LOGI].fillna(0).to_numpy()))[:, 1]
    O["physics logistic"] = oof
    O = pd.DataFrame(O)
    O.to_parquet(path)


# probability average, not per-fold ranks: per-fold ranks would erase the cross-origin level that pooled AP uses
O["base + 20% logistic"] = 0.8 * O["base (V8-like)"] + 0.2 * O["physics logistic"]
base = O["base (V8-like)"].to_numpy()
tl_b = test_like(base, n=800)
rows = []
for name in O.columns:
    p = O[name].to_numpy()
    per = np.array([(ap(y[va], p[va]), ap(y[va & uns], p[va & uns]), ap(y[va & ~uns], p[va & ~uns]),
                     ap(y[va & uns], base[va & uns])) for _, va in folds])
    tl = test_like(p, n=800)
    rows.append(dict(model=name, AP=per[:, 0].mean(), new_basins=per[:, 1].mean(), seen=per[:, 2].mean(),
                     new_folds_up=f"{(per[:, 1] > per[:, 3]).sum()}/14", test_like=tl.mean(),
                     tl_gain=(tl - tl_b).mean(), draws_up=(tl > tl_b).mean()))
R = pd.DataFrame(rows).set_index("model")
print("\n[48] leave-one-year-out, whole-basin holdout:")
print(R.round(4).to_string())

fig, ax = plt.subplots(figsize=(10, 4.5))
(R[["new_basins", "seen"]] - R.loc["base (V8-like)", ["new_basins", "seen"]]).plot.bar(ax=ax, rot=20)
ax.axhline(0, color="k", lw=.8)
ax.set_title("AP change vs V8-like base, by segment (LOYO, basin holdout)")
plt.tight_layout()
plt.savefig(FIG / "48_spatial_generalization.png", dpi=120)
print(f"[fig] {FIG / '48_spatial_generalization.png'}")
