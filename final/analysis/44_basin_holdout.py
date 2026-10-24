"""44 - The test's unseen HUCs are whole new basins (43). Does a basin-grouped holdout change the verdicts?
Part A (EDA, test only): are the 786 test-only HUCs hydrologically different from the seen ones? Adversarial AUC
          on static / climatology / dynamics features, and the drought state at the 4 test origins.
Part B (CV, leave-one-year-out): random-HUC holdout (V2-V7 harness) vs whole-basin holdout, for the exact
          changes that separate V5, V6 and V7. The public board said V5 0.7552 > V6 0.751 > V7 0.748. A harness
          that reproduces that order is the one to trust for the next decisions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb

from common import FIG, WORK
from cvkit import DYN, HOLDOUTS, base_cols, run_scheme, t, te, test_like, test_view, tr, views, y, year

THR = np.log(0.6)
STATIC = ["log_inc_area", "log_cum_area", "headwater", "log_pop_dens", "pop_growth", "clim_cv"]

# ---------------- Part A ----------------
_, E = test_view("v7")
new = ~te.reset_index(drop=True).id.isin(set(tr.id)).values
for grp, cols_ in [("static + climatology", STATIC), ("dynamics", DYN), ("state now", ["anom0", "anom_mean3", "anom_mean12"])]:
    X = E[cols_].to_numpy("float32")
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1, random_state=0), X, new, cv=5,
                          method="predict_proba")[:, 1]
    print(f"[A] adversarial AUC test-only vs seen HUCs on {grp:22s}: {auc(new, p):.3f}")
o = te.reset_index(drop=True).origin_id.values
st = pd.DataFrame({"origin": o, "new": new, "below": E.anom0.values < THR, "clim_cv": E.clim_cv.values,
                   "ar1": E.ar1.values, "flash": E.flashiness.values})
print("[A] share below the 0.6 threshold now, per test origin (seen vs test-only HUCs):")
print(st.groupby(["origin", "new"]).below.mean().unstack().round(3).to_string())
print("[A] medians seen vs test-only: " + ", ".join(f"{c} {st[c][~new].median():.3f} vs {st[c][new].median():.3f}"
                                                   for c in ["clim_cv", "ar1", "flash"]))

# ---------------- Part B ----------------
folds = [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]
path = WORK / "44_oof.parquet"
if path.exists():
    O = pd.read_parquet(path)
else:
    O = {}
    for h in ("huc", "basin"):
        uns = HOLDOUTS[h]
        B6, B7 = views("v6", h), views("v7", h)
        cb = base_cols(B6[0])
        nostat = [c for c in cb if c not in STATIC]
        for name, (V, cols) in {"V5 (v6 repair)": (B6, cb), "V6 (+dynamics)": (B6, cb + DYN),
                                "V7 (+dynamics, v7 repair)": (B7, cb + DYN), "V5 + v7 repair": (B7, cb),
                                "V5 - static": (B6, nostat)}.items():
            O[f"{h} | {name}"] = run_scheme(*V, folds, cols, uns=uns)[0]
            print(f"  done {h} | {name}", flush=True)
    O = pd.DataFrame(O)
    O.to_parquet(path)

rows = []
for h in ("huc", "basin"):
    uns = HOLDOUTS[h]
    base = O[f"{h} | V5 (v6 repair)"].to_numpy()
    tl_b = test_like(base, n=800)
    for name in ["V5 (v6 repair)", "V6 (+dynamics)", "V7 (+dynamics, v7 repair)", "V5 + v7 repair", "V5 - static"]:
        p = O[f"{h} | {name}"].to_numpy()
        per = [(ap(y[va], p[va]), ap(y[va & uns], p[va & uns]), ap(y[va & ~uns], p[va & ~uns])) for _, va in folds]
        per = np.array(per)
        tl = test_like(p, n=800)
        rows.append(dict(holdout=h, model=name, AP=per[:, 0].mean(), unseen=per[:, 1].mean(), seen=per[:, 2].mean(),
                         test_like=tl.mean(), tl_gain=(tl - tl_b).mean(), draws_up=(tl > tl_b).mean()))
R = pd.DataFrame(rows)
print("\n[B] leave-one-year-out, same model, two holdout designs:")
print(R.round(4).to_string(index=False))
print("[reference] public: V5 0.7552 > V6 0.751 > V7 0.748")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for i, h in enumerate(("huc", "basin")):
    g = R[R.holdout == h].set_index("model")
    g[["unseen", "seen"]].plot.bar(ax=ax[i], rot=20)
    ax[i].set_ylim(g[["unseen", "seen"]].values.min() - .02, g[["unseen", "seen"]].values.max() + .01)
    ax[i].set_title(f"LOYO AP by segment, {h} holdout")
plt.tight_layout()
plt.savefig(FIG / "44_basin_holdout.png", dpi=120)
print(f"[fig] {FIG / '44_basin_holdout.png'}")
