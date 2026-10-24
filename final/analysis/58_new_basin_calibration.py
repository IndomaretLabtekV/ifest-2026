"""58 - Is the model systematically mis-levelled on NEW basins, and can the data alone (no leaderboard) fix it?
Basin-holdout leave-one-year-out OOF of the V9-like LightGBM (extra trees, 48). For every validation origin the rows
split into held-out basins (like the 47 new test basins) and seen HUCs.
  A) calibration error per (origin, new/seen): mean prediction - true prevalence, and what it correlates with
  B) data-only corrections, each fitted on OTHER years only (nested by year) and judged on test-like 4-origin draws:
       - segment isotonic: a separate isotonic map for new vs seen rows
       - group shift: ridge on label-free group aggregates (mean logit, share below 0.6 now, mean anomaly) predicting
         the group's logit prevalence, then a constant logit shift per (origin, new/seen)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import HOLDOUTS, t, test_like, tr, views, y, year
from probe import calibrate

THR = np.log(0.6)
p = pd.read_parquet(WORK / "48_oof.parquet")["extra trees"].to_numpy()
new = HOLDOUTS["basin"]
_, VV = views("v7", "basin")
G = pd.DataFrame({"t": t, "year": year, "new": new.astype(int), "y": y, "p": p, "lp": logit(np.clip(p, 1e-5, 1 - 1e-5)),
                  "below": (VV.anom0.values < THR).astype(float), "anom0": VV.anom0.values, "a3": VV.anom_mean3.values})
grp = G.groupby(["t", "new"]).agg(year=("year", "first"), y=("y", "mean"), p=("p", "mean"), lp=("lp", "mean"),
                                  below=("below", "mean"), anom0=("anom0", "mean"), a3=("a3", "mean"), n=("y", "size"))
grp["err"] = grp.p - grp.y
print("[A] mean prediction - true prevalence per (origin, segment):")
for k, g in grp.groupby(level="new"):
    print(f"  {'new basins' if k else 'seen HUCs ':10s} bias {g.err.mean():+.4f}  MAE {g.err.abs().mean():.4f}  "
          f"corr(err, share below now) {np.corrcoef(g.err, g.below)[0, 1]:+.3f}  corr(err, mean anomaly) {np.corrcoef(g.err, g.anom0)[0, 1]:+.3f}")
w = grp.xs(1, level="new").err - grp.xs(0, level="new").err
print(f"[A] new minus seen error within the same origin: mean {w.mean():+.4f}, std {w.std():.4f}")

# ---------------- B: data-only corrections, nested by year ----------------
seg_iso, grp_shift = np.full(len(p), np.nan), np.full(len(p), np.nan)
feat = ["lp", "below", "anom0", "a3"]
for k in range(14):
    tr_rows, te_rows = G.year != k, G.year == k
    out = np.empty(te_rows.sum())
    for s in (0, 1):
        f = IsotonicRegression(out_of_bounds="clip").fit(G.p[tr_rows & (G.new == s)], G.y[tr_rows & (G.new == s)])
        m = (G.new[te_rows] == s).to_numpy()
        out[m] = f.predict(G.p[te_rows][m]) + 1e-9 * G.p[te_rows][m]
    seg_iso[te_rows.to_numpy()] = out
    gtr = grp[grp.year != k]
    reg = RidgeCV(alphas=np.logspace(-2, 3, 11)).fit(gtr[feat].assign(new=gtr.index.get_level_values("new")),
                                                      logit(gtr.y.clip(1e-3, 1 - 1e-3)))
    gte = grp[grp.year == k]
    target = expit(reg.predict(gte[feat].assign(new=gte.index.get_level_values("new"))))
    idx = np.flatnonzero(te_rows)
    keys = list(zip(G.t.values[idx], G.new.values[idx]))
    codes = pd.Series(keys).astype("category").cat.codes.to_numpy()
    cats = pd.Series(keys).astype("category").cat.categories
    prev = np.array([target[list(gte.index).index(c)] for c in cats])
    grp_shift[idx] = calibrate(p[idx], codes, prev)
tl0 = test_like(p, n=800)
print("\n[B] test-like 4-origin AP (nested by year):")
for name, q in [("segment isotonic", seg_iso), ("group shift (ridge)", grp_shift)]:
    tl = test_like(q, n=800)
    per = np.array([ap(y[year == k], q[year == k]) - ap(y[year == k], p[year == k]) for k in range(14)])
    print(f"  {name:22s} {tl.mean():.4f} vs raw {tl0.mean():.4f}: {(tl - tl0).mean():+.4f}, draws up {(tl > tl0).mean():.0%} | "
          f"per-year AP change {per.mean():+.4f} (up {(per > 0).sum()}/14)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k, c in [(0, "tab:blue"), (1, "tab:red")]:
    g = grp.xs(k, level="new")
    ax[0].scatter(g.y, g.p, s=10, c=c, label="new basins" if k else "seen HUCs", alpha=.6)
ax[0].plot([0, .6], [0, .6], "k:"); ax[0].legend()
ax[0].set_xlabel("true prevalence of the group"); ax[0].set_ylabel("mean prediction")
ax[0].set_title("Group calibration per origin (basin holdout)")
ax[1].scatter(grp.xs(1, level="new").below, grp.xs(1, level="new").err, s=10, c="tab:red")
ax[1].axhline(0, color="k", lw=.8)
ax[1].set_xlabel("share of the new basins below 0.6 now"); ax[1].set_ylabel("mean prediction - prevalence")
ax[1].set_title("New basins: calibration error vs current state")
plt.tight_layout()
plt.savefig(FIG / "58_new_basin_calibration.png", dpi=120)
print(f"[fig] {FIG / '58_new_basin_calibration.png'}")
