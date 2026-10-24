"""45 - Which probe grouping is worth its submissions? Now scored with the basin holdout of 44, so the
'new basin' rows behave like the 26% test-only rows. Groups (one binary probe each + 1 constant probe):
  origin (4), origin x zone (12), origin x seen/new basin (8), origin x zone x seen/new (24).
Scores: rank -> isotonic probability (best in 42). Public = random half of the rows, gain measured on the
private half. Roots are picked with the model's own group prevalence as prior (42)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import HOLDOUTS, t, tr, views, y
from probe import calibrate, probe_prevalence

THR = np.log(0.6)
O = pd.read_parquet(WORK / "44_oof.parquet")
p_all = O["basin | V5 (v6 repair)"].to_numpy()
new_all = HOLDOUTS["basin"].astype(int)
_, VV = views("v6", "basin")
zone_all = np.digitize(VV.anom0.values - THR, [0, .5])
months = tr.month.values
by_m = {m: np.unique(t[months == m]) for m in (2, 4, 10, 12)}
idx_of = {k: np.flatnonzero(t == k) for k in np.unique(t)}
rng = np.random.default_rng(1)
rows = []
while len(rows) < 500:
    pick = [rng.choice(by_m[m]) for m in by_m]
    if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
        continue
    ix = np.concatenate([idx_of[k] for k in pick])
    yy, pp = y[ix], p_all[ix]
    org = np.concatenate([np.full(len(idx_of[k]), j) for j, k in enumerate(pick)])
    zn, nw = zone_all[ix], new_all[ix]
    rest = np.setdiff1d(np.flatnonzero(~np.isnan(p_all)), ix)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pd.Series(p_all[rest]).rank(pct=True), y[rest])
    sc = np.clip(iso.predict((pd.Series(pp).rank().to_numpy() - .5) / len(pp)), 1e-4, 1 - 1e-4)
    pub = rng.random(len(ix)) < .5
    r = {}
    for gname, groups in [("origin (5 subs)", org), ("origin x new/seen (9 subs)", org * 2 + nw),
                          ("origin x zone (13 subs)", org * 3 + zn), ("origin x zone x new/seen (25 subs)", (org * 3 + zn) * 2 + nw)]:
        ng = groups.max() + 1
        shares = np.array([(groups == g).mean() for g in range(ng)])
        prior = np.array([sc[groups == g].mean() if (groups == g).any() else 0 for g in range(ng)])
        est = probe_prevalence(yy[pub], groups[pub], shares, prior if ng > 4 else None)
        true = np.array([yy[groups == g].mean() if (groups == g).any() else 0 for g in range(ng)])
        base = ap(yy[~pub], sc[~pub])
        r[gname] = ap(yy[~pub], calibrate(sc, groups, est)[~pub]) - base
        r[gname + " | oracle"] = ap(yy[~pub], calibrate(sc, groups, true)[~pub]) - base
    rows.append(r)
R = pd.DataFrame(rows)
print("[private-half AP gain] mean (share of draws improved):")
for c in R.columns:
    print(f"  {c:44s} {R[c].mean():+.4f} ({(R[c] > 0).mean():.0%})")
fig, ax = plt.subplots(figsize=(12, 4.5))
R[[c for c in R.columns if "oracle" not in c]].boxplot(ax=ax, rot=10, showfliers=False)
ax.axhline(0, color="k", lw=.8)
ax.set_title("Private-half AP gain by probe grouping (basin holdout, public = random half)")
plt.tight_layout()
plt.savefig(FIG / "45_probe_groups.png", dpi=120)
print(f"[fig] {FIG / '45_probe_groups.png'}")
