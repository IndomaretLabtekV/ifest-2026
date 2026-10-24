"""55 - With an unlimited submission budget, is a FINER grouping worth it for level + slope calibration?
Only group-level quantities are probed (prevalence and within-group AP); they belong to the group, so they carry
over to the private rows. Probing individual rows would only memorise public labels, so it is not considered.
Groupings: origin (4 groups, 9 submissions), origin x new/seen basin (8, 17), origin x zone (12, 25).
Level shift vs level + slope, public = random 50% or 30% of the rows, gain measured on the PRIVATE rows.
Basin-holdout OOF of 44 so 'new' rows behave like the 47 new test basins."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import HOLDOUTS, t, tr, views, y
from probe import calibrate, moment_calibrate, probe_prevalence, segment_ap

THR = np.log(0.6)
O = pd.read_parquet(WORK / "44_oof.parquet")
p_all = O["basin | V5 (v6 repair)"].to_numpy()
new_all = HOLDOUTS["basin"].astype(int)
_, VV = views("v6", "basin")
zone_all = np.digitize(VV.anom0.values - THR, [0, .5])
months = tr.month.values
by_m = {m: np.unique(t[months == m]) for m in (2, 4, 10, 12)}
rows = []
rng = np.random.default_rng(11)
for frac in (0.5, 0.3):
    n = 0
    while n < 150:
        pick = [rng.choice(by_m[m]) for m in by_m]
        if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
            continue
        n += 1
        ix = np.concatenate([np.flatnonzero(t == k) for k in pick])
        yy, pp = y[ix], p_all[ix]
        org = np.concatenate([np.full((t == k).sum(), j) for j, k in enumerate(pick)])
        rest = np.setdiff1d(np.flatnonzero(~np.isnan(p_all)), ix)
        iso = IsotonicRegression(out_of_bounds="clip").fit(pd.Series(p_all[rest]).rank(pct=True), y[rest])
        pc = np.clip(iso.predict((pd.Series(pp).rank().to_numpy() - .5) / len(pp)), 1e-4, 1 - 1e-4)
        pub = rng.random(len(ix)) < frac
        pi = yy[pub].mean()
        base = ap(yy[~pub], pc[~pub])
        r = dict(frac=frac)
        for gname, g in [("origin (9)", org), ("origin x new (17)", org * 2 + new_all[ix]),
                         ("origin x zone (25)", org * 3 + zone_all[ix])]:
            ng = g.max() + 1
            shares = np.array([(g == k).mean() for k in range(ng)])
            prior = np.array([pc[g == k].mean() if (g == k).any() else 0 for k in range(ng)])
            prev = probe_prevalence(yy[pub], g[pub], shares, prior if ng > 4 else None)
            ap_obs = []
            for k in range(ng):
                mk = np.zeros(len(pc))
                if (g == k).any():
                    mk[g == k] = 0.01 + 0.99 * pd.Series(pc[g == k]).rank(pct=True).to_numpy()
                a = prev[k] * shares[k] / pi
                v = segment_ap(ap(yy[pub], mk[pub]), a, pi)
                ap_obs.append(v if np.isfinite(v) else np.nan)
            ap_obs = np.array(ap_obs)
            # groups whose AP could not be decoded keep slope 1 (handled inside moment_calibrate)
            lvl = calibrate(pc, g, prev)
            mom, _ = moment_calibrate(pc, g, prev, ap_obs, seed=n)
            r[f"{gname} | level"] = ap(yy[~pub], lvl[~pub]) - base
            r[f"{gname} | level+slope"] = ap(yy[~pub], mom[~pub]) - base
        rows.append(r)
    print(f"  frac {frac} done", flush=True)
R = pd.DataFrame(rows)
print("[55] private-row AP gain, mean (share of draws improved):")
for c in R.columns[1:]:
    print(f"  {c:34s} " + " | ".join(f"public {f:.0%}: {g[c].mean():+.4f} ({(g[c] > 0).mean():.0%})" for f, g in R.groupby("frac")))
fig, ax = plt.subplots(figsize=(12, 4.5))
R[R.frac == .5].drop(columns="frac").boxplot(ax=ax, rot=20, showfliers=False)
ax.axhline(0, color="k", lw=.8)
ax.set_title("Private-row AP gain by probe grouping (public = 50%)")
plt.tight_layout()
plt.savefig(FIG / "55_group_moment.png", dpi=120)
print(f"[fig] {FIG / '55_group_moment.png'}")
