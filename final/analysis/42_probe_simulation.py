"""42 - Leaderboard probing, simulated honestly end to end.
Kaggle scores the public board on a random fraction f of the test rows; f is unknown. For a binary probe that
puts 1 on group S (one origin, or one origin x zone cell) and 0 elsewhere:
    AP_S = a * (k / n_S) + (1 - a) * pi,   a = k / P (share of all positives inside S), pi = P / N
With a random public split n_S / N equals the group's share s of the test rows, so k / n_S = a * pi / s and
    AP_S = (pi / s) a^2 - pi a + pi      -> a quadratic in a that needs neither f nor the public row count.
The constant probe gives pi. Group prevalence = a * pi / s. Roots are chosen so that the shares a sum to 1.
The recovered prevalences (from PUBLIC rows only) drive a constant logit shift per group, and the gain is
measured on the PRIVATE rows, i.e. what probing is worth if the private board is the other rows of the same
4 origins. Scores: model probabilities vs rank-normalised scores (what the V5-V7 notebooks shift)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import t, tr, views, y
from probe import calibrate, probe_prevalence

THR = np.log(0.6)
O = pd.read_parquet(WORK / "41_oof.parquet")
p_all = O["V7-like (+dynamics, v7 repair)"].to_numpy()
_, VV = views("v7")
zone_all = np.digitize(VV.anom0.values - THR, [0, .5])  # 0 dry, 1 onset, 2 safe
months = tr.month.values
by_m = {m: np.unique(t[months == m]) for m in (2, 4, 10, 12)}
idx_of = {k: np.flatnonzero(t == k) for k in np.unique(t)}


rng = np.random.default_rng(0)
rows, err_rows = [], []
while len(rows) < 600:
    pick = [rng.choice(by_m[m]) for m in by_m]
    if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
        continue
    ix = np.concatenate([idx_of[k] for k in pick])
    yy, pp = y[ix], p_all[ix]
    org = np.concatenate([np.full(len(idx_of[k]), j) for j, k in enumerate(pick)])
    oz = org * 3 + zone_all[ix]
    rk = (pd.Series(pp).rank().to_numpy() - .5) / len(pp)
    # a global isotonic map from rank to probability, learned on other years (here: all OOF outside the draw)
    rest = np.setdiff1d(np.flatnonzero(~np.isnan(p_all)), ix)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pd.Series(p_all[rest]).rank(pct=True), y[rest])
    rk_iso = np.clip(iso.predict(rk), 1e-4, 1 - 1e-4)
    for f in (1.0, 0.5, 0.3):
        pub = rng.random(len(ix)) < f
        prv = ~pub if f < 1 else pub
        r = dict(f=f)
        for gname, groups, ng in [("origin", org, 4), ("origin x zone", oz, 12)]:
            shares = np.array([(groups == g).mean() for g in range(ng)])
            prior = np.array([pp[groups == g].mean() if (groups == g).any() else 0 for g in range(ng)])
            est = probe_prevalence(yy[pub], groups[pub], shares, prior if ng > 4 else None)
            true = np.array([yy[groups == g].mean() if (groups == g).any() else 0 for g in range(ng)])
            err_rows.append(dict(f=f, groups=gname, mae=np.abs(est - true).mean()))
            for sname, sc in [("prob", pp), ("rank", rk), ("rank->isotonic", rk_iso)]:
                r[f"{gname} | {sname}"] = ap(yy[prv], calibrate(sc, groups, est)[prv]) - ap(yy[prv], sc[prv])
            r[f"{gname} | oracle prob"] = ap(yy[prv], calibrate(pp, groups, true)[prv]) - ap(yy[prv], pp[prv])
        rows.append(r)
R = pd.DataFrame(rows)
E = pd.DataFrame(err_rows)
print("[recovery] mean absolute error of the probed prevalence vs the true prevalence of ALL rows in the group:")
print(E.groupby(["groups", "f"]).mae.mean().unstack().round(4).to_string())
print("\n[gain on PRIVATE rows] mean AP gain (share of draws improved); f = public fraction, f=1 means public=private:")
for c in R.columns[1:]:
    line = "  " + f"{c:34s}"
    for f, g in R.groupby("f"):
        line += f" | f={f:.1f} {g[c].mean():+.4f} ({(g[c] > 0).mean():.0%})"
    print(line)
print("\n[submissions needed] origin: 1 constant + 4 = 5; origin x zone: 1 + 12 = 13 (one group can be implied by the sum)")

fig, ax = plt.subplots(figsize=(11, 4.5))
g5 = R[R.f == 0.5].drop(columns="f")
g5.boxplot(ax=ax, rot=20, showfliers=False)
ax.axhline(0, color="k", lw=.8)
ax.set_title("Private-row AP gain from probing public rows (f = 0.5)")
plt.tight_layout()
plt.savefig(FIG / "42_probe_simulation.png", dpi=120)
print(f"[fig] {FIG / '42_probe_simulation.png'}")
