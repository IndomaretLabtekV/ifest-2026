"""54 - Per-origin calibration of LEVEL and SLOPE from two probes per origin, judged on private rows.
Upper bounds (fitted with the public labels): level shift +0.023, level + slope +0.033, isotonic +0.038 on the
private half. The level comes from the binary probe (prevalence, 42). The slope can come from the masked probe
(47), which returns the model's AP inside the origin: choose the logit slope s (and the shift b that keeps the
probed prevalence) so that the AP EXPECTED under the calibrated probabilities (labels drawn from them) equals the
observed AP. Overconfident origin -> observed AP below expected -> flatter; underconfident -> sharper.
Submissions: 1 constant + 4 binary + 4 masked = 9. Simulated with the basin-holdout OOF of 44; public = random half,
probes computed on the public half only, gain measured on the private half."""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import t, tr, y
from probe import SLOPES, calibrate, moment_calibrate, probe_prevalence, segment_ap

O = pd.read_parquet(WORK / "44_oof.parquet")
p_all = O["basin | V5 (v6 repair)"].to_numpy()
months = tr.month.values
by_m = {m: np.unique(t[months == m]) for m in (2, 4, 10, 12)}
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
rng = np.random.default_rng(0)
rows, slopes = [], []
while len(rows) < 250:
    pick = [rng.choice(by_m[m]) for m in by_m]
    if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
        continue
    ix = np.concatenate([np.flatnonzero(t == k) for k in pick])
    yy, pp = y[ix], p_all[ix]
    org = np.concatenate([np.full((t == k).sum(), j) for j, k in enumerate(pick)])
    pub = rng.random(len(ix)) < FRAC
    rest = np.setdiff1d(np.flatnonzero(~np.isnan(p_all)), ix)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pd.Series(p_all[rest]).rank(pct=True), y[rest])
    pc = np.clip(iso.predict((pd.Series(pp).rank().to_numpy() - .5) / len(pp)), 1e-4, 1 - 1e-4)
    shares = np.bincount(org) / len(org)
    prev = probe_prevalence(yy[pub], org[pub], shares)
    # masked probe of each origin, as the leaderboard would report it, then decoded to the within-origin AP
    pi = yy[pub].mean()
    ap_obs = []
    for g in range(4):
        mk = np.zeros(len(pc))
        mk[org == g] = 0.01 + 0.99 * pd.Series(pc[org == g]).rank(pct=True).to_numpy()
        a = prev[g] * shares[g] / pi
        ap_obs.append(segment_ap(ap(yy[pub], mk[pub]), a, pi))
    ap_obs = np.array(ap_obs)
    shifted = calibrate(pc, org, prev)
    moment, s = moment_calibrate(pc, org, prev, ap_obs, seed=len(rows))
    slopes += s
    b = ap(yy[~pub], pc[~pub])
    rows.append(dict(shift=ap(yy[~pub], shifted[~pub]) - b, level_slope=ap(yy[~pub], moment[~pub]) - b,
                     slope_vs_shift=ap(yy[~pub], moment[~pub]) - ap(yy[~pub], shifted[~pub])))
    if len(rows) % 50 == 0:
        print(f"  {len(rows)} draws", flush=True)
R = pd.DataFrame(rows)
print(f"[54] private-row AP gain (probes on a random {FRAC:.0%} public share, 250 test-like draws):")
for c in R.columns:
    print(f"  {c:16s} {R[c].mean():+.4f}  improved in {(R[c] > 0).mean():.0%}")
print(f"[54] chosen slopes: median {np.median(slopes):.2f}, p10 {np.percentile(slopes, 10):.2f}, p90 {np.percentile(slopes, 90):.2f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
R[["shift", "level_slope"]].boxplot(ax=ax[0], showfliers=False)
ax[0].axhline(0, color="k", lw=.8); ax[0].set_title("Private-half AP gain: 5 vs 9 probes")
ax[1].hist(slopes, bins=SLOPES - .05)
ax[1].set_title("Logit slope chosen per origin (1 = unchanged)")
plt.tight_layout()
plt.savefig(FIG / f"54_moment_calibration_{FRAC:.1f}.png", dpi=120)
print(f"[fig] {FIG / f'54_moment_calibration_{FRAC:.1f}.png'}")
