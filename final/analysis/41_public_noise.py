"""41 - Is a public difference of -0.004 (V5->V6) or -0.007 (V5->V7) bigger than leaderboard noise?
The public score is ONE draw: 4 origins, and possibly only a random fraction f of their rows (Kaggle splits
the test file into public and private rows). Here the same three model variants that separate V5/V6/V7 are
scored out-of-fold (leave-one-year-out), then many test-like draws are made: 4 origins in months 2/4/10/12,
more than 12 months apart, and a random fraction f of their rows. The paired difference distribution tells how
likely the observed public differences are if the CV-measured gain is real, and how likely the private
difference keeps the sign of the public one."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import DYN, base_cols, run_scheme, t, tr, views, y, year

folds = [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]
path = WORK / "41_oof.parquet"
if path.exists():
    O = pd.read_parquet(path)
else:
    B6, B7 = views("v6"), views("v7")
    cols = base_cols(B6[0])
    O = pd.DataFrame({
        "V5-like (v6 repair)": run_scheme(*B6, folds, cols)[0],
        "V6-like (+dynamics)": run_scheme(*B6, folds, cols + DYN)[0],
        "V7-like (+dynamics, v7 repair)": run_scheme(*B7, folds, cols + DYN)[0],
    })
    O.to_parquet(path)
names = list(O.columns)
P = O.to_numpy()
months = tr.month.values
by_m = {m: np.unique(t[months == m]) for m in (2, 4, 10, 12)}
idx_of = {k: np.flatnonzero(t == k) for k in np.unique(t)}
rng = np.random.default_rng(0)
rows = []
for f in (1.0, 0.5, 0.3):
    for _ in range(3000):
        pick = [rng.choice(by_m[m]) for m in by_m]
        if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
            continue
        ix = np.concatenate([idx_of[k] for k in pick])
        pub = rng.random(len(ix)) < f
        a_pub = [ap(y[ix[pub]], P[ix[pub], j]) for j in range(3)]
        a_prv = [ap(y[ix[~pub]], P[ix[~pub], j]) for j in range(3)] if f < 1 else [np.nan] * 3
        rows.append(dict(f=f, **{f"pub{j}": v for j, v in enumerate(a_pub)}, **{f"prv{j}": v for j, v in enumerate(a_prv)}))
D = pd.DataFrame(rows)
D.to_csv(WORK / "41_draws.csv", index=False)
obs = {1: -0.0042, 2: -0.0072}
print("[paired public difference vs V5-like] mean, std, P(diff <= observed public diff)")
for f, g in D.groupby("f"):
    for j in (1, 2):
        dd = g[f"pub{j}"] - g.pub0
        line = (f"  f={f:.1f} {names[j]:32s} mean {dd.mean():+.4f} std {dd.std():.4f} | "
                f"P(diff <= {obs[j]:+.4f}) = {(dd <= obs[j]).mean():.3f}")
        if f < 1:
            dp = g[f"prv{j}"] - g.prv0
            same = np.sign(dd) == np.sign(dp)
            line += (f" | corr(public diff, private diff) {np.corrcoef(dd, dp)[0, 1]:.3f}, same sign {same.mean():.2f}, "
                     f"P(private diff > 0 | public diff <= obs) {(dp[dd <= obs[j]] > 0).mean():.2f}")
        print(line)
g1 = D[D.f == 1.0]
both = ((g1.pub1 - g1.pub0) <= obs[1]) & ((g1.pub2 - g1.pub0) <= obs[2])
print(f"\n[joint] full-row public: P(V6 <= -0.0042 AND V7 <= -0.0072) = {both.mean():.3f}")
print(f"[scale] std of the absolute public score of one variant: {g1.pub0.std():.4f} "
      f"(the level moves 10x more than the paired difference)")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for j, c in [(1, "tab:blue"), (2, "tab:red")]:
    ax[0].hist(g1[f"pub{j}"] - g1.pub0, bins=60, alpha=.5, color=c, label=names[j])
    ax[0].axvline(obs[j], color=c, ls="--")
ax[0].axvline(0, color="k", lw=.8)
ax[0].legend(fontsize=8)
ax[0].set_title("Paired AP difference vs V5-like on 4-origin draws (dashed: observed public)")
g5 = D[D.f == 0.5]
ax[1].scatter(g5.pub2 - g5.pub0, g5.prv2 - g5.prv0, s=4, alpha=.3)
ax[1].axhline(0, color="k", lw=.8); ax[1].axvline(0, color="k", lw=.8)
ax[1].set_xlabel("public difference (50% of rows)"); ax[1].set_ylabel("private difference (other 50%)")
ax[1].set_title("V7-like minus V5-like: public vs private")
plt.tight_layout()
plt.savefig(FIG / "41_public_noise.png", dpi=120)
print(f"[fig] {FIG / '41_public_noise.png'}")
