"""66 - Decision under leaderboard noise: V8-style blend (public 0.75548) or V14-style blend (public 0.751)?
The robust CV (63) says the V14 style is better in 13/14 years; the public says the opposite by 0.0045. Which to trust
for the PRIVATE board? Simulation on the validation OOF: draw a test-like set (4 origins with calendar months
2/4/10/12, > 12 months apart, rows of seen HUCs AND held-out basins), split its rows at random into public/private,
rank-blend the members INSIDE the draw (as the notebook ranks the whole test at once; ranking per year first would
erase the level between years, the 62 artefact), and record the paired differences on both parts.
Then: P(private diff > 0 | public diff ~ the observed -0.0045)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import TEST_MONTHS, t, tr, y

A = {"V8 final: lgbV8+xgb+cat": ["lgb V8", "xgboost", "catboost"]}
B = {"V14 final: tuned+dart": ["lgb V13 tuned", "lgb dart"],
     "tuned+dart+xgb+cat": ["lgb V13 tuned", "lgb dart", "xgboost", "catboost"],
     "lgbV8+tuned+dart": ["lgb V8", "lgb V13 tuned", "lgb dart"],
     "all5: lgbV8+tuned+dart+xgb+cat": ["lgb V8", "lgb V13 tuned", "lgb dart", "xgboost", "catboost"]}
OBS = 0.751 - 0.75548
month = tr.month.values
by_m = {m: np.unique(t[month == m]) for m in TEST_MONTHS}
idx_t = pd.Series(np.arange(len(t))).groupby(t).apply(np.array).to_dict()


def blend(rows, members, P, col):
    S = P[np.ix_(rows, [col[m] for m in members])]
    return (S.argsort(0).argsort(0) / len(rows)).mean(1)


recs = []
for seed in (42, 1, 7):  # the three whole-basin splits of 63
    O = pd.read_parquet(WORK / f"63_oof_s{seed}.parquet")
    P, col = O.to_numpy(), {c: i for i, c in enumerate(O.columns)}
    rng = np.random.default_rng(seed)
    n0 = len(recs)
    while len(recs) - n0 < 2 * len(B) * 300:
            pick = [rng.choice(by_m[m]) for m in TEST_MONTHS]
            if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
                    continue
            rows = np.concatenate([idx_t[k] for k in pick])
            yy = y[rows]
            pa = blend(rows, A["V8 final: lgbV8+xgb+cat"], P, col)
            for frac in (0.3, 0.5):
                    pub = rng.random(len(rows)) < frac
                    for name, mem in B.items():
                            pb = blend(rows, mem, P, col)
                            recs.append(dict(seed=seed, frac=frac, cand=name, full=ap(yy, pb) - ap(yy, pa),
                                                              pub=ap(yy[pub], pb[pub]) - ap(yy[pub], pa[pub]),
                                                              priv=ap(yy[~pub], pb[~pub]) - ap(yy[~pub], pa[~pub]),
                                                              ap_a=ap(yy, pa)))
R = pd.DataFrame(recs)
R["draw"] = np.arange(len(R)) // (2 * len(B))
print(f"[66] {R.groupby(['frac', 'cand']).size().iloc[0]} test-like draws per setting; V8-blend AP per draw "
      f"mean {R.ap_a.mean():.4f}, p10-p90 {R.ap_a.quantile(.1):.3f}-{R.ap_a.quantile(.9):.3f}")
for (frac, cand), g in R.groupby(["frac", "cand"]):
    r = np.corrcoef(g.pub, g.priv)[0, 1]
    near = g[(g.pub - OBS).abs() < 0.003]
    print(f"  public {frac:.0%} | {cand:22s} diff full {g.full.mean():+.4f} (>0 in {(g.full > 0).mean():.0%}) | "
          f"sd public {g.pub.std():.4f} sd private {g.priv.std():.4f} corr {r:.2f} | P(public diff <= {OBS:+.4f}) "
          f"{(g.pub <= OBS).mean():.0%} | given public ~{OBS:+.4f} (n={len(near)}): private diff {near.priv.mean():+.4f}, "
          f"P(private > 0) {(near.priv > 0).mean():.0%}")
R.to_csv(WORK / "66_decision.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
g = R[(R.frac == 0.3) & (R.cand == "V14 final: tuned+dart")]
ax[0].scatter(g.pub * 1000, g.priv * 1000, s=6, alpha=.4)
ax[0].axvline(OBS * 1000, color="r", ls="--", label=f"observed public diff {OBS * 1000:+.1f}")
ax[0].axhline(0, color="k", lw=.8); ax[0].axvline(0, color="k", lw=.8)
ax[0].set_xlabel("public diff V14 - V8 (x1000, 30% of rows)"); ax[0].set_ylabel("private diff (x1000)")
ax[0].set_title("Paired public vs private difference, test-like draws")
ax[0].legend()
for cand in B:
    ax[1].hist(R[(R.frac == 0.3) & (R.cand == cand)].full * 1000, bins=50, alpha=.5, label=cand)
ax[1].axvline(0, color="k")
ax[1].set_xlabel("full-test AP diff vs V8 blend (x1000)")
ax[1].set_title("How often a candidate beats the V8 blend on a whole test-like set")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "66_public_private_decision.png", dpi=120)
print(f"[fig] {FIG / '66_public_private_decision.png'}")

print("\n[66] conditional on the OBSERVED public diff of V14 (~-0.0045), what happens to each candidate in private?")
for frac in (0.3, 0.5):
    W = R[R.frac == frac].pivot(index="draw", columns="cand", values=["pub", "priv"])
    near = W[(W["pub"]["V14 final: tuned+dart"] - OBS).abs() < 0.003]
    print(f"  public {frac:.0%} (n={len(near)} draws over 3 basin splits):")
    for c in B:
        print(f"    {c:32s} private diff vs V8 blend {near['priv'][c].mean():+.4f}, P(>0) {(near['priv'][c] > 0).mean():.0%}")
