"""28 - Is there anything left in HOW the scores are combined across origins?
AP over the test pools 4 origins, so the relative level of scores between origins matters, not only the
ranking inside each origin. Variants tested on test-like 4-origin draws:
  raw                      the model's scores
  per-origin rank          ranks recomputed inside each origin (kills cross-origin level information)
  per-origin z(logit)      standardise each origin's logits
  oracle prevalence shift  shift each origin's logits so its mean predicted probability equals the TRUE
                           positive rate of that origin  <- upper bound of what leaderboard probing of
                           per-origin prevalence (14_probe_math.py) could ever buy
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap

from common import WORK

D = pd.read_parquet(WORK / "18_oof.parquet")
tr = pd.read_parquet(WORK / "train.parquet").drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
tl = pd.read_csv(WORK / "timeline.csv").dropna(subset=["t"])
tr = tr.merge(tl[["origin_id", "t"]], on="origin_id", how="left")
D = D.assign(origin=tr.origin_id.values[D.index], month=tr.month.values[D.index])
print(f"[data] OOF rows {len(D)}, folds {D.fold.nunique()}, origins {D.origin.nunique()}")

lg = logit(np.clip(D.p.values, 1e-6, 1 - 1e-6))


def variants(g):
    l = logit(np.clip(g.p.values, 1e-6, 1 - 1e-6))
    o = g.origin.values
    out = {"raw": g.p.values}
    out["per-origin rank"] = pd.Series(l).groupby(o).transform(lambda s: rankdata(s) / len(s)).to_numpy()
    out["per-origin z(logit)"] = pd.Series(l).groupby(o).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9)).to_numpy()
    shift = pd.Series(index=g.index, dtype=float)
    for oo, idx in g.groupby("origin").groups.items():
        sub = g.loc[idx]
        target = sub.y.mean()
        lo, hi = -10.0, 10.0
        for _ in range(40):  # bisection on a constant logit shift
            mid = (lo + hi) / 2
            if expit(logit(np.clip(sub.p.values, 1e-6, 1 - 1e-6)) + mid).mean() < target:
                lo = mid
            else:
                hi = mid
        shift.loc[idx] = (lo + hi) / 2
    out["oracle prevalence shift"] = expit(l + shift.to_numpy())
    return out


rng = np.random.default_rng(0)
scores = {k: [] for k in ["raw", "per-origin rank", "per-origin z(logit)", "oracle prevalence shift"]}
for f, g in D.groupby("fold"):
    by_m = [g.origin[g.month == m].unique() for m in (2, 4, 10, 12)]
    for _ in range(150):
        pick = [rng.choice(o) for o in by_m if len(o)]
        s = g[g.origin.isin(pick)]
        if s.y.nunique() < 2:
            continue
        v = variants(s)
        for k in scores:
            scores[k].append(ap(s.y, v[k]))
R = pd.DataFrame(scores)
print("\n[4-origin draws] mean AP by score transformation:")
print(R.mean().round(4).to_string())
print("\n[vs raw] mean difference and share of draws improved:")
for k in R.columns[1:]:
    d = R[k] - R["raw"]
    print(f"  {k:26s} {d.mean():+.4f}  improved in {(d > 0).mean():.1%} of draws")
prev = D.groupby("origin").y.mean()
print(f"\n[context] per-origin positive rate in train: min {prev.min():.3f} median {prev.median():.3f} max {prev.max():.3f}")
print(f"[context] correlation between origin mean prediction and origin true rate: "
      f"{np.corrcoef(D.groupby('origin').p.mean(), prev)[0, 1]:.3f}")
