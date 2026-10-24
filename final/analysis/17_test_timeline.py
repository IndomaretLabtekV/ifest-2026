"""17 - Re-test the 'test origins are >=12 months from train' conclusion of 02.
02 matched raw baseflow values, but 11 showed every test HUC is rescaled by a random factor, so exact
value matching could never succeed. Here seen-HUC test rows are divided by their exact factor first,
then matched against the train timeline for every lag offset."""
import numpy as np
import pandas as pd

from common import FIG, WORK, load_clean
from prep import clim_scale

tr, te = load_clean()
tr = tr.drop_duplicates(["id", "origin_id"])
tl = pd.read_csv(WORK / "timeline.csv").dropna(subset=["t"])
tr = tr.merge(tl[["origin_id", "t"]], on="origin_id")

# exact per-HUC factor from climatology (constant within HUC, see 11)
e_tr, e_te = 10.0 ** clim_scale(tr), 10.0 ** clim_scale(te)
tab = pd.concat([pd.DataFrame({"id": tr.id, "m": tr.month, "c": tr.clim_median_now * e_tr}),
                 pd.DataFrame({"id": tr.id, "m": tr.month % 12 + 1, "c": tr.clim_median_next * e_tr})]).groupby(["id", "m"]).c.median()
T = te[te.id.isin(tr.id)].copy()
T["fac"] = (T.clim_median_now * e_te[T.index] / tab.reindex(pd.MultiIndex.from_arrays([T.id, T.month])).values)
T["fac"] = T.groupby("id").fac.transform("median")
print(f"[factor] seen test rows {len(T)}, factor finite {np.isfinite(T.fac).mean():.3f}")

# train: value of (id, absolute month m) for baseflow
long = pd.concat([pd.DataFrame({"id": tr.id, "m": tr.t - k, "v": tr[f"q_bf_inc_m3_lag{k}"]}) for k in range(12)])
long = long[long.v > 0]
long["key"] = long.v.round(0)
vals = long.drop_duplicates(["id", "key", "m"])

for o, g in T.groupby("origin_id"):
    votes = {}
    for k in range(12):
        v = (g[f"q_bf_inc_m3_lag{k}"] / g.fac).round(0)
        q = pd.DataFrame({"id": g.id.values, "key": v.values}).dropna()
        q = q[q.key > 0]
        m = q.merge(vals, on=["id", "key"])
        # a hit at train month m for lag k means origin month = m + k
        for pos, n in (m.m + k).value_counts().items():
            votes[pos] = votes.get(pos, 0) + n
    top = sorted(votes.items(), key=lambda x: -x[1])[:3]
    print(f"[timeline] test origin {o} (month {g.month.iloc[0]}): best implied chain position -> {top}"
          f"  (train chain = 0..{int(tr.t.max())})")

# robust check: best 12-month window match in log space (no rounding), per sampled test row
S = T[np.isfinite(T.fac)].sample(400, random_state=0)
series = long.groupby(["id", "m"]).v.median()
best, best_pos = [], []
have = set(series.index.get_level_values(0))
for _, r in S.iterrows():
    if r.id not in have:
        continue
    s = series.loc[r.id]
    q = np.log(np.array([r[f"q_bf_inc_m3_lag{k}"] for k in range(12)], float) / r.fac)
    d = []
    for pos in range(int(s.index.min()) + 11, int(s.index.max()) + 12 + 11):
        ref = np.log(s.reindex([pos - k for k in range(12)]).to_numpy(float))
        ok = np.isfinite(q) & np.isfinite(ref)
        if ok.sum() >= 4:
            d.append((np.median(np.abs(q[ok] - ref[ok])), pos, ok.sum()))
    if d:
        m = min(d)
        best.append(m[0]); best_pos.append(m[1])
best = np.array(best)
print(f"\n[window] rows {len(best)}: best-window median |log diff| quantiles "
      f"{np.round(np.quantile(best, [.01, .1, .5, .9]), 4).tolist()}; share < 0.01 (exact overlap): {(best < .01).mean():.3f}")
# reference: a train row matched against its own series gives ~0
print("[window] an exact overlap would give 0.000; natural year-to-year analog distance is the median above")
