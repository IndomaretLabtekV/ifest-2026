"""06 - Climatology gaps. Unseen test HUCs only expose clim for the months around test origins
(2,4,10,12 -> 1,2,3,4,5,10,11,12), so Jun-Sep are missing and every anomaly feature for those lags is NaN.
Simulate that exact situation on train HUCs (where all 12 months are known) and score gap-fill methods."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import FIG, WORK, load_clean
from fe import clim_table
from prep import repair

tr, te = load_clean()
tr = tr.drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
full = clim_table(pd.concat([tr, te]))
seen = full.dropna().index.intersection(tr.id.unique())
print(f"[gap] test HUCs with incomplete clim: {full.reindex(te.id.unique()).isna().any(axis=1).sum()} / {te.id.nunique()}")
print("[gap] missing months among them:", full.reindex(te.id.unique()).isna().sum().to_dict())

# simulate: per train HUC keep 4 rows whose origin months are 2,4,10,12 (like the test)
sim = (tr[tr.id.isin(seen) & tr.month.isin([2, 4, 10, 12])].sample(frac=1, random_state=0)
       .drop_duplicates(["id", "month"]).reset_index(drop=True))
sim = repair(sim)
part = clim_table(sim)
miss = part.isna()
print(f"[sim] simulated HUCs {part.shape[0]}, missing cells per HUC {miss.sum(1).mean():.1f} (months {list(part.columns[miss.any()])})")

# method A: cyclic interpolation of log clim across calendar months
def fill_cyclic(P):
    L = np.log1p(P)
    T = pd.concat([L, L, L], axis=1)
    T.columns = range(36)
    T = T.T.interpolate(limit_direction="both").T
    return np.expm1(T.iloc[:, 12:24].set_axis(P.columns, axis=1))

# method B: the HUC's own observed ws for that calendar month (lags of its 4 rows), rescaled by known months
def fill_empirical(P, rows):
    long = pd.concat([pd.DataFrame({"id": rows.id, "m": (rows.month - 1 - k) % 12 + 1, "v": rows[f"q_ws_cum_m3_lag{k}"]})
                      for k in range(12)])
    emp = long.groupby(["id", "m"]).v.median().unstack().reindex(index=P.index, columns=P.columns)
    k = np.log((P + 1) / (emp + 1)).median(axis=1)
    return P.fillna(np.exp(k).values[:, None] * (emp + 1) - 1)

# method C: basin-mean seasonal shape x HUC level
def fill_shape(P, ref):
    shape = np.log1p(ref).sub(np.log1p(ref).mean(1), axis=0).mean()
    lvl = np.log1p(P).sub(shape, axis=1).mean(1)
    return P.fillna(pd.DataFrame(np.expm1(lvl.values[:, None] + shape.values[None, :]), P.index, P.columns))

# method D: kNN analogues - HUCs whose known-month seasonal shape is closest lend their Jun-Sep shape
def fill_knn(P, ref, k=15):
    known = [m for m in P.columns if P[m].notna().mean() > .99]
    R = np.log1p(ref); R = R.sub(R[known].mean(1), axis=0)
    Q = np.log1p(P); lvl = Q[known].mean(1); Q = Q.sub(lvl, axis=0)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(R[known].values)
    _, idx = nn.kneighbors(Q[known].fillna(0).values)
    self_hit = np.asarray(R.index, dtype=object)[idx] == np.asarray(Q.index, dtype=object)[:, None]
    idx = np.array([i[~s][:k] for i, s in zip(idx, self_hit)])
    est = R.values[idx].mean(1) + lvl.values[:, None]
    return P.fillna(pd.DataFrame(np.expm1(est), P.index, P.columns))

truth = full.loc[part.index]
methods = {"cyclic interp": fill_cyclic(part), "empirical ws": fill_empirical(part, sim),
           "national shape": fill_shape(part, full.loc[seen])}
methods["kNN analog"] = fill_knn(part, full.loc[seen])
methods["kNN+empirical avg"] = np.expm1((np.log1p(methods["kNN analog"]) + np.log1p(methods["empirical ws"])) / 2)
methods["empirical+cyclic avg"] = np.expm1((np.log1p(methods["cyclic interp"]) + np.log1p(methods["empirical ws"])) / 2)
res = {}
for n, P in methods.items():
    e = (np.log(P + 1) - np.log(truth + 1))[miss].stack().dropna()
    res[n] = {"median |log err|": e.abs().median(), "within 20%": (e.abs() < np.log(1.2)).mean(),
              "bias": e.median()}
res = pd.DataFrame(res).T
print(res.round(3).to_string())

# does it matter for the 0.6 threshold? decision flip rate of 'ws < 0.6 clim' on the missing months
ws = pd.concat([pd.DataFrame({"id": sim.id, "m": (sim.month - 1 - k) % 12 + 1, "v": sim[f"q_ws_cum_m3_lag{k}"]})
                for k in range(12)])
for n, P in methods.items():
    a = ws.join(P.stack().rename("c"), on=["id", "m"]).join(truth.stack().rename("ct"), on=["id", "m"])
    a = a[a.m.isin([6, 7, 8, 9])]
    print(f"[flip] {n:22s} below-0.6 decision agrees with true clim: {((a.v < .6 * a.c) == (a.v < .6 * a.ct)).mean():.3f}")

best = res["median |log err|"].idxmin()
e = (np.log(methods[best] + 1) - np.log(truth + 1))[miss].stack()
plt.figure(figsize=(7, 4)); plt.hist(e.clip(-2, 2), bins=80, color="#4C72B0")
plt.title(f"log error of filled clim (Jun-Sep), method: {best}"); plt.tight_layout()
plt.savefig(FIG / "06_clim_gap.png", dpi=110); plt.close()
