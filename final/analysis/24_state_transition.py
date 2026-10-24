"""24 - State-conditioned transition probability (screen before modelling).
20/21 used every history year with equal weight. Recovery from a deep drought (depleted groundwater)
should differ from recovery from a mild one, so history growths are weighted by how close the
history year's origin-month anomaly is to the query's current anomaly: w = exp(-|a_hist - a0| / bw).
Label-free. Screened by univariate AUC inside the dry and onset zones against the unweighted p_exceed."""
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

from prep import repair

h = importlib.import_module("13_v2_cv")
FOLDS, y, tr = h.FOLDS, h.y, h.tr
THR = np.log(0.6)
trn, va, uns = FOLDS[1]
part = ~va & ~uns
F = h.fold_features(1)
src = part | va
X = repair(pd.concat([tr[part], h.rescale(tr[va], seed=101)]).sort_index())
Fs = F[src]

# growth entries: destination = lag k, origin = lag k+1; origin anomaly = anom{k+1} of that row
parts = []
for k in range(11):
    a, b = X[f"q_ws_cum_m3_lag{k}"].abs().values, X[f"q_ws_cum_m3_lag{k + 1}"].abs().values
    parts.append(pd.DataFrame({"id": X.id.values, "T": X.t.values - k, "cm": (X.month.values - 1 - k) % 12 + 1,
                               "g": np.log((a + 1) / (b + 1)), "a": Fs[f"anom{k + 1}"].values}))
G = pd.concat(parts).dropna().groupby(["id", "cm", "T"])[["g", "a"]].median().reset_index()
hist = {key: grp[["T", "g", "a"]].to_numpy() for key, grp in G.groupby(["id", "cm"])}

q = np.where(va)[0]
req = -(F.margin_raw_persist.values[q])
a0 = F.anom0.values[q]
out = {n: np.full(len(q), np.nan) for n in ["plain", "bw0.25", "bw0.5", "bw1.0"]}
for j, (i, hid) in enumerate(zip(q, tr.id.values[q])):
    H = hist.get((hid, tr.month.values[i] % 12 + 1))
    if H is None:
        continue
    H = H[np.abs(H[:, 0] - (tr.t.values[i] + 1)) > 12]
    if len(H) < 5:
        continue
    ex = (H[:, 1] > req[j]).astype(float)
    out["plain"][j] = ex.mean()
    for bw in (0.25, 0.5, 1.0):
        w = np.exp(-np.abs(H[:, 2] - a0[j]) / bw)
        out[f"bw{bw}"][j] = (w * ex).sum() / w.sum()

yv = y[q]
m0 = a0 - THR
for zone, mask in [("dry", m0 < 0), ("onset", (m0 >= 0) & (m0 < .5)), ("all", np.ones(len(q), bool))]:
    ok = mask & ~np.isnan(out["plain"])
    s = " | ".join(f"{n} {max(auc(yv[ok], -v[ok]), 1 - auc(yv[ok], -v[ok])):.3f}" for n, v in out.items())
    print(f"[{zone:5s}] n={ok.sum():6d} univariate AUC: {s}")
