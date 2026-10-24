"""20 - Label-free climatological transition probability, screened before any model run.
For HUC h and calendar transition m -> m+1, collect historical month-to-month growth
g = log(ws(m+1)/ws(m)) from the lag columns of train-part rows (years within +-12 months of the
query target excluded). Required growth to stay above the threshold: req = log(0.6*clim_next/ws0).
Feature: P(g > req) over the history (recovery / survival probability), plus the growth spread.
Screen: univariate AUC inside the dry and onset zones vs the existing margin_raw_persist and seas_step."""
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

h = importlib.import_module("13_v2_cv")
FOLDS, y, tr = h.FOLDS, h.y, h.tr
THR = np.log(0.6)
trn, va, uns = FOLDS[1]
part = ~va & ~uns
F = h.fold_features(1)

# history of growths from train-part rows, deduplicated by absolute destination month
rows = tr[part]
g = []
for k in range(11):
    a, b = rows[f"q_ws_cum_m3_lag{k}"].abs(), rows[f"q_ws_cum_m3_lag{k + 1}"].abs()
    g.append(pd.DataFrame({"id": rows.id.values, "T": rows.t.values - k, "cm": (rows.month.values - 1 - k) % 12 + 1,
                           "g": np.log((a.values + 1) / (b.values + 1))}))
G = pd.concat(g).groupby(["id", "T", "cm"]).g.median().reset_index()
hist = {key: grp[["T", "g"]].to_numpy() for key, grp in G.groupby(["id", "cm"])}

q = np.where(va)[0]
ws0 = np.exp(F.anom0.values[q])  # ws0/clim_now up to EPS; required growth uses margins directly
req = -(F.margin_raw_persist.values[q])  # log(0.6*clim_next/ws0)
p_exceed, spread = np.full(len(q), np.nan), np.full(len(q), np.nan)
cm_t = tr.month.values[q] % 12 + 1
for j, (i, hid, cm) in enumerate(zip(q, tr.id.values[q], cm_t)):
    H = hist.get((hid, cm))
    if H is None:
        continue
    H = H[np.abs(H[:, 0] - (tr.t.values[i] + 1)) > 12]
    if len(H) >= 3:
        p_exceed[j] = (H[:, 1] > req[j]).mean()
        spread[j] = H[:, 1].std()

yv = y[q]
m0 = F.anom0.values[q] - THR
for zone, mask in [("dry", m0 < 0), ("onset", (m0 >= 0) & (m0 < .5)), ("all", np.ones(len(q), bool))]:
    ok = mask & ~np.isnan(p_exceed)
    r = {"p_exceed (new)": auc(yv[ok], -p_exceed[ok]),
         "margin_raw_persist": auc(yv[ok], -F.margin_raw_persist.values[q][ok]),
         "seas_step": auc(yv[ok], -F.seas_step.values[q][ok]),
         "margin_dry": auc(yv[ok], -F.margin_dry.values[q][ok])}
    print(f"[{zone:5s}] n={ok.sum():6d} coverage {ok.sum() / mask.sum():.2f} | " +
          " | ".join(f"{k} AUC {max(v, 1 - v):.3f}" for k, v in r.items()))
