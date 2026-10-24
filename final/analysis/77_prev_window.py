"""77 - The chain also gives three test origins a PREVIOUS window (14 months earlier: months t-25 .. t-14), i.e. two
years of own history instead of one. Useful mainly for the 786 new HUCs, which have no train history (the transition
features need 5 years). Judged like 68 (leave-one-year-out + basin holdout, purge 26) on rows that have both windows:
  nx     : + next window (the V16 feature set)
  nx+pv  : + previous window anomalies pv_anom0..11"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from cvkit import HOLDOUTS, PARAMS, base_cols, t, tr, views, y, year

uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
V13 = {**PARAMS, "learning_rate": 0.0273, "num_leaves": 53, "min_data_in_leaf": 91, "feature_fraction": 0.328,
       "bagging_fraction": 0.822, "lambda_l2": 9.27, "extra_trees": True}


def shift(F, gap, prefix):
    idx = F.index.to_numpy()
    src = pd.DataFrame(F[[f"anom{k}" for k in range(12)]].to_numpy(), index=pd.MultiIndex.from_arrays([tr.id.values[idx], t[idx]]))
    out = src.reindex(pd.MultiIndex.from_arrays([tr.id.values[idx], t[idx] + gap])).to_numpy()
    return pd.DataFrame(out, index=F.index, columns=[f"{prefix}_anom{k}" for k in range(12)])


NX, PV = [f"nx_anom{k}" for k in range(12)], [f"pv_anom{k}" for k in range(12)]
TV = pd.concat([TV, shift(TV, 14, "nx"), shift(TV, -14, "pv")], axis=1)
VV = pd.concat([VV, shift(VV, 14, "nx"), shift(VV, -14, "pv")], axis=1)
full = VV.reindex(np.arange(len(tr)))
both = np.isfinite(full.nx_anom11.to_numpy()) & np.isfinite(full.pv_anom0.to_numpy())
res = {"nx": [], "nx+pv": []}
res_new = {"nx": [], "nx+pv": []}
for k in range(1, 13):
    fit = ((t < 12 * k - 12 - 26) | (t >= 12 * k + 12 + 26)) & ~uns
    va = (year == k) & both
    f, v = np.flatnonzero(fit), np.flatnonzero(va)
    fh = f[np.isfinite(TV.loc[f, "nx_anom11"].to_numpy()) & np.isfinite(TV.loc[f, "pv_anom0"].to_numpy())]
    for name, c in [("nx", cols + NX), ("nx+pv", cols + NX + PV)]:
        p = lgb.train(V13, lgb.Dataset(TV.loc[fh, c].to_numpy("float32"), y[fh]), 549).predict(VV.loc[v, c].to_numpy("float32"))
        res[name].append(ap(y[va], p))
        u = uns[va]
        res_new[name].append(ap(y[va][u], p[u]))
    print(f"  year {k:2d}: nx {res['nx'][-1]:.4f} nx+pv {res['nx+pv'][-1]:.4f} | new basins {res_new['nx'][-1]:.4f} -> {res_new['nx+pv'][-1]:.4f}", flush=True)
d, dn = np.array(res["nx+pv"]) - np.array(res["nx"]), np.array(res_new["nx+pv"]) - np.array(res_new["nx"])
print(f"\n[77] previous window on top of next window: {d.mean():+.4f} (up {(d > 0).sum()}/12); new basins {dn.mean():+.4f} (up {(dn > 0).sum()}/12)")
