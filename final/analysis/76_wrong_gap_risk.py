"""76 - Risk if a chain link is misdated. 74 leaves links 169 -> 94 and 94 -> 81 at 14 months with moderate
confidence (26 not excluded). If the truth were 26 months, the next-window features fed to the gap-14 model would come
from months target+14 .. target+25. How much would that cost versus not using the features at all?
Leave-one-year-out + basin holdout, purge 26 (as 68); the model is trained with gap-14 windows, then scored on the
same validation rows with (a) the right gap-14 windows, (b) wrong gap-26 windows, (c) no next window (base model)."""
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
NX = [f"nx_anom{k}" for k in range(12)]


def nxt(F, gap):
    idx = F.index.to_numpy()
    src = pd.DataFrame(F[[f"anom{k}" for k in range(12)]].to_numpy(), index=pd.MultiIndex.from_arrays([tr.id.values[idx], t[idx]]))
    return pd.DataFrame(src.reindex(pd.MultiIndex.from_arrays([tr.id.values[idx], t[idx] + gap])).to_numpy(), index=F.index, columns=NX)


TN = pd.concat([TV, nxt(TV, 14)], axis=1)
V14, V26 = pd.concat([VV, nxt(VV, 14)], axis=1), pd.concat([VV, nxt(VV, 26)], axis=1)
both = np.isfinite(V26.reindex(np.arange(len(tr))).nx_anom11.to_numpy())
res = {k: [] for k in ["base", "right gap 14", "wrong gap 26"]}
for k in range(12):
    fit = ((t < 12 * k - 12 - 26) | (t >= 12 * k + 12 + 26)) & ~uns
    va = (year == k) & both
    f, v = np.flatnonzero(fit), np.flatnonzero(va)
    fh = f[np.isfinite(TN.loc[f, "nx_anom11"].to_numpy())]
    mb = lgb.train(V13, lgb.Dataset(TV.loc[f, cols].to_numpy("float32"), y[f]), 549)
    mn = lgb.train(V13, lgb.Dataset(TN.loc[fh, cols + NX].to_numpy("float32"), y[fh]), 549)
    res["base"].append(ap(y[va], mb.predict(VV.loc[v, cols].to_numpy("float32"))))
    res["right gap 14"].append(ap(y[va], mn.predict(V14.loc[v, cols + NX].to_numpy("float32"))))
    res["wrong gap 26"].append(ap(y[va], mn.predict(V26.loc[v, cols + NX].to_numpy("float32"))))
    print(f"  year {k:2d}: " + " | ".join(f"{n} {res[n][-1]:.4f}" for n in res), flush=True)
R = pd.DataFrame(res)
b = R["base"]
print("\n[76] mean AP over 12 years on rows that have both windows:")
for n in R:
    print(f"  {n:14s} {R[n].mean():.4f} ({(R[n] - b).mean():+.4f} vs base, up {(R[n] > b).sum()}/12)")
