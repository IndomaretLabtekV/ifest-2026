"""69 - Richer next-window features and robustness over three whole-basin splits.
68: the next origin's 12 anomalies (months target+2 .. target+13) add +0.035 AP (13/13 years). Rain is regional, so
the basin mean of the next window's closest months should denoise the single-HUC signal (the same reason basin_anom0
helps). Variants, all with the 26-month purge of 68:
  base        : V13 feature set
  nx          : + nx_anom0..11
  nx+basin    : + basin/downstream means of the closest next-window months (nx_anom11, nx_anom10, nx_anom9)
Each on basin splits 42, 1, 7; gain reported per year on rows that have a next window."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import WORK
from cvkit import BASIN, HOLDOUTS, PARAMS, base_cols, basin_holdout, t, tr, views, y, year

GAP = 14
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 1, 7]
V13 = {**PARAMS, "learning_rate": 0.0273, "num_leaves": 53, "min_data_in_leaf": 91, "feature_fraction": 0.328,
       "bagging_fraction": 0.822, "lambda_l2": 9.27, "extra_trees": True}
NX = [f"nx_anom{k}" for k in range(12)]
NXB = [f"nx_basin_anom{k}" for k in (11, 10, 9)] + [f"nx_down_anom{k}" for k in (11, 10)]
to = tr.to_id.astype(str).str.strip().str.lower().values


def add_next(F):
    idx = F.index.to_numpy()
    ids, tt = tr.id.values[idx], t[idx]
    src = pd.DataFrame(F[[f"anom{k}" for k in range(12)]].to_numpy(), index=pd.MultiIndex.from_arrays([ids, tt]))
    nxt = pd.DataFrame(src.reindex(pd.MultiIndex.from_arrays([ids, tt + GAP])).to_numpy(), index=F.index, columns=NX)
    key = pd.DataFrame({"basin": pd.Series(ids).map(BASIN).values, "t": tt}, index=F.index)
    out = [F, nxt]
    for k in (11, 10, 9):
        out.append(nxt[f"nx_anom{k}"].groupby([key.basin, key.t]).transform("mean").rename(f"nx_basin_anom{k}"))
    own = pd.DataFrame(nxt[["nx_anom11", "nx_anom10"]].to_numpy(), index=pd.MultiIndex.from_arrays([ids, tt]))
    down = own.reindex(pd.MultiIndex.from_arrays([to[idx], tt])).to_numpy()
    out.append(pd.DataFrame(down, index=F.index, columns=["nx_down_anom11", "nx_down_anom10"]))
    return pd.concat(out, axis=1)


rows = []
for s in SEEDS:
    name = "basin" if s == 42 else f"basin{s}"
    if s != 42:
        HOLDOUTS[name] = basin_holdout(seed=s)
    uns = HOLDOUTS[name]
    TV, VV = views("v7", name)
    cols = base_cols(TV)
    TV, VV = add_next(TV), add_next(VV)
    has = np.isfinite(VV.reindex(np.arange(len(tr)))["nx_anom11"].to_numpy())
    VAR = {"base": cols, "nx": cols + NX, "nx+basin": cols + NX + NXB}
    oof = {v: np.full(len(tr), np.nan) for v in VAR}
    for k in range(13):
        fit = ((t < 12 * k - 12 - GAP) | (t >= 12 * k + 12 + GAP)) & ~uns
        va = (year == k) & has
        f, v = np.flatnonzero(fit), np.flatnonzero(va)
        fh = f[np.isfinite(TV.loc[f, "nx_anom11"].to_numpy())]
        for vn, c in VAR.items():
            ff = f if vn == "base" else fh
            m = lgb.train(V13, lgb.Dataset(TV.loc[ff, c].to_numpy("float32"), y[ff]), 549)
            oof[vn][va] = m.predict(VV.loc[v, c].to_numpy("float32"))
    pd.DataFrame(oof).to_parquet(WORK / f"69_oof_s{s}.parquet")
    per = {vn: np.array([ap(y[(year == k) & has], oof[vn][(year == k) & has]) for k in range(13)]) for vn in VAR}
    pn = {vn: np.array([ap(y[(year == k) & has & uns], oof[vn][(year == k) & has & uns]) for k in range(13)]) for vn in VAR}
    for vn in VAR:
        d = per[vn] - per["base"]
        rows.append(dict(seed=s, variant=vn, ap=per[vn].mean(), gain=d.mean(), up=int((d > 0).sum()),
                         newbasin=pn[vn].mean(), gain_new=(pn[vn] - pn["base"]).mean()))
        print(f"  seed {s} {vn:9s} AP {per[vn].mean():.4f} gain {d.mean():+.4f} up {(d > 0).sum()}/13 | "
              f"new basins {pn[vn].mean():.4f} ({(pn[vn] - pn['base']).mean():+.4f})", flush=True)
    d = per["nx+basin"] - per["nx"]
    print(f"  seed {s}: basin aggregates on top of nx {d.mean():+.4f} (up {(d > 0).sum()}/13)", flush=True)
R = pd.DataFrame(rows)
R.to_csv(WORK / "69_next_window_rich.csv", index=False)
print("\n[69] summary over basin splits:")
print(R.pivot(index="variant", columns="seed", values="gain").round(4).to_string())
