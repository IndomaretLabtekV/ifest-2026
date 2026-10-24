"""12 - Does the per-HUC rescaling found in 11 explain public 0.423 vs CV 0.703?
On every CV fold, multiply all volume columns (72 lags + 2 clim) of the validation rows by a random
per-HUC factor (log-uniform 0.25..4, the shape measured in 11), then score:
  A  V1 pipeline, no rescale            (reproduces the V1 CV)
  B  V1 pipeline, rescaled validation   (what the test actually looked like to V1)
  C  split-aware + scale-free pipeline, rescaled validation (the V2 fix)
HUCs held out as 'unseen' are removed from the train part entirely, like test-only HUCs."""
import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import CLIM, LAG_COLS, WORK, load_clean
from fe import build, clim_table, clim_table_split, network

cv = importlib.import_module("07_cv_ablation")
FOLDS, y = cv.FOLDS, cv.y
tr = cv.tr.copy()
_, te = load_clean()
net = network(tr, te)
VOL = LAG_COLS + CLIM
PARAMS = {**cv.PARAMS, "learning_rate": 0.05}
DROP = ["origin_anom0", "origin_below0", "basin_size"]


def rescale(df, rows, seed):
    out = df.copy()
    ids = out.id[rows].unique()
    f = pd.Series(np.exp(np.random.default_rng(seed).uniform(np.log(.25), np.log(4), len(ids))), ids)
    out.loc[rows, VOL] = out.loc[rows, VOL].mul(out.id[rows].map(f).values, axis=0)
    return out


def fit_score(F, trn, va, uns):
    X = F.drop(columns=[c for c in DROP if c in F]).to_numpy("float32")
    m = lgb.train(PARAMS, lgb.Dataset(X[trn], y[trn]), 300)
    p = m.predict(X[va])
    return ap(y[va], p), ap(y[va][~uns[va]], p[~uns[va]]), ap(y[va][uns[va]], p[uns[va]]), ap(y[va], -F.anom0.fillna(0).values[va])


rows = []
for i, (trn, va, uns) in enumerate(FOLDS):
    keep = trn | va
    # V1: one pooled climatology over every row
    FA = build(tr, clim_table(tr), net)
    sc = rescale(tr, va, seed=i)
    FB = build(sc, clim_table(sc), net)
    # V2: climatology per split; train part excludes held-out HUCs entirely
    part_tr = ~va & ~uns
    Ct = clim_table_split(tr[part_tr])
    FC = pd.concat([build(tr[part_tr], Ct, net, scale_free=True),
                    build(sc[va], clim_table_split(sc[va], ref=Ct), net, scale_free=True)]).reindex(tr.index)
    for name, F in [("A V1, no rescale", FA), ("B V1, rescaled valid", FB), ("C V2 fix, rescaled valid", FC)]:
        r = fit_score(F, trn & part_tr if name.startswith("C") else trn, va, uns)
        rows.append({"fold": i, "variant": name, "AP": r[0], "seen": r[1], "unseen": r[2], "persistence": r[3]})
        print(f"fold {i} {name:26s} AP {r[0]:.4f} seen {r[1]:.4f} unseen {r[2]:.4f} | persistence {r[3]:.4f}", flush=True)

R = pd.DataFrame(rows)
print(R.groupby("variant")[["AP", "seen", "unseen", "persistence"]].mean().round(4).to_string())
R.to_csv(WORK / "12_rescale_simulation.csv", index=False)
