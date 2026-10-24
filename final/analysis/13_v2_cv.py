"""13 - V2 validation harness. Builds features per fold exactly like the test is built:
validation rows get a random per-HUC volume rescale (found in 11), climatology is built per split,
held-out HUCs never appear in the train part. Features are cached so model experiments are cheap.
Reports block AP plus a test-like AP: pooled AP over random sets of 4 validation origins whose
calendar months are 2, 4, 10, 12 (the public test is exactly 4 such origins)."""
import importlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import CLIM, LAG_COLS, WORK, load_clean
from fe import build, clim_table_split, network

cv = importlib.import_module("07_cv_ablation")
FOLDS = cv.FOLDS
tr, te = load_clean()
tr = tr.drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
tl = pd.read_csv(WORK / "timeline.csv").dropna(subset=["t"])
tr = tr.merge(tl[["origin_id", "t"]], on="origin_id", how="left")
assert (tr.id.values == cv.ids).all()
y, t = tr.label.values, tr.t.values
net = network(tr, te)
VOL = LAG_COLS + CLIM
DROP = ["origin_anom0", "origin_below0", "basin_size"]
PARAMS = cv.PARAMS


def rescale(df, seed):
    ids = df.id.unique()
    f = pd.Series(np.exp(np.random.default_rng(seed).uniform(np.log(.25), np.log(4), len(ids))), ids)
    out = df.copy()
    out[VOL] = out[VOL].mul(out.id.map(f).values, axis=0)
    return out


def fold_features(i, extra=False):
    path = WORK / f"v2{'x' if extra else ''}_fold{i}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    trn, va, uns = FOLDS[i]
    part_tr = ~va & ~uns
    Ct = clim_table_split(tr[part_tr])
    sv = rescale(tr[va], seed=100 + i)
    F = pd.concat([build(tr[part_tr], Ct, net, scale_free=True, extra=extra),
                   build(sv, clim_table_split(sv, ref=Ct), net, scale_free=True, extra=extra)]).reindex(tr.index)
    F.to_parquet(path)
    return F


def test_like_ap(p, va, n_sets=200, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"o": tr.origin_id.values[va], "m": tr.month.values[va], "y": y[va], "p": p})
    by_m = {m: d.o[d.m == m].unique() for m in (2, 4, 10, 12)}
    scores = []
    for _ in range(n_sets):
        pick = [rng.choice(by_m[m]) for m in by_m if len(by_m[m])]
        s = d[d.o.isin(pick)]
        scores.append(ap(s.y, s.p))
    return float(np.mean(scores)), float(np.percentile(scores, 10)), float(np.percentile(scores, 90))


def evaluate(name, preds):
    rows = []
    for (trn, va, uns), p in zip(FOLDS, preds):
        tl_mean, tl_lo, tl_hi = test_like_ap(p, va)
        rows.append((ap(y[va], p), ap(y[va][~uns[va]], p[~uns[va]]), ap(y[va][uns[va]], p[uns[va]]), tl_mean, tl_lo, tl_hi))
    r = np.array(rows)
    print(f"{name:34s} AP {r[:, 0].mean():.4f} | seen {r[:, 1].mean():.4f} unseen {r[:, 2].mean():.4f} | "
          f"test-like 4-origin AP {r[:, 3].mean():.4f} (p10 {r[:, 4].mean():.3f}, p90 {r[:, 5].mean():.3f}) | "
          f"per-fold {np.round(r[:, 0], 4).tolist()}", flush=True)
    return r


def lgb_preds(cols_fn=lambda F: [c for c in F.columns if c not in DROP], params=PARAMS, extra=False, weight_fn=None):
    preds, iters = [], []
    for i, (trn, va, uns) in enumerate(FOLDS):
        F = fold_features(i, extra)
        cols = cols_fn(F)
        X = F[cols].to_numpy("float32")
        fit = trn & ~uns & ~va
        w = weight_fn(fit) if weight_fn else None
        m = lgb.train(params, lgb.Dataset(X[fit], y[fit], weight=w), 5000, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        preds.append(m.predict(X[va], num_iteration=m.best_iteration)); iters.append(m.best_iteration)
    print("   iters", iters)
    return preds


if __name__ == "__main__":
    for i in range(len(FOLDS)):
        fold_features(i)
    which = sys.argv[1:] or ["persist", "v2", "v2-basin"]
    if "persist" in which:
        evaluate("persistence (-anom0)", [-fold_features(i).anom0.fillna(0).values[va] for i, (_, va, _) in enumerate(FOLDS)])
    if "v2" in which:
        evaluate("V2 LGB (E4 set, scale-free)", lgb_preds())
    if "v2-basin" in which:
        basin = lambda F: [c for c in F.columns if c not in DROP and not c.startswith(("basin_", "down_", "up_anom", "anom0_vs"))]
        evaluate("V2 LGB without basin/network", lgb_preds(basin))
