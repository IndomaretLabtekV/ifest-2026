"""Round 7 harness (34-39). Features are built ONCE in two views instead of once per fold:

  train view : seen HUCs only, original scale, climatology from those rows (what the model trains on)
  valid view : every HUC, volumes rescaled by a random per-HUC factor (11), climatology built only from
               origins in the test calendar months 2/4/10/12 (so each HUC knows 8 of 12 months, exactly
               like the test) and filled from the train table / kNN analogs.

This is exact for row-level features: climatology values are constants per (HUC, month), the rescale is
per HUC, and network aggregates are taken within one origin, so a row's features do not depend on which
other origins sit in its fold. Any validation scheme then reduces to choosing (fit rows, valid rows)."""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import CLIM, LAG_COLS, WORK, load_clean
from fe import basin_root, build, clim_table_split, dynamics, network
from prep import network_fill, repair

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

TEST_MONTHS = (2, 4, 10, 12)
DROP = ["origin_anom0", "origin_below0", "basin_size", "log_clim_next"]
DYN = ["ar1", "flashiness", "slope", "since_below", "amp_ratio", "use_trend"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, seed=42,
              num_threads=20, deterministic=True, force_row_wise=True)
# fixed rounds: no early stopping on the validation fold, so no scheme gets to peek at its own labels
ROUNDS = 300
T0 = pd.Timestamp("2000-09-01")  # t = 0, anchored in 34 (thermoelectric data starts at t = 88 = Jan 2008)


def load():
    tr, te = load_clean()
    tr = tr.drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
    tl = pd.read_csv(WORK / "timeline.csv").dropna(subset=["t"])
    tr = tr.merge(tl[["origin_id", "t"]], on="origin_id", how="left")
    tr["t"] = tr.t.astype(int)
    return tr, te


tr, te = load()
y, t = tr.label.values.astype(int), tr.t.values
year = t // 12  # Sep-Aug hydrological year index, 0 = Sep 2000 - Aug 2001
_ids = np.sort(tr.id.unique())
UNS = tr.id.isin(set(np.random.default_rng(42).choice(_ids, int(0.25 * len(_ids)), replace=False))).values
BASIN = basin_root(pd.concat([tr[["id", "to_id"]], te[["id", "to_id"]]]).drop_duplicates("id").set_index("id").to_id)


def basin_holdout(seed=42, frac=0.25):
    """44: the 786 test-only HUCs are 47 whole basins disjoint from the train basins, so the test's unseen part is
    a SPATIAL holdout. Hold out whole train basins (random order) until frac of the HUCs is reached."""
    b = BASIN.reindex(_ids)
    sizes = b.value_counts()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())
    take, n = [], 0
    for k in order:
        if n >= frac * len(_ids):
            break
        take.append(k)
        n += sizes[k]
    return tr.id.map(BASIN).isin(set(take)).values


UNS_BASIN = basin_holdout()
HOLDOUTS = {"huc": UNS, "basin": UNS_BASIN}


def rescale(df, seed=7):
    ids = df.id.unique()
    f = pd.Series(np.exp(np.random.default_rng(seed).uniform(np.log(.25), np.log(4), len(ids))), ids)
    out = df.copy()
    v = out[LAG_COLS + CLIM]
    # the test carries literal -9999 at the train rate: corruption was injected AFTER the rescale (38)
    out[LAG_COLS + CLIM] = v.mul(out.id.map(f).values, axis=0).where(v != -9999, -9999.0)
    return out


REPAIRS = {
    # V1-V6 repair: spike test before zero test, no storm check
    "v6": lambda d: repair(d, zero_first=False, storm=False),
    # 38: zero test first + budget-backed storm check + river-network water budget
    "v7": lambda d: network_fill(d, repair(d)),
}


def views(version="v6", holdout="huc"):
    """(train view indexed like tr[~uns], valid view indexed like tr) for one repair version and holdout. Cached."""
    tag = ("" if version == "v6" else f"_{version}") + ("" if holdout == "huc" else f"_{holdout}")
    uns = HOLDOUTS[holdout]
    pt, pv = WORK / f"r7{tag}_trainview.parquet", WORK / f"r7{tag}_validview.parquet"
    if pt.exists() and pv.exists():
        return pd.read_parquet(pt), pd.read_parquet(pv)
    net = network(tr, te)
    fix = REPAIRS[version]
    seen = tr[~uns]
    Ct = clim_table_split(seen)
    TV = build(fix(seen), Ct, net, do_repair=False, scale_free=True)
    TV = pd.concat([TV, dynamics(TV, seen)[DYN]], axis=1)
    sv = rescale(tr)
    Cv = clim_table_split(sv[sv.month.isin(TEST_MONTHS)], ref=Ct)
    VV = build(fix(sv), Cv, net, do_repair=False, scale_free=True)
    VV = pd.concat([VV, dynamics(VV, sv)[DYN]], axis=1)
    TV.to_parquet(pt); VV.to_parquet(pv)
    return TV, VV


def test_view(version="v7"):
    """Features of the real test (climatology from the test rows, filled from the full-train table),
    plus the full-train features, built exactly like the notebook. Cached."""
    pt, pe = WORK / f"r8_{version}_trainfull.parquet", WORK / f"r8_{version}_test.parquet"
    if pt.exists() and pe.exists():
        return pd.read_parquet(pt), pd.read_parquet(pe)
    net = network(tr, te)
    fix = REPAIRS[version]
    C = clim_table_split(tr)
    FT = build(fix(tr), C, net, do_repair=False, scale_free=True)
    FT = pd.concat([FT, dynamics(FT, tr)[DYN]], axis=1)
    tt = te.reset_index(drop=True)
    FE = build(fix(tt), clim_table_split(tt, ref=C), net, do_repair=False, scale_free=True)
    FE = pd.concat([FE, dynamics(FE, tt)[DYN]], axis=1)
    FT.to_parquet(pt); FE.to_parquet(pe)
    return FT, FE


def base_cols(F):
    return [c for c in F.columns if c not in DROP and c not in DYN]


def fit_predict(TV, VV, fit, va, cols, params=PARAMS, rounds=ROUNDS, weight=None, train_view_for_valid=False, uns=None):
    """fit / va are boolean masks over tr rows. Fit rows must be seen HUCs (they come from the train view)."""
    uns = UNS if uns is None else uns
    fit = fit & ~uns
    Xf = TV.loc[np.flatnonzero(fit), cols].to_numpy("float32")
    src = TV if train_view_for_valid else VV
    Xv = src.loc[np.flatnonzero(va), cols].to_numpy("float32")
    w = None if weight is None else weight[fit]
    m = lgb.train(params, lgb.Dataset(Xf, y[fit], weight=w), rounds)
    return m.predict(Xv)


def run_scheme(TV, VV, folds, cols, uns=None, **kw):
    """folds: list of (fit, va). Returns OOF prediction array over tr (NaN where never validated)
    and per-fold table."""
    uns = UNS if uns is None else uns
    oof = np.full(len(tr), np.nan)
    rows = []
    for i, (fit, va) in enumerate(folds):
        p = fit_predict(TV, VV, fit, va, cols, uns=uns, **kw)
        oof[va] = p
        u = uns[va]
        rows.append(dict(fold=i, n_fit=int((fit & ~uns).sum()), n_va=int(va.sum()), prev=y[va].mean(),
                         ap=ap(y[va], p), ap_seen=ap(y[va][~u], p[~u]) if (~u).any() else np.nan,
                         ap_unseen=ap(y[va][u], p[u]) if u.any() else np.nan))
    return oof, pd.DataFrame(rows)


def test_like(oof, n=400, seed=0, mask=None):
    """Pooled AP over draws of 4 validated origins with calendar months 2, 4, 10, 12 that are pairwise more
    than 12 months apart (the public test is exactly such a set). Returns the array of draw scores."""
    ok = ~np.isnan(oof) if mask is None else (~np.isnan(oof) & mask)
    d = pd.DataFrame({"t": t[ok], "m": tr.month.values[ok], "y": y[ok], "p": oof[ok]})
    by_m = {m: np.unique(d.t[d.m == m]) for m in TEST_MONTHS}
    grp = {k: g for k, g in d.groupby("t")}
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        pick = [rng.choice(by_m[m]) for m in TEST_MONTHS]
        if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
            continue
        s = pd.concat([grp[k] for k in pick])
        out.append(ap(s.y, s.p))
    return np.array(out)


def date_of(tt):
    return (T0 + pd.DateOffset(months=int(tt))).strftime("%Y-%m")


if __name__ == "__main__":
    import time
    s = time.time()
    TV, VV = views()
    print(f"views built in {time.time() - s:.0f}s: train view {TV.shape}, valid view {VV.shape}, unseen share {UNS.mean():.3f}")
    assert len(VV) == len(tr) and len(TV) == (~UNS).sum()
    assert date_of(88) == "2008-01" and date_of(0) == "2000-09"
    # seen rows: valid view anomalies must equal train view anomalies up to the kNN/transfer fill (rescale-free)
    idx = np.flatnonzero(~UNS)[:5000]
    d = (TV.loc[idx, "anom0"] - VV.loc[idx, "anom0"]).abs()
    print(f"seen rows |anom0 train view - valid view| median {d.median():.4f} p95 {d.quantile(.95):.4f}")
    print("ok")
