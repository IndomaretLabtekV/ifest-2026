"""Feature engineering (ported verbatim into the Kaggle notebook).
Label = 1  <=>  ws_cum(t+1) <= 0.6 * clim(t+1)   (reverse-engineered in 03, 99.99% exact).
So every feature is framed as 'where is supply now vs its climatology, and where is it heading'."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from prep import clim_scale, repair

THR = np.log(0.6)
EPS = 1.0


def clim_table(df: pd.DataFrame) -> pd.DataFrame:
    """id x calendar-month climatology (scale-corrected), pooled from clim_now and clim_next of all rows.
    Uses only feature columns, so pooling train+test is transductive but label-free."""
    e = 10.0 ** clim_scale(df)
    a = pd.DataFrame({"id": df.id, "m": df.month, "c": df.clim_median_now * e})
    b = pd.DataFrame({"id": df.id, "m": df.month % 12 + 1, "c": df.clim_median_next * e})
    C = pd.concat([a, b]).groupby(["id", "m"]).c.median().unstack().reindex(columns=range(1, 13))
    return fill_clim_knn(C)


def fill_clim_knn(C: pd.DataFrame, k: int = 15, ref: pd.DataFrame = None) -> pd.DataFrame:
    """HUCs seen only at test origins lack Jun-Sep climatology. Borrow the seasonal shape of the k
    complete HUCs with the closest shape on the known months (06: median |log err| 0.13 vs 0.27 cyclic)."""
    ref = np.log1p((C if ref is None else ref).dropna())
    out = C.copy()
    for pattern, grp in C[C.isna().any(axis=1)].groupby(lambda i: tuple(C.loc[i].isna())):
        known = [m for m, miss in zip(C.columns, pattern) if not miss]
        if len(known) < 3:
            continue
        R = ref.sub(ref[known].mean(1), axis=0)
        Q = np.log1p(grp[known])
        lvl = Q.mean(1)
        _, idx = NearestNeighbors(n_neighbors=k).fit(R[known].values).kneighbors(Q.sub(lvl, axis=0).values)
        est = pd.DataFrame(np.expm1(R.values[idx].mean(1) + lvl.values[:, None]), grp.index, C.columns)
        out.loc[grp.index] = grp.fillna(est)
    return out


def clim_table_split(df: pd.DataFrame, ref: pd.DataFrame = None) -> pd.DataFrame:
    """Climatology built from ONE split only (the test rescales every HUC's volumes by a random
    per-HUC factor, found in 11, so train and test climatologies must never be pooled).
    Missing months of a HUC that also exists in `ref` (the train table) are filled with the train
    seasonal shape times the HUC's factor, which is exact because the factor is constant per HUC."""
    e = 10.0 ** clim_scale(df)
    a = pd.DataFrame({"id": df.id, "m": df.month, "c": df.clim_median_now * e})
    b = pd.DataFrame({"id": df.id, "m": df.month % 12 + 1, "c": df.clim_median_next * e})
    C = pd.concat([a, b]).groupby(["id", "m"]).c.median().unstack().reindex(columns=range(1, 13))
    if ref is not None:
        both = C.index.intersection(ref.dropna().index)
        R = ref.loc[both]
        k = np.log(C.loc[both] / R).median(axis=1)
        C.loc[both] = C.loc[both].fillna(R.mul(np.exp(k), axis=0))
        return fill_clim_knn(C, ref=ref)
    return fill_clim_knn(C)


def basin_root(id_to: pd.Series) -> pd.Series:
    """Follow to_id downstream until the chain leaves the dataset -> terminal HUC = basin id."""
    nxt = id_to.to_dict()
    root = {}
    for h in nxt:
        path, cur = [], h
        while cur in nxt and cur not in root and cur not in path and nxt[cur] in nxt:
            path.append(cur); cur = nxt[cur]
        r = root.get(cur, cur)
        for p in path:
            root[p] = r
        root.setdefault(h, r)
    return pd.Series(root)


def build(df: pd.DataFrame, C: pd.DataFrame, net: pd.DataFrame, do_repair=True, scale_free=False, extra=False) -> pd.DataFrame:
    x = repair(df) if do_repair else df.copy()
    # aux_fake_mm_* = inc_aream2 / 1e4 (+-0.3%): the one aux family worth keeping, as a backup copy of the area
    inc = df.inc_aream2.abs().fillna(df[["aux_fake_mm_04", "aux_fake_mm_09", "aux_fake_mm_14"]].median(axis=1) * 1e4)
    cum = df.cum_aream2.abs().fillna(inc)
    L = lambda f, k: x[f"{f}_lag{k}"].clip(lower=0)
    F = pd.DataFrame(index=df.index)
    F["month"] = df.month
    F["m_sin"], F["m_cos"] = np.sin(2 * np.pi * df.month / 12), np.cos(2 * np.pi * df.month / 12)

    # static
    F["log_inc_area"], F["log_cum_area"] = np.log1p(inc), np.log1p(cum)
    F["headwater"] = np.isclose(inc, cum, rtol=1e-6).astype(int)
    F["log_pop_dens"] = np.log1p(df.pop_acs_2020.abs() / (inc / 1e6 + 1e-3))
    F["pop_growth"] = np.log1p(df.pop_dec_2020.abs()) - np.log1p(df.pop_dec_2000.abs())

    # climatology for every lag month (lag k is calendar month month-k)
    ci = C.index.get_indexer(df.id)
    Cm = C.to_numpy()
    def clim_at(month):
        v = Cm[ci, month - 1]
        return np.where(ci >= 0, v, np.nan)
    clim_now, clim_next = clim_at(df.month.values), clim_at(df.month.values % 12 + 1)
    F["log_clim_next"] = np.log1p(clim_next)
    F["seas_step"] = np.log((clim_next + EPS) / (clim_now + EPS))
    F["clim_cv"] = np.nanstd(np.log1p(Cm), axis=1)[ci]

    A = np.column_stack([np.log((L("q_ws_cum_m3", k) + EPS) / (clim_at((df.month.values - 1 - k) % 12 + 1) + EPS))
                         for k in range(12)])
    for k in range(12):
        F[f"anom{k}"] = A[:, k]
    F["anom_mean3"], F["anom_mean6"], F["anom_mean12"] = A[:, :3].mean(1), A[:, :6].mean(1), A.mean(1)
    F["anom_min6"], F["anom_std12"] = A[:, :6].min(1), A.std(1)
    F["n_below12"] = (A < THR).sum(1)
    run = np.zeros(len(A))
    alive = np.ones(len(A), bool)
    for k in range(12):
        alive &= A[:, k] < THR
        run += alive
    F["drought_run"] = run

    ws0, bf0, qf0 = L("q_ws_cum_m3", 0), L("q_bf_inc_m3", 0), L("q_qf_inc_m3", 0)
    up0 = (ws0 - bf0 - qf0).clip(lower=0)
    # persistence forecasts of next month's ratio, expressed as margin to the 0.6 threshold
    F["margin_anom_persist"] = A[:, 0] - THR
    F["margin_raw_persist"] = np.log((ws0 + EPS) / (clim_next + EPS)) - THR
    F["margin_lastyear"] = A[:, 11] - THR
    # dry scenario: no quickflow next month, baseflow recedes at its current rate
    rec = ((L("q_bf_inc_m3", 0) + EPS) / (L("q_bf_inc_m3", 1) + EPS)).clip(0.2, 1.5)
    F["bf_recession"] = np.log(rec)
    F["margin_dry"] = np.log((bf0 * rec + up0 + EPS) / (clim_next + EPS)) - THR
    F["qf_share0"] = qf0 / (ws0 + EPS)
    F["up_share0"] = up0 / (ws0 + EPS)
    F["bf_share0"] = bf0 / (ws0 + EPS)
    for k in (1, 2, 3):
        F[f"ws_trend{k}"] = np.log((ws0 + EPS) / (L("q_ws_cum_m3", k) + EPS))
    F["bf_trend1"] = np.log((bf0 + EPS) / (L("q_bf_inc_m3", 1) + EPS))
    F["qf_mean3_share"] = sum(L("q_qf_inc_m3", k) for k in range(3)) / (sum(L("q_ws_cum_m3", k) for k in range(3)) + EPS)

    use0 = L("q_ir_wd_inc_m3", 0) + L("q_th_wd_inc_m3", 0) + L("q_ps_wd_inc_m3", 0)
    F["use_ratio0"] = np.log1p(use0 / (ws0 + EPS))
    F["use_clim_next"] = np.log1p(use0 / (clim_next + EPS))
    F["ir_share"] = L("q_ir_wd_inc_m3", 0) / (use0 + EPS)

    # river network, cross-sectional at the same origin (no future information)
    key = pd.DataFrame({"id": df.id.values, "origin_id": df.origin_id.values, "a0": A[:, 0], "below": A[:, 0] < THR})
    key["to_id"] = df.to_id.values
    key["basin"] = key.id.map(net["basin"])
    a0 = key.drop_duplicates(["id", "origin_id"]).set_index(["id", "origin_id"]).a0
    F["down_anom0"] = a0.reindex(pd.MultiIndex.from_arrays([key.to_id, key.origin_id])).to_numpy()
    upm = key.groupby(["to_id", "origin_id"]).a0.mean()
    F["up_anom0"] = upm.reindex(pd.MultiIndex.from_arrays([key.id, key.origin_id])).to_numpy()
    g = key.groupby(["basin", "origin_id"])
    F["basin_anom0"] = g.a0.transform("mean").to_numpy()
    F["basin_below0"] = g.below.transform("mean").to_numpy()
    F["basin_size"] = g.a0.transform("size").to_numpy()
    go = key.groupby("origin_id")
    F["origin_anom0"] = go.a0.transform("mean").to_numpy()
    F["origin_below0"] = go.below.transform("mean").to_numpy()
    F["anom0_vs_basin"] = A[:, 0] - F["basin_anom0"]
    if extra:
        # baseflow = groundwater memory: its share of normal supply bounds next month from below
        cn, cx = clim_now + EPS, clim_next + EPS
        F["bf_anom0"] = np.log((bf0 + EPS) / cn)
        F["bf_anom_next_dry"] = np.log((bf0 * rec + EPS) / cx)
        F["bf_anom11"] = np.log((L("q_bf_inc_m3", 11) + EPS) / cx)
        F["qf_anom0"] = np.log((qf0 + EPS) / cn)
        F["up_anom_now"] = np.log((up0 + EPS) / cn)
        # anomaly trend with the seasonal cycle removed
        F["anom_slope1"], F["anom_slope2"] = A[:, 0] - A[:, 1], A[:, 0] - A[:, 2]
        F["anom_accel"] = (A[:, 0] - A[:, 1]) - (A[:, 1] - A[:, 2])
        # richer regional state at the same origin
        key["m3"], key["dry"] = A[:, :3].mean(1), F["margin_dry"].values
        key["run"] = F["drought_run"].values
        g = key.groupby(["basin", "origin_id"])
        F["basin_anom_mean3"] = g.m3.transform("mean").to_numpy()
        F["basin_margin_dry"] = g.dry.transform("mean").to_numpy()
        F["basin_drought_run"] = g.run.transform("mean").to_numpy()
        dry = key.drop_duplicates(["id", "origin_id"]).set_index(["id", "origin_id"]).dry
        F["down_margin_dry"] = dry.reindex(pd.MultiIndex.from_arrays([key.to_id, key.origin_id])).to_numpy()
    if scale_free:
        # absolute volume levels are randomised per HUC in the test -> keep only ratios
        F = F.drop(columns=["log_clim_next"])
    return F.replace([np.inf, -np.inf], np.nan).astype("float32")


def network(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    m = pd.concat([tr[["id", "to_id"]], te[["id", "to_id"]]]).drop_duplicates("id").set_index("id").to_id
    return pd.DataFrame({"basin": basin_root(m)})


if __name__ == "__main__":
    r = basin_root(pd.Series({"a": "b", "b": "c", "c": "OUTLET", "d": "c", "e": "zz"}))
    assert r.to_dict() == {"a": "c", "b": "c", "c": "c", "d": "c", "e": "e"}, r.to_dict()
    print("ok")


def transition_features(query: pd.DataFrame, history: pd.DataFrame, min_hist: int = 3, max_years: int = None,
                        seed: int = 0) -> pd.DataFrame:
    """Label-free climatological transition probability (screened in 20).
    history rows contribute month-to-month growths g = log(ws(m+1)/ws(m)) from their lag columns,
    keyed by (HUC, destination calendar month, absolute destination time T). For a query row the growths
    into its target calendar month are used, excluding |T - (t+1)| <= 12 (the query's own year).
    Both frames need columns id, t, month and the ws lags; volumes may be rescaled per HUC (ratios only)."""
    parts = []
    for k in range(11):
        a, b = history[f"q_ws_cum_m3_lag{k}"].abs().values, history[f"q_ws_cum_m3_lag{k + 1}"].abs().values
        parts.append(pd.DataFrame({"id": history.id.values, "T": history.t.values - k,
                                   "cm": (history.month.values - 1 - k) % 12 + 1, "g": np.log((a + 1) / (b + 1))}))
    G = pd.concat(parts).dropna().groupby(["id", "cm", "T"]).g.median().reset_index()
    hist = {key: grp[["T", "g"]].to_numpy() for key, grp in G.groupby(["id", "cm"])}
    ws0 = query.q_ws_cum_m3_lag0.abs().values
    clim_next = query["_clim_next"].values
    req = np.log((0.6 * clim_next + 1) / (ws0 + 1))
    out = np.full((len(query), 4), np.nan)
    rng = np.random.default_rng(seed)
    for j, (hid, cm, t) in enumerate(zip(query.id.values, query.month.values % 12 + 1, query.t.values)):
        H = hist.get((hid, cm))
        if H is None:
            continue
        H = H[np.abs(H[:, 0] - (t + 1)) > 12, 1]
        if max_years is not None and len(H) > max_years:
            # emulate a HUC that only has `max_years` years of history (test-only HUCs have exactly 3)
            H = rng.choice(H, max_years, replace=False)
        if len(H) >= min_hist:
            out[j] = [(H > req[j]).mean(), np.median(H) - req[j], H.std(), len(H)]
    return pd.DataFrame(out, index=query.index, columns=["trans_p_exceed", "trans_margin_med", "trans_g_std", "trans_n"])


def dynamics(F: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Catchment dynamics from the row's own 12 anomalies (31): memory, flashiness, trend, recovery
    timing. Row-internal and scale-free, so it also applies to test-only HUCs."""
    A = F[[f"anom{k}" for k in range(12)]].to_numpy("float64")
    A = np.where(np.isfinite(A), A, np.nan)
    mu = np.nanmean(A, axis=1, keepdims=True)
    C = A - mu
    num = np.nansum(C[:, :-1] * C[:, 1:], axis=1)
    den = np.nansum(C * C, axis=1)
    ar1 = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    d = np.abs(np.diff(A, axis=1))
    spread = np.nanmax(A, axis=1) - np.nanmin(A, axis=1)
    flash = np.nanmean(d, axis=1) / np.maximum(spread, 1e-6)
    k = np.arange(12)
    kk = k - k.mean()
    slope = np.nansum(C * kk, axis=1) / (kk ** 2).sum()
    below = A < THR
    since = np.where(below.any(axis=1), below.argmax(axis=1), 12)
    x = repair(df)
    L = lambda f, j: x[f"{f}_lag{j}"].abs().to_numpy()
    use0 = L("q_ir_wd_inc_m3", 0) + L("q_th_wd_inc_m3", 0) + L("q_ps_wd_inc_m3", 0)
    use3 = L("q_ir_wd_inc_m3", 3) + L("q_th_wd_inc_m3", 3) + L("q_ps_wd_inc_m3", 3)
    use_trend = np.log((use0 + 1) / (use3 + 1))
    th_share = L("q_th_wd_inc_m3", 0) / (use0 + 1)
    ps_share = L("q_ps_wd_inc_m3", 0) / (use0 + 1)
    return pd.DataFrame({"ar1": ar1, "flashiness": flash, "slope": slope, "since_below": since,
                         "amp_ratio": spread, "use_trend": use_trend, "th_share": th_share,
                         "ps_share": ps_share}, index=F.index).astype("float32")
