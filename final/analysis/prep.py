"""Preprocessing functions (ported verbatim into the Kaggle notebook)."""
import numpy as np
import pandas as pd

FLOWS = ["q_bf_inc_m3", "q_qf_inc_m3", "q_ws_cum_m3", "q_ir_wd_inc_m3", "q_th_wd_inc_m3", "q_ps_wd_inc_m3"]
SENTINELS = [-9999.0, -9.999]
# families that are smooth month-to-month; qf (storm runoff) is legitimately spiky
# thermoelectric switches on/off, so it is excluded (spike rule damaged 1.8% of its clean cells)
SPIKE_FAMS = {"q_bf_inc_m3": 8.0, "q_ws_cum_m3": 12.0, "q_ps_wd_inc_m3": 5.0}
ZERO_FAMS = ["q_bf_inc_m3", "q_ps_wd_inc_m3", "q_ws_cum_m3"]


def _block(df, fam):
    return df[[f"{fam}_lag{k}" for k in range(12)]].to_numpy(dtype="float64", copy=True)


def _interp_log(x):
    lx = pd.DataFrame(np.log1p(x)).T.interpolate(limit_direction="both").T.to_numpy()
    return np.where(np.isnan(x), np.expm1(lx), x)


def _neighbour_max(x):
    p = np.pad(x, ((0, 0), (1, 1)), constant_values=np.nan)
    return np.fmax(p[:, :-2], p[:, 2:])


def _neighbour_min(x):
    p = np.pad(x, ((0, 0), (1, 1)), constant_values=np.nan)
    return np.fmin(p[:, :-2], p[:, 2:])


def clean_flags(x, fam, spike=True, zero=True, zero_first=True):
    """Stage 1: turn every detectable corruption into NaN (sign flips are just un-flipped).
    zero_first (38): detect injected zeros BEFORE spikes and ignore zero neighbours in the spike test.
    The old order flagged a clean value as a 'spike' whenever its only neighbour was an injected zero
    (neighbour max = 0), then overwrote it by interpolation."""
    x[np.isin(x, SENTINELS)] = np.nan
    x = np.abs(x)
    if zero_first and zero and fam in ZERO_FAMS:
        x[(x == 0) & (_neighbour_min(x) > 10)] = np.nan
    if spike and fam in SPIKE_FAMS:
        nb = _neighbour_max(np.where(x == 0, np.nan, x) if zero_first else x)
        x[x > SPIKE_FAMS[fam] * nb] = np.nan
    if not zero_first and zero and fam in ZERO_FAMS:
        x[(x == 0) & (_neighbour_min(x) > 10)] = np.nan
    return x


def clean_blocks(df, spike=True, zero=True, zero_first=True, storm=True):
    """Every family's 12 lags with detected corruption set to NaN.
    storm (38): a ws value flagged as a spike is kept when the water budget backs it, i.e.
    ws <= 1.5 * (bf + qf + neighbouring upstream inflow). Real storms raise quickflow in the same month;
    injected x10-100 spikes hit ws alone. At lag 0 (one neighbour) the plain rule flagged ~5k clean storm
    months; this keeps 74% of all false flags while missing 5 of 7,033 injected spikes."""
    B = {f: clean_flags(_block(df, f), f, spike, zero, zero_first) for f in FLOWS}
    if storm and spike:
        clean = lambda f: np.abs(np.where(np.isin(_block(df, f), SENTINELS), np.nan, _block(df, f)))
        x, bf, qf = clean("q_ws_cum_m3"), clean("q_bf_inc_m3"), clean("q_qf_inc_m3")
        with np.errstate(invalid="ignore"):
            u_nb = np.nan_to_num(_neighbour_max(np.clip(x - bf - qf, 0, None)))
            ok = (x > 0) & (x <= 1.5 * (bf + qf + u_nb))
        B["q_ws_cum_m3"] = np.where(np.isnan(B["q_ws_cum_m3"]) & ok, x, B["q_ws_cum_m3"])
    return B


def repair(df: pd.DataFrame, spike=True, zero=True, identity=True, zero_first=True, storm=True) -> pd.DataFrame:
    """Returns a copy of df with all 72 lag columns repaired."""
    B = clean_blocks(df, spike, zero, zero_first, storm)
    if identity:
        # baseflow is smooth -> plain interpolation beats the budget identity for it (checked in 04)
        bf, qf, ws = _interp_log(B["q_bf_inc_m3"]), B["q_qf_inc_m3"], B["q_ws_cum_m3"]
        # upstream inflow U = ws - bf - qf is smooth; fill it, then close the budget ws = bf + qf + U
        U = _interp_log(np.clip(ws - bf - qf, 0, None))
        ws = np.where(np.isnan(ws), bf + np.nan_to_num(qf) + U, ws)
        qf = np.where(np.isnan(qf), np.clip(ws - bf - U, 0, None), qf)
        B["q_bf_inc_m3"], B["q_qf_inc_m3"], B["q_ws_cum_m3"] = bf, qf, ws
    out = df.copy()
    for f, x in B.items():
        out[[f"{f}_lag{k}" for k in range(12)]] = _interp_log(x)
    return out


def network_fill(df: pd.DataFrame, X: pd.DataFrame, tol: float = 0.01, min_cells: int = 6) -> pd.DataFrame:
    """Second repair pass (38): the river-network water budget ws(h) = bf(h) + qf(h) + sum_u ws(u) over the
    upstream HUCs u whose rows sit in the same origin. The test rescales every HUC by its own factor, so the
    identity holds as ws(h) = bf(h) + qf(h) + sum_u c_u ws(u) with c_u = f_h / f_u; c is fitted per downstream
    HUC by NNLS on its clean cells (pooled over all its rows), and used only where the fit error < tol.
    Refills flagged ws, qf and bf cells of X (the output of repair)."""
    from scipy.optimize import nnls

    fams = ("q_ws_cum_m3", "q_bf_inc_m3", "q_qf_inc_m3")
    B = clean_blocks(df)
    flag = {f: np.isnan(B[f]) for f in fams}
    ws, bf, qf = (X[[f"{f}_lag{k}" for k in range(12)]].to_numpy(dtype="float64", copy=True) for f in fams)
    pos = pd.Series(np.arange(len(df)), pd.MultiIndex.from_arrays([df.id.values, df.origin_id.values]))
    pos = pos[~pos.index.duplicated()]
    down = pos.reindex(pd.MultiIndex.from_arrays([df.to_id.values, df.origin_id.values])).to_numpy()
    e = pd.DataFrame({"d": down, "u": np.arange(len(df)), "h": df.to_id.values, "uh": df.id.values}).dropna()
    e["d"] = e.d.astype(int)
    anyflag = flag["q_ws_cum_m3"] | flag["q_bf_inc_m3"] | flag["q_qf_inc_m3"]
    new_ws, new_bf, new_qf = ws.copy(), bf.copy(), qf.copy()
    for _, g in e.groupby("h", sort=False):
        ups = {u: i for i, u in enumerate(pd.unique(g.uh))}
        rows = np.unique(g.d.values)
        ri = {r: i for i, r in enumerate(rows)}
        S = np.zeros((len(rows), 12, len(ups)))
        Sbad = np.zeros((len(rows), 12), bool)
        for d, u, uh in zip(g.d.values, g.u.values, g.uh.values):
            S[ri[d], :, ups[uh]] = ws[u]
            Sbad[ri[d]] |= flag["q_ws_cum_m3"][u]
        clean = ~anyflag[rows] & ~Sbad & np.isfinite(S).all(-1)
        if clean.sum() < min_cells:
            continue
        U = np.clip(ws[rows] - bf[rows] - qf[rows], 0, None)
        c, _ = nnls(S[clean], U[clean])
        up = np.nan_to_num(S @ c)
        pred = bf[rows] + qf[rows] + up
        if np.median(np.abs(pred[clean] - ws[rows][clean]) / np.maximum(ws[rows][clean], 1e-9)) > tol:
            continue
        fw, fb, fq = flag["q_ws_cum_m3"][rows], flag["q_bf_inc_m3"][rows], flag["q_qf_inc_m3"][rows]
        new_ws[rows] = np.where(fw & ~fb & ~fq, pred, new_ws[rows])
        new_qf[rows] = np.where(fq & ~fw & ~fb, np.clip(ws[rows] - bf[rows] - up, 0, None), new_qf[rows])
        new_bf[rows] = np.where(fb & ~fw & ~fq, np.clip(ws[rows] - qf[rows] - up, 0, None), new_bf[rows])
    out = X.copy()
    for f, v in zip(fams, (new_ws, new_bf, new_qf)):
        out[[f"{f}_lag{k}" for k in range(12)]] = v
    return out


def clim_scale(df: pd.DataFrame) -> pd.Series:
    """Exponent e (multiple of 3) such that clim * 10**e is on the same scale as ws_cum.
    Estimated per HUC from same-calendar-month pairs: lag0 <-> clim_now, lag11 <-> clim_next."""
    a = np.log10((df.q_ws_cum_m3_lag0.abs() / df.clim_median_now).where(lambda s: s > 0))
    b = np.log10((df.q_ws_cum_m3_lag11.abs() / df.clim_median_next).where(lambda s: s > 0))
    h = pd.concat([a, b]).groupby(pd.concat([df.id, df.id])).median()
    return (3 * np.round(df.id.map(h) / 3)).fillna(0)


if __name__ == "__main__":
    row = {f"{f}_lag{k}": 100.0 for f in FLOWS for k in range(12)}
    row.update({"q_ws_cum_m3_lag0": 300.0, "q_ws_cum_m3_lag2": 300.0, "q_ws_cum_m3_lag1": np.nan,
                "q_bf_inc_m3_lag4": 5000.0, "q_bf_inc_m3_lag5": -100.0, "q_ps_wd_inc_m3_lag6": 0.0})
    for k in range(3, 12):
        row[f"q_ws_cum_m3_lag{k}"] = 300.0
    r = repair(pd.DataFrame([row])).iloc[0]
    assert abs(r.q_ws_cum_m3_lag1 - 300) < 1e-6, r.q_ws_cum_m3_lag1  # identity fill
    assert abs(r.q_bf_inc_m3_lag4 - 100) < 1e-6 and r.q_bf_inc_m3_lag5 == 100  # spike + sign flip
    assert abs(r.q_ps_wd_inc_m3_lag6 - 100) < 1e-6  # injected zero
    d = pd.DataFrame({"id": ["a", "a", "b"], "q_ws_cum_m3_lag0": [1000.0, 2000, 5], "clim_median_now": [1, 2, 5],
                      "q_ws_cum_m3_lag11": [np.nan] * 3, "clim_median_next": [1.0] * 3})
    assert clim_scale(d).tolist() == [3, 3, 0]
    # network identity: h has one upstream u rescaled by 1/2 relative to h, so c_u must come out as 2
    rng = np.random.default_rng(0)
    rows = []
    for o in range(3):
        wsu = rng.uniform(50, 100, 12)
        bfh, qfh = rng.uniform(10, 20, 12), rng.uniform(1, 5, 12)
        rows.append({"id": "u", "to_id": "h", "origin_id": o, **{f"{f}_lag{k}": v for f, a in
                     [("q_ws_cum_m3", wsu / 2), ("q_bf_inc_m3", wsu / 4), ("q_qf_inc_m3", wsu / 4)] for k, v in enumerate(a)}})
        wsh = bfh + qfh + wsu
        rows.append({"id": "h", "to_id": "outlet", "origin_id": o, **{f"{f}_lag{k}": v for f, a in
                     [("q_ws_cum_m3", wsh), ("q_bf_inc_m3", bfh), ("q_qf_inc_m3", qfh)] for k, v in enumerate(a)}})
    net = pd.DataFrame(rows)
    for f in ("q_ir_wd_inc_m3", "q_th_wd_inc_m3", "q_ps_wd_inc_m3"):
        for k in range(12):
            net[f"{f}_lag{k}"] = 1.0
    truth = net.loc[1, "q_ws_cum_m3_lag5"]
    net.loc[1, "q_ws_cum_m3_lag5"] = -9999.0
    fixed = network_fill(net, repair(net))
    assert abs(fixed.loc[1, "q_ws_cum_m3_lag5"] - truth) < 1e-6 * truth, (fixed.loc[1, "q_ws_cum_m3_lag5"], truth)
    print("ok")
