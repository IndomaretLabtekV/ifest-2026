"""75 - A population clock to date the test origins. Public supply (ps) is generated as annual level x monthly
pattern (73). If the annual level follows the HUC's population, interpolated between censuses, then for one HUC
  level(Y2) - level(Y1) ~ r_huc * (Y2 - Y1),   r_huc = log(pop_2020 / pop_2010) / 10 (per year, static columns).
Regressing the window-to-window level difference on r_huc across HUCs then gives the time difference in YEARS.
(1) validate on train windows with known distance (2010-2014 is inside the 2010-2020 census interval);
(2) apply to every pair of test origins and compare with chain A (169, 94, 81, 14 at 0, 14, 28, 42 months).
The per-HUC random scale of the test cancels in the difference between two origins of the same HUC."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG
from cvkit import T0, te, tr, t
from prep import repair

cal = (T0.month - 1 + np.arange(400)) % 12 + 1
yr = T0.year + (T0.month - 1 + np.arange(400)) // 12

x = repair(tr)
L = pd.DataFrame({"id": tr.id.values, "m": t, "v": x.q_ps_wd_inc_m3_lag0.abs().values}).query("v > 0").drop_duplicates(["id", "m"])
L["cm"], L["cy"], L["lv"] = cal[L.m], yr[L.m], np.log(L.v)
sd = L.groupby("id").lv.std()
L = L[L.id.isin(sd[sd > 1e-6].index)]
a = L.groupby(["id", "cy"]).lv.transform("mean")
B = (L.lv - a).groupby([L.id, L.cm]).mean().unstack()
B = B.sub(B.mean(1), axis=0)
pop = pd.concat([tr, te]).drop_duplicates("id").set_index("id")
r = (np.log(pop.pop_dec_2020.abs() + 1) - np.log(pop.pop_dec_2010.abs() + 1)) / 10
r0 = (np.log(pop.pop_dec_2010.abs() + 1) - np.log(pop.pop_dec_2000.abs() + 1)) / 10


def window_level(df, xr, month_col):
    """per row: mean over the 12 lags of (log ps - b[HUC, calendar month])"""
    Bm = B.reindex(df.id.values)
    vals = []
    for k in range(12):
        cm = (df[month_col].values - 1 - k) % 12 + 1
        b = Bm.to_numpy()[np.arange(len(df)), cm - 1]
        vals.append(np.log(xr[f"q_ps_wd_inc_m3_lag{k}"].abs().replace(0, np.nan).values) - b)
    return np.nanmean(np.column_stack(vals), 1)


def slope_years(d, rate):
    ok = np.isfinite(d) & np.isfinite(rate) & (np.abs(rate) > 0.002)
    X = np.column_stack([np.ones(ok.sum()), rate[ok]])
    coef = np.linalg.lstsq(X, d[ok], rcond=None)[0]
    return coef[1], ok.sum()


# (1) train validation: window pairs (s, s+g) for windows ending in 2011-2014 (inside the 2010-2020 census interval)
tr_lv = pd.Series(window_level(tr, x, "month"), index=pd.MultiIndex.from_arrays([tr.id.values, t]))
tr_lv = tr_lv[~tr_lv.index.duplicated()]
W = tr_lv.unstack()
rows = []
for s in range(130, 168):
    for g in (12, 14, 24, 26, 28, 36):
        if s - g < 100 or s not in W.columns or s - g not in W.columns:
            continue
        d = (W[s] - W[s - g]).reindex(r.index).to_numpy()
        est, n = slope_years(d, r.reindex(W.index).reindex(r.index).to_numpy())
        rows.append(dict(end=s, gap_true=g / 12, est=est, n=n))
V = pd.DataFrame(rows)
print("[75] train validation (windows ending 2011-2014), estimated years vs true years:")
print(V.groupby("gap_true").est.agg(["mean", "std", "count"]).round(2).to_string())
K = (V.est * V.gap_true).sum() / (V.gap_true ** 2).sum()  # clock rate: estimate units per true year (fit through 0)
print(f"[75] clock rate {K:.4f} per year; calibrated train error sd {((V.est / K) - V.gap_true).std():.2f} years")

# (2) test pairs
te_ = te.reset_index(drop=True)
xt = repair(te_)
te_lv = pd.Series(window_level(te_, xt, "month"), index=pd.MultiIndex.from_arrays([te_.id.values, te_.origin_id.values]))
T = te_lv[~te_lv.index.duplicated()].unstack()
orig = sorted(T.columns)
A_POS = {169: 0, 94: 14, 81: 28, 14: 42}
print("\n[75] test: estimated years between origins (column minus row) and chain-A prediction:")
res = []
for o1, o2 in itertools.combinations(orig, 2):
    d = (T[o2] - T[o1]).reindex(r.index).to_numpy()
    est, n = slope_years(d, r.to_numpy())
    res.append(dict(pair=f"{o1}->{o2}", est_years=est / K, chainA_years=(A_POS[o2] - A_POS[o1]) / 12, n=n))
R = pd.DataFrame(res)
print(R.round(2).to_string(index=False))
# absolute placement: least squares positions (years) with origin 169 at 0
idx = {o: i for i, o in enumerate(orig)}
M = np.zeros((len(res) + 1, len(orig))); v = np.zeros(len(res) + 1)
for k, row in enumerate(res):
    o1, o2 = map(int, row["pair"].split("->"))
    M[k, idx[o2]], M[k, idx[o1]], v[k] = 1, -1, row["est_years"]
M[-1, idx[169]] = 1
pos = np.linalg.lstsq(M, v, rcond=None)[0]
print("[75] least-squares positions (years after origin 169): " + ", ".join(f"{o}: {pos[idx[o]]:+.2f}" for o in orig))
print("[75] chain A positions (years):                         " + ", ".join(f"{o}: {A_POS[o] / 12:+.2f}" for o in orig))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(V.gap_true + np.random.default_rng(0).normal(0, .03, len(V)), V.est / K, s=10)
ax[0].plot([0, 3.2], [0, 3.2], "k--", lw=.8)
ax[0].set_xlabel("true years between train windows"); ax[0].set_ylabel("population-clock estimate (years)")
ax[0].set_title("Validation on train")
ax[1].scatter(R.chainA_years, R.est_years)
for rr in R.itertuples():
    ax[1].annotate(rr.pair, (rr.chainA_years, rr.est_years), fontsize=8)
ax[1].plot([-4, 4], [-4, 4], "k--", lw=.8)
ax[1].set_xlabel("chain A years"); ax[1].set_ylabel("population-clock estimate (years)")
ax[1].set_title("Test origin pairs")
plt.tight_layout()
plt.savefig(FIG / "75_population_clock.png", dpi=120)
print(f"[fig] {FIG / '75_population_clock.png'}")
