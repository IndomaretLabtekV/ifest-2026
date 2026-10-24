"""34 - Where in calendar time are train and test? (decides which validation scheme mirrors the test)
Evidence used, all from the data itself (no external data):
  a) origin_id runs 0..251 = 252 months = 21 hydrological years (Oct 1999 - Sep 2020, 'periode 2000-2020')
  b) thermoelectric withdrawals are exactly zero everywhere until one chain month, then switch on at once.
     The reconstructed chain (02) starts in September, so a switch-on in a January anchors the calendar.
  c) public-supply withdrawals drift slowly per HUC; the test windows (unscaled by the exact per-HUC factor
     from 11) are compared with the per-year train levels."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, WORK
from cvkit import T0, date_of, te, tr, y
from prep import clim_scale, repair

t = tr.t.values
tl = pd.read_csv(WORK / "timeline.csv")
ids_tr, ids_te = set(tr.origin_id), set(te.origin_id)
all_ids = set(range(int(max(ids_tr | ids_te)) + 1))
print(f"[ids] origin_id range 0..{max(all_ids)} -> {len(all_ids)} ids = {len(all_ids) / 12:.0f} years x 12 months")
print(f"[ids] train {len(ids_tr)}, public test {len(ids_te)}, never shown {len(all_ids - ids_tr - ids_te)}")

# b) thermoelectric switch-on, per absolute month over every lag copy
long = pd.concat([pd.DataFrame({"m": t - k, "v": tr[f"q_th_wd_inc_m3_lag{k}"].values}) for k in range(12)])
nz = long.assign(nz=(long.v.abs() > 0) & (long.v != -9999)).groupby("m").nz.mean()
on = int(nz.index[nz > 0.002].min())
cal = (8 + on) % 12 + 1  # t = 0 is a September (chain start, 02)
print(f"[thermo] nonzero share is 0.0000 for months {int(nz.index.min())}..{on - 1}, then {nz.loc[on]:.4f} at chain month {on}"
      f" (calendar month {cal})")
print(f"[anchor] t=0 = {date_of(0)} puts the switch-on at {date_of(on)}; the USGS thermoelectric reanalysis starts"
      f" January 2008 -> train chain = {date_of(0)} .. {date_of(t.max())}")
print(f"[anchor] first chain origin's 12-month window = {date_of(-11)} .. {date_of(0)}: exactly water year 2000,"
      f" i.e. the first origin with a full lag window")
pre = 11
post = len(all_ids) - len(ids_tr) - pre
print(f"[ids] 252 = {pre} months without a full window + {len(ids_tr)} train + {post} months after the chain"
      f" ({date_of(t.max() + 1)} .. 2020-09); public test takes {len(ids_te)} of those, {post - len(ids_te)} unseen ids remain")

# where can test origins sit: full thermo coverage in all 12 lags + no lag overlap with the chain (17)
for o, g in te.groupby("origin_id"):
    s = [((g[f"q_th_wd_inc_m3_lag{k}"].abs() > 0) & (g[f"q_th_wd_inc_m3_lag{k}"] != -9999)).mean() for k in range(12)]
    print(f"[test] origin {o} month {g.month.iloc[0]:2d}: thermo nonzero share lag0..11 min {min(s):.4f} max {max(s):.4f}")
print(f"[test] all 12 lags carry thermoelectric -> windows after {date_of(on)}; no overlap with the chain (17) ->"
      f" origins between {date_of(t.max() + 12)} and 2020-08")

# c) public-supply drift: unscale seen test HUCs by the exact factor, compare with per-year train levels
e_tr, e_te = 10.0 ** clim_scale(tr), 10.0 ** clim_scale(te)
tab = pd.concat([pd.DataFrame({"id": tr.id, "m": tr.month, "c": tr.clim_median_now * e_tr}),
                 pd.DataFrame({"id": tr.id, "m": tr.month % 12 + 1, "c": tr.clim_median_next * e_tr})]
                ).groupby(["id", "m"]).c.median()
T = te[te.id.isin(tr.id)].copy()
T["fac"] = T.clim_median_now * e_te[T.index] / tab.reindex(pd.MultiIndex.from_arrays([T.id, T.month])).values
T["fac"] = T.groupby("id").fac.transform("median")
xr, xt = repair(tr), repair(T)
fam = "q_ps_wd_inc_m3"
S = pd.concat([pd.DataFrame({"id": tr.id.values, "m": t - k, "v": xr[f"{fam}_lag{k}"].abs().values}) for k in range(12)])
S = S[(S.v > 0) & (S.m >= 0)].groupby(["id", "m"]).v.median().reset_index()
S["cm"], S["yr"] = (8 + S.m) % 12 + 1, (T0.year + (S.m + T0.month - 1) // 12)
S["lv"] = np.log(S.v)
S["dev"] = S.lv - S.groupby(["id", "cm"]).lv.transform("mean")
drift = S.groupby("yr").dev.median()
base = S.groupby(["id", "cm"]).lv.mean()
test_dev = {}
for o, g in T.groupby("origin_id"):
    gx, om = xt.loc[g.index], g.month.iloc[0]
    d = [np.log(gx[f"{fam}_lag{k}"].abs() / g.fac).values
         - base.reindex(pd.MultiIndex.from_arrays([g.id, np.full(len(g), (om - 1 - k) % 12 + 1)])).values for k in range(12)]
    test_dev[o] = np.nanmedian(np.concatenate(d)[np.isfinite(np.concatenate(d))])
print("\n[ps drift] per-HUC median log deviation from the HUC-month mean, by calendar year:")
print("  " + ", ".join(f"{int(k)}: {v:+.3f}" for k, v in drift.items()))
print("  test origins: " + ", ".join(f"{o}: {v:+.3f}" for o, v in test_dev.items()))
print(f"  every test origin sits below the latest train years ({drift.iloc[-2:].mean():+.3f}): the decline that starts"
      f" in 2013 continues -> consistent with a later period")

# prevalence through time with the anchored calendar
prev = pd.Series(y).groupby(t).mean()
wy = pd.Series(y).groupby(tr.t.values // 12).mean()
print("\n[prevalence] by Sep-Aug year: " + ", ".join(f"{date_of(12 * k)[:4]}/{int(date_of(12 * k)[:4]) + 1 - 2000:02d}"
                                                    f" {v:.3f}" for k, v in wy.items()))
dates = [T0 + pd.DateOffset(months=int(k)) for k in prev.index]
top = prev.sort_values(ascending=False).head(6)
print("[prevalence] driest target months: " + ", ".join(f"{date_of(k + 1)} {v:.3f}" for k, v in top.items()))

fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
ax[0].plot([T0 + pd.DateOffset(months=int(k)) for k in nz.index], nz.values, color="tab:red")
ax[0].axvline(T0 + pd.DateOffset(months=on), ls="--", color="k")
ax[0].set_title(f"Thermoelectric nonzero share: switches on at chain month {on} = {date_of(on)}")
ax[1].plot(dates, prev.values, lw=1)
ax[1].axhline(y.mean(), color="gray", ls=":")
ax[1].axvspan(T0 + pd.DateOffset(months=int(t.max()) + 12), pd.Timestamp("2020-08-01"), color="tab:orange", alpha=.2,
              label="where the test origins must lie")
ax[1].set_xlim(T0, pd.Timestamp("2020-09-01"))
ax[1].legend(loc="upper left")
ax[1].set_title("Positive rate per origin (anchored calendar)")
ax[2].plot(drift.index, drift.values, "o-", label="train, per-year median")
for o, v in test_dev.items():
    ax[2].axhline(v, ls="--", lw=1, color="tab:orange")
ax[2].axhline(np.nan, ls="--", color="tab:orange", label="test origins (level only, year unknown)")
ax[2].legend()
ax[2].set_title("Public-supply drift per HUC (log, unscaled)")
plt.tight_layout()
plt.savefig(FIG / "34_time_anchor.png", dpi=120)
print(f"\n[fig] {FIG / '34_time_anchor.png'}")
print("[implication] the test is a FORWARD extrapolation, 1 to 6 years after the last train month. Blocked CV"
      " (13) trains on the future to predict the past in 3 of 4 folds; a forward scheme with a >=12-month gap mirrors"
      " the test directly (35).")
