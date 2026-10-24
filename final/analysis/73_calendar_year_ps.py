"""73 - Decide the order of the test chain with a DETERMINISTIC property of the data instead of weather statistics.
Public-supply withdrawals are generated as (annual level of the calendar year) x (fixed monthly pattern of the HUC):
in train the two-way fit log ps = a[HUC, year] + b[HUC, month] leaves a residual of 0.022 (water years 0.031).
So two test windows that share a calendar year must imply the same annual level a[Y] for a HUC, to about 2%.
Hypotheses (71): A 169 -> 94 -> 81 -> 14 (gaps 14,14,14), B 81 -> 14 -> 169 -> 94 (gaps 14,18,14).
  - 81 (Jan-Feb of year Y) and 14 (May-Dec of year Y) share a year under BOTH -> positive control.
  - 14 (Jan-Apr) and 169 (Nov-Dec) share a year ONLY under B -> the decisive test.
  - 169 and 94 never share a year under either -> negative control.
The monthly pattern b is fitted per seen HUC on train; the per-HUC test scale factor is the same for all four origins,
so it cancels in the difference of implied levels between two origins."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG
from cvkit import T0, te, tr, t
from prep import repair

cal = (T0.month - 1 + np.arange(400)) % 12 + 1
yr = T0.year + (T0.month - 1 + np.arange(400)) // 12

# monthly pattern b[id, m] from train, two-way model in calendar years
x = repair(tr)
L = pd.DataFrame({"id": tr.id.values, "m": t, "v": x.q_ps_wd_inc_m3_lag0.abs().values})
L = L[L.v > 0]
L["cm"], L["cy"], L["lv"] = cal[L.m], yr[L.m], np.log(L.v)
L = L.drop_duplicates(["id", "m"])
moving = L.groupby("id").lv.std()
moving = set(moving[moving > 1e-6].index)
L = L[L.id.isin(moving)]
a = L.groupby(["id", "cy"]).lv.transform("mean")
b = (L.lv - a).groupby([L.id, L.cm]).mean()
B = b.unstack()
B = B.sub(B.mean(1), axis=0)
yearly = L.groupby(["id", "cy"]).lv.mean().groupby("id").diff().dropna()
print(f"[73] train: {len(moving)} HUCs with moving public supply; year-to-year change of the annual level: "
      f"median |d| {yearly.abs().median():.3f}, p10 {yearly.abs().quantile(.1):.3f}")

# same statistic on train pairs with known answer: implied level from months of year Y seen in two DIFFERENT windows
te_ = te.reset_index(drop=True)
xt = repair(te_)
orig = te_.groupby("origin_id").month.first().to_dict()


def implied_level(o, months):
    """per HUC: mean over the given calendar months of (log ps - b[month]); months = lag indices of origin o."""
    g = te_[te_.origin_id == o]
    xs = xt.loc[g.index]
    vals = []
    for k in months:
        cm = (orig[o] - 1 - k) % 12 + 1
        v = np.log(xs[f"q_ps_wd_inc_m3_lag{k}"].abs().replace(0, np.nan).values) - B.reindex(g.id.values)[cm].values
        vals.append(v)
    return pd.Series(np.nanmean(np.column_stack(vals), 1), index=g.id.values)


def lags_of(o, cal_months):
    """lag indices k of origin o whose calendar month is in cal_months and in the same calendar year as each other"""
    return [k for k in range(12) if (orig[o] - 1 - k) % 12 + 1 in cal_months]


TESTS = {
    "81 Jan-Feb vs 14 May-Dec (share a year in A and B: positive control)": ((81, [1, 2]), (14, list(range(5, 13)))),
    "14 Jan-Apr vs 169 Nov-Dec (share a year ONLY in B: decisive)": ((14, [1, 2, 3, 4]), (169, [11, 12])),
    "169 Jan-Oct vs 94 Jan-Dec (never share: negative control)": ((169, list(range(1, 11))), (94, list(range(1, 13)))),
    "94 Jan-Dec vs 81 Mar-Dec (never share)": ((94, list(range(1, 13))), (81, list(range(3, 13)))),
    # controls with EXACTLY the decisive test's months (Jan-Apr vs Nov-Dec)
    "94 Jan-Apr vs 94 Nov-Dec (same window, same year: matched positive)": ((94, [1, 2, 3, 4]), (94, [11, 12])),
    "14 Jan-Apr vs 94 Nov-Dec (never share: matched negative)": ((14, [1, 2, 3, 4]), (94, [11, 12])),
    "81 Jan-Feb vs 169 Nov-Dec (never share: matched negative)": ((81, [1, 2]), (169, [11, 12])),
}
fig, ax = plt.subplots(figsize=(9, 4.8))
rows = []
for name, ((o1, m1), (o2, m2)) in TESTS.items():
    l1, l2 = implied_level(o1, lags_of(o1, m1)), implied_level(o2, lags_of(o2, m2))
    d = (l1 - l2.reindex(l1.index)).dropna()
    d = d[np.isfinite(d)]
    rows.append(dict(test=name, n=len(d), median_abs=d.abs().median(), share_lt_002=(d.abs() < 0.02).mean(),
                     share_lt_005=(d.abs() < 0.05).mean()))
    ax.hist(d.clip(-.4, .4), bins=160, histtype="step", density=True, label=name[:40])
R = pd.DataFrame(rows)
print("\n[73] difference of implied annual public-supply level between two test windows (moving HUCs):")
print(R.round(3).to_string(index=False))
print("[73] reading: sharing a calendar year -> |d| ~ residual 0.02; different years -> |d| ~ year-to-year change")
ax.set_xlabel("implied annual level, window 1 - window 2 (log)"); ax.set_ylabel("density")
ax.set_title("Same calendar year gives identical annual level")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "73_calendar_year_ps.png", dpi=120)
print(f"[fig] {FIG / '73_calendar_year_ps.png'}")
