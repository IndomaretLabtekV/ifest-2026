"""11 - Test forensics: is the test built/corrupted the same way as train?
For HUCs seen in both splits, static columns and the climatology of a (HUC, calendar month) must be
identical, because they do not depend on time. Any mismatch means the test columns carry a different
transformation that the V1 pipeline did not undo."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, STATIC, WORK, load_clean
from prep import clim_scale

tr, te = load_clean()
tr = tr.drop_duplicates(["id", "origin_id"])
seen = sorted(set(te.id) & set(tr.id))
print(f"[ids] seen HUCs {len(seen)} / test {te.id.nunique()}")

# 1. static columns per HUC: train value vs test value --------------------------------
st_tr = tr.groupby("id")[STATIC].median().loc[seen]
st_te = te.groupby("id")[STATIC].median().loc[seen]
print("\n[static] share of seen HUCs whose test value equals train value (rtol 1e-3):")
for c in STATIC:
    r = st_te[c] / st_tr[c]
    print(f"  {c:14s} equal {np.isclose(st_te[c], st_tr[c], rtol=1e-3).mean():.3f}   ratio median {r.median():.4f}"
          f"  p5 {r.quantile(.05):.4f}  p95 {r.quantile(.95):.4f}")
to_tr = tr.groupby("id").to_id.first().loc[seen]
to_te = te.groupby("id").to_id.first().loc[seen]
print(f"[network] seen HUCs whose to_id is identical in train and test: {(to_tr == to_te).mean():.3f}")

# 2. climatology of the same (HUC, calendar month) ---------------------------------------
def clim_long(df):
    a = pd.DataFrame({"id": df.id, "m": df.month, "c": df.clim_median_now, "src": "now"})
    b = pd.DataFrame({"id": df.id, "m": df.month % 12 + 1, "c": df.clim_median_next, "src": "next"})
    return pd.concat([a, b])

ct = clim_long(tr).groupby(["id", "m"]).c.median()
ce = clim_long(te).groupby(["id", "m"]).c.median()
j = pd.concat([ct.rename("train"), ce.rename("test")], axis=1).dropna()
lr = np.log10(j.test / j.train)
print(f"\n[clim raw] (HUC, month) pairs present in both: {len(j)}")
print("[clim raw] log10(test/train) rounded:", lr.round(1).value_counts().head(8).to_dict())
print(f"[clim raw] |log10 ratio| < 0.01 (equal): {(lr.abs() < .01).mean():.3f}")
e_tr, e_te = clim_scale(tr).groupby(tr.id).first(), clim_scale(te).groupby(te.id).first()
fixed = j.assign(train=j.train * 10.0 ** e_tr.reindex(j.index.get_level_values(0)).values,
                 test=j.test * 10.0 ** e_te.reindex(j.index.get_level_values(0)).values)
lf = np.log10(fixed.test / fixed.train)
print(f"[clim fixed] after per-split scale fix, equal: {(lf.abs() < .01).mean():.3f}; off by 1e3: {(lf.abs().round() == 3).mean():.3f}")
print("[clim scale] test exponent counts:", e_te.value_counts().to_dict())
bad = e_te[~np.isfinite(e_te)].index
print(f"[clim scale] HUCs with non-finite exponent: {len(bad)}; their clim_now values:",
      te[te.id.isin(bad)].clim_median_now.head(6).tolist())

# 3. current state: share of rows already below 0.6 x climatology --------------------------
def below_now(df):
    c = df.clim_median_now * 10.0 ** clim_scale(df)
    return (df.q_ws_cum_m3_lag0.abs() / c < 0.6)

bt, be = below_now(tr), below_now(te)
print(f"\n[state] rows currently below 0.6 clim: train {bt.mean():.3f}  test {be.mean():.3f}")
print("[state] test per origin:", be.groupby(te.origin_id).mean().round(3).to_dict())
tm = tr[tr.month.isin([2, 4, 10, 12])]
print("[state] train same months, per origin distribution: p10/p50/p90 =",
      below_now(tm).groupby(tm.origin_id).mean().quantile([.1, .5, .9]).round(3).tolist())
# and the relationship we rely on: label rate given 'below now', per origin in train
g = pd.DataFrame({"o": tr.origin_id, "b": bt.values, "y": tr.label.values}).groupby("o")
per_o = pd.DataFrame({"below": g.b.mean(), "pos": g.y.mean()})
print(f"[state] train corr(origin below-rate, origin positive-rate) = {per_o.corr().iloc[0, 1]:.3f}")
fit = np.polyfit(per_o.below, per_o.pos, 1)
print(f"[state] linear fit pos = {fit[0]:.2f}*below + {fit[1]:.3f} -> implied test positive rate "
      f"{np.polyval(fit, be.mean()):.3f}; per origin", {o: round(float(np.polyval(fit, v)), 3)
                                                       for o, v in be.groupby(te.origin_id).mean().items()})

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].hist(lr.clip(-4, 4), bins=120, color="#4C72B0"); ax[0].set_title("log10(test clim / train clim), same HUC-month")
ax[1].hist(lf.clip(-4, 4), bins=120, color="#55A868"); ax[1].set_title("same, after per-split scale fix")
ax[2].scatter(per_o.below, per_o.pos, s=10, label="train origins")
for o, v in be.groupby(te.origin_id).mean().items():
    ax[2].axvline(v, ls="--", c="#C44E52")
ax[2].set_xlabel("share below 0.6 clim now"); ax[2].set_ylabel("positive rate next month"); ax[2].legend()
ax[2].set_title("origin-level state vs label rate (red = test origins)")
plt.tight_layout(); plt.savefig(FIG / "11_test_forensics.png", dpi=100); plt.close()

# 4. per-HUC rescaling of the test volumes ------------------------------------------------------
# Within a seen HUC, log(test clim / train clim) is constant across months -> one factor per HUC.
e_tr_row, e_te_row = 10.0 ** clim_scale(tr), 10.0 ** clim_scale(te)
tab = pd.concat([pd.DataFrame({"id": tr.id, "m": tr.month, "c": tr.clim_median_now * e_tr_row}),
                 pd.DataFrame({"id": tr.id, "m": tr.month % 12 + 1, "c": tr.clim_median_next * e_tr_row})]
                ).groupby(["id", "m"]).c.median()
T = te[te.id.isin(tr.id)].copy()
T["lf"] = np.log10(T.clim_median_now * e_te_row[T.index] / tab.reindex(pd.MultiIndex.from_arrays([T.id, T.month])).values)
g = T.groupby("id").lf.agg(["mean", "std"])
print(f"\n[rescale] within-HUC std of log10(test clim/train clim): median {g['std'].median():.4f}, max {g['std'].max():.4f}")
f = 10 ** g["mean"]
print(f"[rescale] per-HUC factor: min {f.min():.3f} p5 {f.quantile(.05):.3f} median {f.median():.3f}"
      f" p95 {f.quantile(.95):.3f} max {f.max():.3f}; share within 1% of 1.0: {(np.abs(f - 1) < .01).mean():.3f}")
print("[rescale] histogram of log10 factor:", np.histogram(g["mean"], bins=np.linspace(-.7, .7, 8))[0].tolist())
# does the same factor apply to every volume family? divide test lags by the factor, compare to train medians
T["fac"] = T.id.map(f)
for fam in ["q_ws_cum_m3", "q_bf_inc_m3", "q_qf_inc_m3", "q_ps_wd_inc_m3", "q_ir_wd_inc_m3"]:
    ref = pd.DataFrame({"id": tr.id, "m": tr.month, "v": tr[f"{fam}_lag0"].abs()}).groupby(["id", "m"]).v.median()
    base = ref.reindex(pd.MultiIndex.from_arrays([T.id, T.month])).values
    raw = np.log10(T[f"{fam}_lag0"].abs() / base).replace([np.inf, -np.inf], np.nan)
    unf = np.log10(T[f"{fam}_lag0"].abs() / T.fac / base).replace([np.inf, -np.inf], np.nan)
    print(f"[rescale] {fam:15s} IQR of log10(test/train-median) raw {raw.quantile(.75) - raw.quantile(.25):.3f}"
          f" -> after dividing by factor {unf.quantile(.75) - unf.quantile(.25):.3f}")
for c in ["inc_aream2", "pop_acs_2020"]:
    print(f"[rescale] {c}: corr of log factor with log(test/train) = n/a, test/train ratio median {(st_te[c] / st_tr[c]).median():.4f} (not rescaled)")
f.rename("factor").to_csv(WORK / "test_huc_factor.csv")
plt.figure(figsize=(7, 3.5)); plt.hist(g["mean"], bins=60, color="#C44E52")
plt.title("per-HUC log10 rescale factor, test vs train"); plt.tight_layout()
plt.savefig(FIG / "11_rescale_factor.png", dpi=100); plt.close()
