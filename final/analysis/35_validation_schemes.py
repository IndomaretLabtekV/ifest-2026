"""35 - Seven validation schemes on the SAME model and features, from leaky to test-faithful.
Leak paths closed one at a time:
  L1 random rows          same month appears in up to 12 rows (lag windows) -> near-duplicates on both sides
  L2 group by HUC         removes HUC identity leak, keeps same-time leak (regional weather shared by all HUCs)
  L3 time blocks          removes same-time leak, keeps boundary leak (windows overlapping the block edge)
  L4 blocks + purge 12    removes the overlap
  L5 L4 + unseen HUCs + rescaled/8-month-climatology valid view   (= V2..V6 harness, rebuilt in cvkit)
  L6 forward, gap 12      train only on the past, >=12 months before the validation year (34: test is future)
  L7 leave-one-year-out   14 folds, purge 12: the widest per-year distribution of AP
Questions: how much does each leak inflate AP, and where does the public 0.7552 fall?"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import UNS, base_cols, date_of, run_scheme, t, test_like, tr, views, y, year

TV, VV = views()
cols = base_cols(TV)
rng = np.random.default_rng(0)
n = len(tr)
seen = ~UNS
edges = np.linspace(t.min(), t.max() + 1, 5).astype(int)


def blocks(purge):
    return [((t < lo - purge) | (t >= hi + purge), (t >= lo) & (t < hi)) for lo, hi in zip(edges[:-1], edges[1:])]


rand = rng.integers(0, 5, n)
hucf = pd.Series(rng.integers(0, 5, tr.id.nunique()), tr.id.unique())[tr.id].values
S = {
    "L1 random rows (5-fold)": ([(rand != k, (rand == k) & seen) for k in range(5)], True),
    "L2 group by HUC (5-fold)": ([(hucf != k, (hucf == k) & seen) for k in range(5)], True),
    "L3 time blocks, no purge": ([(f, v & seen) for f, v in blocks(0)], True),
    "L4 time blocks + purge 12": ([(f, v & seen) for f, v in blocks(12)], True),
    "L5 L4 + unseen HUC + test view": (blocks(12), False),
    "L6 forward, gap 12 (6 years)": ([(t < 12 * k - 12, year == k) for k in range(8, 14)], False),
    "L7 leave-one-year-out, purge 12": ([((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)], False),
}

res, tables, oofs = [], {}, {}
for name, (folds, tv) in S.items():
    oof, tab = run_scheme(TV, VV, folds, cols, train_view_for_valid=tv)
    tables[name], oofs[name] = tab, oof
    ok = ~np.isnan(oof)
    tl = test_like(oof) if not tv else np.array([np.nan])
    res.append(dict(scheme=name, fold_mean=tab.ap.mean(), fold_std=tab.ap.std(), pooled=ap(y[ok], oof[ok]),
                    seen=tab.ap_seen.mean(), unseen=tab.ap_unseen.mean(), test_like=np.nanmean(tl),
                    tl_p10=np.nanpercentile(tl, 10), tl_p90=np.nanpercentile(tl, 90)))
    print(f"{name:34s} fold AP {tab.ap.mean():.4f} +-{tab.ap.std():.4f} | pooled {res[-1]['pooled']:.4f} | "
          f"test-like {res[-1]['test_like']:.4f} (p10 {res[-1]['tl_p10']:.3f} p90 {res[-1]['tl_p90']:.3f}) | "
          f"per-fold {np.round(tab.ap.values, 3).tolist()}", flush=True)
R = pd.DataFrame(res).set_index("scheme")
R.to_csv(WORK / "35_schemes.csv")
np.save(WORK / "35_oof_L6.npy", oofs["L6 forward, gap 12 (6 years)"])
np.save(WORK / "35_oof_L7.npy", oofs["L7 leave-one-year-out, purge 12"])
print("\n" + R.round(4).to_string())

base = R.loc["L5 L4 + unseen HUC + test view", "fold_mean"]
print("\n[leak] inflation over the test-faithful L5 fold AP:")
for k in list(S)[:4]:
    print(f"  {k:34s} {R.loc[k, 'fold_mean'] - base:+.4f}")

# per-year distribution and the role of prevalence (L7)
L7 = tables["L7 leave-one-year-out, purge 12"].assign(start=[date_of(12 * k) for k in range(14)])
L7["lift"] = (L7.ap - L7.prev) / (1 - L7.prev)
r = np.corrcoef(L7.prev, L7.ap)[0, 1]
b = np.polyfit(L7.prev, L7.ap, 1)
print("\n[L7] per year:\n" + L7[["start", "prev", "ap", "ap_seen", "ap_unseen", "lift"]].round(4).to_string(index=False))
print(f"[L7] corr(prevalence, AP) = {r:.3f}; AP ~ {b[1]:.3f} + {b[0]:.3f} x prevalence; lift std {L7.lift.std():.4f} vs AP std {L7.ap.std():.4f}")

tl7 = test_like(oofs["L7 leave-one-year-out, purge 12"], n=2000)
pub = 0.7552
print(f"[public] 0.7552 sits at percentile {100 * (tl7 < pub).mean():.0f} of {len(tl7)} test-like draws under L7 "
      f"(median {np.median(tl7):.4f}, p10 {np.percentile(tl7, 10):.4f}, p90 {np.percentile(tl7, 90):.4f})")
d = pd.DataFrame({"t": t, "y": y})
prev_draw = []
rng2 = np.random.default_rng(0)
by_m = {m: np.unique(t[tr.month.values == m]) for m in (2, 4, 10, 12)}
pm = d.groupby("t").y.mean()
for _ in range(2000):
    pick = [rng2.choice(by_m[m]) for m in by_m]
    prev_draw.append(pm.loc[pick].mean())
print(f"[public] 4-origin prevalence varies p10 {np.percentile(prev_draw, 10):.3f} .. p90 {np.percentile(prev_draw, 90):.3f};"
      f" the AP needed for 0.7552 at the L7 slope corresponds to prevalence ~{(pub - b[1]) / b[0]:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
R.fold_mean.iloc[::-1].plot.barh(ax=ax[0], xerr=R.fold_std.iloc[::-1], color=["tab:green"] * 3 + ["tab:red"] * 4)
ax[0].axvline(pub, color="k", ls="--", label="public V5 0.7552")
ax[0].set_xlim(0.5, 1.0)
ax[0].legend()
ax[0].set_title("Fold AP by validation scheme (red = leaky)")
ax[1].scatter(L7.prev, L7.ap)
for _, rr in L7.iterrows():
    ax[1].annotate(rr.start[:4], (rr.prev, rr.ap), fontsize=8)
xx = np.linspace(L7.prev.min(), L7.prev.max(), 10)
ax[1].plot(xx, np.polyval(b, xx), "k:")
ax[1].set_xlabel("year prevalence"); ax[1].set_ylabel("AP")
ax[1].set_title(f"L7 leave-one-year-out: AP follows prevalence (r={r:.2f})")
ax[2].hist(tl7, bins=40, color="tab:blue", alpha=.7)
ax[2].axvline(pub, color="k", ls="--", label="public 0.7552")
ax[2].legend()
ax[2].set_title("L7 test-like 4-origin AP draws")
plt.tight_layout()
plt.savefig(FIG / "35_validation_schemes.png", dpi=120)
print(f"[fig] {FIG / '35_validation_schemes.png'}")
