"""01 - Data quality audit: number formats, month formats, missing, negatives, aux_* traps.
Writes cleaned parquet caches used by all later scripts."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, LAG_COLS, NUM_COLS, WORK, load_raw, parse_month, parse_num

tr, te = load_raw()
print("raw shapes", tr.shape, te.shape)

# 1. number-format patterns ---------------------------------------------------
pat = pd.concat([tr[c] for c in NUM_COLS]).dropna().str.replace(r"\d+", "9", regex=True)
print("\n[format] share of numeric cells by pattern:")
print((pat.value_counts(normalize=True) * 100).round(3).head(10).to_string())

# 2. resolve the ambiguous 'ddd,ddd' cells (decimal comma vs thousands sep) ------
# aux_twin is 0.0 exactly where the source was comma-formatted, so it cannot arbitrate.
# Use temporal continuity instead: lag0 and lag1 are adjacent months of the same series.
aux = [c for c in tr.columns if c.startswith("aux")]
p = {c: parse_num(tr[c]) for c in LAG_COLS}
for fam in ["q_bf_inc_m3", "q_qf_inc_m3", "q_ws_cum_m3", "q_ps_wd_inc_m3"]:
    a, b = f"{fam}_lag1", f"{fam}_lag2"
    amb = tr[a].str.fullmatch(r"\d{1,3},\d{3}").fillna(False) & (p[a] > 1) & (p[b] > 1)
    r_dec = np.log10(p[a][amb] / p[b][amb]).abs().median()
    r_thou = np.log10(p[a][amb] * 1000 / p[b][amb]).abs().median()
    print(f"[ambiguous] {fam}_lag1 n={amb.sum():5d}: median |log10(lag1/lag2)| decimal-comma {r_dec:.2f} vs thousands {r_thou:.2f}")
for tw in [c for c in aux if "twin" in c]:
    t = pd.to_numeric(tr[tw], errors="coerce") * 1000
    src = max(LAG_COLS, key=lambda c: (np.isclose(p[c], t, rtol=1e-4)).mean())
    comma = tr[src].str.contains(",", na=False)
    print(f"[twin] {tw} = {src}/1000 match {np.isclose(p[src], t, rtol=1e-4).mean():.3f};"
          f" twin==0 where source has a comma: {(t[comma] == 0).mean():.3f}")

# 3. parse everything -----------------------------------------------------------
def clean(df):
    out = df[["row_id", "id", "to_id"]].copy()
    # to_id carries case/whitespace corruption ('ea4300f0560e ' / 'Feddeb35eb1c'); ids are clean lowercase hex
    out["to_id"] = df.to_id.str.strip().str.lower()
    out["origin_id"] = df["origin_id"].astype(int)
    out["month"] = parse_month(df["month"])
    for c in NUM_COLS:
        out[c] = parse_num(df[c])
    for c in aux:
        if "junk" not in c:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    if "label" in df:
        out["label"] = df["label"].astype(int)
    out["row_id"] = out["row_id"].astype(int)
    return out

ctr, cte = clean(tr), clean(te)
for n, raw_, c_ in [("train", tr, ctr), ("test", te, cte)]:
    bad = raw_.to_id != raw_.to_id.str.strip().str.lower()
    print(f"[to_id] {n}: non-canonical (excl. OUTLET) {(bad & (raw_.to_id != 'OUTLET')).mean():.4f};"
          f" distinct raw {raw_.to_id.nunique()} -> normalised {c_.to_id.nunique()};"
          f" max to_id per id after fix {c_.groupby('id').to_id.nunique().max()}")
print("\n[month] raw non-numeric month strings:", (~tr.month.str.isdigit()).sum(), "train,", (~te.month.str.isdigit()).sum(), "test")

# 4. missing / negative / scientific-notation ---------------------------------
miss = pd.DataFrame({"train": ctr[NUM_COLS].isna().mean(), "test": cte[NUM_COLS].isna().mean()}) * 100
neg = pd.DataFrame({"train": (ctr[NUM_COLS] < 0).mean(), "test": (cte[NUM_COLS] < 0).mean()}) * 100
print(f"\n[missing %] mean train {miss.train.mean():.2f}  test {miss.test.mean():.2f}; max {miss.train.max():.2f}")
print(f"[negative %] mean train {neg.train.mean():.3f}  test {neg.test.mean():.3f}; cols with any: {(neg.train > 0).sum()}")
rows_any_nan = ctr[NUM_COLS].isna().any(axis=1).mean()
print(f"[missing] rows with >=1 NaN: train {rows_any_nan:.1%}")
sci = te["clim_median_now"].str.contains("e", na=False).mean(), tr["clim_median_now"].str.contains("e", na=False).mean()
print(f"[sci-notation clim_median_now] test {sci[0]:.1%} train {sci[1]:.1%}  (clim stored with 6 sig. digits)")

# 5. aux_* audit ----------------------------------------------------------------
print("\n[aux] train vs test summary:")
for c in [c for c in aux if "junk" not in c]:
    print(f"  {c:18s} train mean {ctr[c].mean():12.4g} nuniq {ctr[c].nunique():7d} | test mean {cte[c].mean():12.4g} nuniq {cte[c].nunique():6d}"
          f" | corr(label) {ctr[c].corr(ctr.label):+.4f}")
fm = ctr["aux_fake_mm_04"] * 1e4 / ctr["inc_aream2"]
print(f"[aux] fake_mm*1e4/inc_aream2: median {fm.median():.4f}, IQR {fm.quantile(.25):.4f}-{fm.quantile(.75):.4f}")

# 6. figures ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
miss.train.groupby(lambda c: c.rsplit("_lag", 1)[0]).mean().sort_values().plot.barh(ax=ax[0], color="#4C72B0")
ax[0].set_title("Mean missing % by column family (train)")
neg.train.groupby(lambda c: c.rsplit("_lag", 1)[0]).mean().sort_values().plot.barh(ax=ax[1], color="#C44E52")
ax[1].set_title("Mean negative % by column family (train)")
plt.tight_layout(); plt.savefig(FIG / "01_missing_negative.png", dpi=110); plt.close()

ctr.to_parquet(WORK / "train.parquet"); cte.to_parquet(WORK / "test.parquet")
print("\nsaved", ctr.shape, cte.shape)
