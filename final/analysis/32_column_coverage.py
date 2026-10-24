"""32 - Closure audit: account for every one of the 102 raw columns.
For each column: how the pipeline uses it, its univariate AP against the label, and the train/test
distribution shift (Kolmogorov-Smirnov statistic on the parsed values). A column that is dropped must
show why: no signal, or a shift that makes it unusable."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score as ap

from common import CLIM, LAG_COLS, ROOT, STATIC, load_clean, load_raw

raw_tr, raw_te = load_raw()
tr, te = load_clean()
tr = tr.drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
y = tr.label.values

AUX = [c for c in raw_tr.columns if c.startswith("aux")]
role = {}
for c in ["id"]:
    role[c] = "key: HUC identity, never a model feature (target encoding cost -11 pts, see 08)"
role["to_id"] = "river network: normalised (case/whitespace), builds basin and up/downstream features"
role["origin_id"] = "key: random id, time order recovered from lag overlap (02); never a feature"
role["month"] = "feature: parsed from mixed text/number, used raw and as sin/cos"
role["row_id"] = "key: submission order"
role["label"] = "target: = 1 iff ws_cum(t+1) <= 0.6 * clim_next (99.99% exact, see 03)"
for c in STATIC:
    role[c] = "feature: log area / population density / growth"
for c in LAG_COLS:
    role[c] = "feature source: repaired (04) then turned into scale-free anomalies and ratios"
for c in CLIM:
    role[c] = "feature source: per-split climatology table, scale corrected (11)"
for c in AUX:
    if "fake_mm" in c:
        role[c] = "used: backup copy of inc_aream2 (fills missing area)"
    elif "twin" in c:
        role[c] = "dropped: duplicates a lag column, and maps to a DIFFERENT family in train vs test"
    elif "constant" in c:
        role[c] = "dropped: constant 0 in train but -1/999 in test"
    elif "junk" in c:
        role[c] = "dropped: random reference text"
    else:
        role[c] = "dropped: standardised noise"

rows = []
for c in raw_tr.columns:
    if c in ("row_id", "Unnamed: 0"):
        continue
    v_tr = pd.to_numeric(tr[c], errors="coerce") if c in tr.columns else pd.to_numeric(raw_tr[c], errors="coerce")
    v_te = (pd.to_numeric(te[c], errors="coerce") if c in te.columns
            else pd.to_numeric(raw_te[c], errors="coerce") if c in raw_te.columns else pd.Series(dtype=float))
    a, ks = np.nan, np.nan
    if v_tr.notna().sum() > 100 and v_tr.nunique() > 1:
        s = v_tr.replace([np.inf, -np.inf], np.nan).fillna(v_tr.median()).to_numpy()
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        a = max(ap(y, s), ap(y, -s))
        if v_te.notna().sum() > 100:
            ks = ks_2samp(v_tr.dropna().sample(min(20000, v_tr.notna().sum()), random_state=0),
                          v_te.dropna().sample(min(20000, v_te.notna().sum()), random_state=0)).statistic
    rows.append({"column": c, "role": role.get(c, "?"), "univariate_AP": a, "train_test_KS": ks})
T = pd.DataFrame(rows)
T["group"] = T.column.str.replace(r"_lag\d+$", " (12 lags)", regex=True).str.replace(r"_\d\d$", "_*", regex=True)

print(f"[coverage] columns accounted for: {len(T)} of {raw_tr.shape[1] - 1}")
g = T.groupby("group").agg(n=("column", "size"), role=("role", "first"),
                           AP=("univariate_AP", "mean"), KS=("train_test_KS", "mean")).sort_values("AP", ascending=False)
print(g.round(3).to_string())
print(f"\n[coverage] random-ranker AP = {y.mean():.3f}")
drop = T[T.role.str.startswith("dropped")]
print(f"[coverage] dropped columns: {len(drop)}; their best univariate AP {drop.univariate_AP.max():.3f} "
      f"(vs random {y.mean():.3f}); max train/test KS {drop.train_test_KS.max():.3f}")
used = T[~T.role.str.startswith(("dropped", "key"))]
print(f"[coverage] used columns: {len(used)}; median KS {used.train_test_KS.median():.3f}")
T.to_csv(ROOT / "outputs" / "column_coverage.csv", index=False)
print("[coverage] wrote outputs/column_coverage.csv")
