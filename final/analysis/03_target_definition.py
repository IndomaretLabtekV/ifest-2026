"""03 - Reverse-engineer the label. label = 1 if SUI(target month) >= threshold.
Using the recovered timeline, the row at origin t+1 carries the *actual* target-month
values in its lag0 columns. Test candidate SUI formulas and find the threshold."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import FIG, WORK, load_clean

tr, _ = load_clean()
tl = pd.read_csv(WORK / "timeline.csv").dropna(subset=["t"])
tr = tr.drop_duplicates(["id", "origin_id"]).merge(tl[["origin_id", "t"]], on="origin_id")
tr["t"] = tr.t.astype(int)
print(f"[timeline] chain starts at calendar month {tl.loc[tl.t == 0, 'month'].item()}, {tr.t.max() + 1} months long")

f = ["q_bf_inc_m3", "q_qf_inc_m3", "q_ws_cum_m3", "q_ir_wd_inc_m3", "q_th_wd_inc_m3", "q_ps_wd_inc_m3"]
nxt = tr[["id", "t"] + [f"{c}_lag0" for c in f]].copy()
nxt["t"] -= 1
nxt.columns = ["id", "t"] + [f"y_{c}" for c in f]
d = tr.merge(nxt, on=["id", "t"])
print(f"[join] rows with observed next month: {len(d)} / {len(tr)}")

bf, qf, ws, ir, th, ps = (d[f"y_{c}"] for c in f)
use = ir + th + ps
cands = {
    "use/ws_cum": use / ws,
    "use/(bf+qf)": use / (bf + qf),
    "(ir+ps)/ws_cum": (ir + ps) / ws,
    "use/clim_next": use / d.clim_median_next,
    "ws_cum/clim_next (low=stress)": -(ws / d.clim_median_next),
    "use/(ws_cum+use)": use / (ws + use),
    "ws_next/ws_now (drop)": -(ws / d.q_ws_cum_m3_lag0),
}
print("\n[candidates] separability of label by next-month quantity:")
res = []
for name, s in cands.items():
    s = s.replace([np.inf, -np.inf], np.nan)
    ok = s.notna()
    res.append((name, roc_auc_score(d.label[ok], s[ok]), average_precision_score(d.label[ok], s[ok])))
res = pd.DataFrame(res, columns=["formula", "AUC", "AP"]).sort_values("AUC", ascending=False)
print(res.to_string(index=False))

best = res.formula.iloc[0]
s = cands[best].replace([np.inf, -np.inf], np.nan)
print(f"\n[threshold] best formula: {best}")
for q in [0.0, 0.01, 0.05]:
    lo, hi = s[d.label == 1].quantile(q), s[d.label == 0].quantile(1 - q)
    print(f"  q={q:.2f}: min SUI|label1 {lo:.4f}   max SUI|label0 {hi:.4f}")
grid = np.linspace(0, 1, 201)
acc = [((s >= g) == d.label).mean() for g in grid]
g = grid[int(np.argmax(acc))]
print(f"  best global threshold {g:.3f} -> accuracy {max(acc):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for lab, col in [(0, "#4C72B0"), (1, "#C44E52")]:
    ax[0].hist(s[d.label == lab].clip(0, 2), bins=200, alpha=.6, color=col, label=f"label={lab}", density=True)
ax[0].axvline(g, color="k", ls="--"); ax[0].set_title(f"next-month {best} by label"); ax[0].legend()
ax[1].plot(grid, acc); ax[1].set_title("accuracy of (SUI >= thr) vs threshold"); ax[1].set_xlabel("threshold")
plt.tight_layout(); plt.savefig(FIG / "03_label_definition.png", dpi=110); plt.close()
d[["id", "origin_id", "t", "label"]].assign(sui_next=s.values).to_parquet(WORK / "sui_next.parquet")
