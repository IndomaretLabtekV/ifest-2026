"""04 - Preprocessing + verification against ground truth.
Every monthly value in train appears in up to 12 rows (lag0..lag11 of consecutive origins),
so the cross-row median is a near-exact ground truth. Each repair step is scored against it:
did it move corrupted cells toward the truth, and did it damage clean cells?"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, FLOWS, LAG, WORK, load_clean
from prep import clim_scale, repair

tr, te = load_clean()
tl = pd.read_csv(WORK / "timeline.csv").dropna(subset=["t"])
tr = tr.merge(tl[["origin_id", "t"]], on="origin_id")

# ground truth: consensus value of (id, absolute month) across all rows that contain it
def truth(frame, fam):
    long = pd.concat([pd.DataFrame({"id": frame.id, "m": frame.t - k, "v": frame[f"{fam}_lag{k}"]}) for k in LAG])
    return long.groupby(["id", "m"]).v.median()

CONFIGS = {
    "v1 interp only": dict(spike=False, zero=False, identity=False),
    "v2 +spike": dict(spike=True, zero=False, identity=False),
    "v3 +zero": dict(spike=True, zero=True, identity=False),
    "v4 +identity": dict(spike=True, zero=True, identity=True),
}
GT, IDX = {}, [pd.MultiIndex.from_arrays([tr.id, tr.t - k]) for k in LAG]
for fam in FLOWS:
    gt = truth(tr, fam)
    GT[fam] = np.column_stack([gt.reindex(ix).to_numpy() for ix in IDX])

def err(X, G):
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.abs(X - G) / np.abs(G)
    return np.where(np.isnan(X), np.inf, e)

report, keep = [], {}
for cname, cfg in CONFIGS.items():
    fixed = repair(tr, **cfg)
    for fam in FLOWS:
        cols = [f"{fam}_lag{k}" for k in LAG]
        G, raw, X = GT[fam], tr[cols].to_numpy(), fixed[cols].to_numpy()
        ok = ~np.isnan(G) & (np.abs(G) > 0)
        bad = ok & ~(err(raw, G) < 1e-3)
        e = err(X, G)
        report.append({"config": cname, "family": fam, "corrupt_%": 100 * bad.mean(),
                       "fixed_within_5%": 100 * (e[bad] < 0.05).mean(),
                       "fixed_within_25%": 100 * (e[bad] < 0.25).mean(),
                       "clean_damaged_%": 100 * (e[ok & ~bad] > 0.05).mean()})
        if fam == "q_ws_cum_m3":
            keep[cname] = (raw[bad], X[bad], G[bad])

rep = pd.DataFrame(report)
print(rep.pivot(index="family", columns="config", values="fixed_within_5%").round(1).to_string())
print("\n[within 25%]"); print(rep.pivot(index="family", columns="config", values="fixed_within_25%").round(1).to_string())
print("\n[clean cells damaged %]"); print(rep.pivot(index="family", columns="config", values="clean_damaged_%").round(3).to_string())

fam = "q_ws_cum_m3"
r, g = keep["v1 interp only"][0], keep["v1 interp only"][2]
kind = np.select([np.isnan(r), r == -9999, r < 0, r == 0], ["NaN", "-9999", "negative", "zero"], "spike")
for cname in CONFIGS:
    rel = np.abs(keep[cname][1] - g) / np.abs(g)
    print(f"[{fam}] {cname:15s} within 5% by type:", pd.Series(rel < 0.05).groupby(kind).mean().round(3).to_dict())
r, f, g = keep["v4 +identity"]

# clim scale fix check (row-level estimate vs target-derived estimate done in 03)
s_tr, s_te = clim_scale(tr), clim_scale(te)
print(f"\n[clim] HUCs with clim off by 1e3: train {(s_tr.groupby(tr.id).first() != 0).sum()}/{tr.id.nunique()}"
      f"  test {(s_te.groupby(te.id).first() != 0).sum()}/{te.id.nunique()}  (factor exponents {sorted(set(s_te.unique()))})")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for a, (lbl, x) in zip(ax, [("raw", r), ("repaired", f), ("ground truth", g)]):
    a.hist(np.sign(x) * np.log10(np.abs(np.nan_to_num(x, nan=1e-9)) + 1), bins=100, color="#4C72B0")
    a.set_title(f"{fam} corrupted cells: {lbl}"); a.set_xlabel("sign*log10(|v|+1)")
plt.tight_layout(); plt.savefig(FIG / "04_repair_check.png", dpi=110); plt.close()
