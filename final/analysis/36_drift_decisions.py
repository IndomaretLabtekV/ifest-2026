"""36 - Concept drift and model decisions under the forward scheme (34 showed the test is 1-6 years ahead).
A) Horizon: fixed 5-year training window, gap 12/24/36/48 months before the validation year.
   If AP falls with the gap, older data is less relevant and recency weighting should help.
B) Window: all past data vs last 5 years vs last 8 years, gap 12.
C) Decisions re-judged by L5 (blocked, old harness) vs L6 (forward) vs L7 (leave-one-year-out):
   V6 dynamics features (public said V6 < V5) and recency weights. A scheme that agrees with the public
   verdict on V6 earns more trust for the next decisions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, WORK
from cvkit import DYN, base_cols, date_of, run_scheme, t, test_like, views, year

TV, VV = views()
cols = base_cols(TV)

# A) horizon
rows = []
for k in (10, 11, 12, 13):
    for gap in (12, 24, 36, 48):
        end = 12 * k - gap
        _, tab = run_scheme(TV, VV, [((t < end) & (t >= end - 60), year == k)], cols)
        rows.append(dict(val_year=k, gap=gap, ap=tab.ap[0]))
        print(f"[horizon] val year {k} gap {gap:2d}: AP {tab.ap[0]:.4f}", flush=True)
H = pd.DataFrame(rows)
Hp = H.pivot(index="val_year", columns="gap", values="ap")
rel = Hp.sub(Hp[12], axis=0)
print("\n[horizon] AP by gap (rows: validation year):\n" + Hp.round(4).to_string())
print("[horizon] change vs gap 12, mean over years: " + ", ".join(f"gap {g}: {v:+.4f}" for g, v in rel.mean().items()))
slope = np.polyfit(np.repeat(Hp.columns.values[None], len(Hp), 0).ravel() / 12, rel.values.ravel(), 1)[0]
print(f"[horizon] fitted drift: {slope:+.4f} AP per extra year of gap")

# B) training window length, gap 12, forward years 8..13
win = {}
for name, lo in [("all past", None), ("last 8y", 96), ("last 5y", 60)]:
    folds = [((t < 12 * k - 12) & ((t >= 12 * k - 12 - lo) if lo else True), year == k) for k in range(8, 14)]
    _, tab = run_scheme(TV, VV, folds, cols)
    win[name] = tab.ap.values
    print(f"[window] {name:8s} AP {tab.ap.mean():.4f} per-fold {np.round(tab.ap.values, 4).tolist()}", flush=True)

# C) decisions under three schemes
edges = np.linspace(t.min(), t.max() + 1, 5).astype(int)
SCH = {
    "L5 blocked": [((t < lo - 12) | (t >= hi + 12), (t >= lo) & (t < hi)) for lo, hi in zip(edges[:-1], edges[1:])],
    "L6 forward": [(t < 12 * k - 12, year == k) for k in range(8, 14)],
    "L7 LOYO": [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)],
}


out = []
for sname, folds in SCH.items():
    base_oof, base = run_scheme(TV, VV, folds, cols)
    tl_base = test_like(base_oof, n=600)
    cands = {"+V6 dynamics": dict(cols=cols + DYN)}
    if sname == "L6 forward":
        cands["recency hl 3y"] = dict(cols=cols, hl=3)
        cands["recency hl 6y"] = dict(cols=cols, hl=6)
    for cname, c in cands.items():
        if "hl" in c:
            oof = np.full(len(t), np.nan)
            aps = []
            for fit, va in folds:
                end = t[fit].max()
                w = 0.5 ** ((end - t) / (12 * c["hl"]))
                o, tab = run_scheme(TV, VV, [(fit, va)], c["cols"], weight=w)
                oof[va] = o[va]
                aps.append(tab.ap[0])
            aps = np.array(aps)
        else:
            oof, tab = run_scheme(TV, VV, folds, c["cols"])
            aps = tab.ap.values
        d = aps - base.ap.values
        tl = test_like(oof, n=600)
        out.append(dict(scheme=sname, change=cname, fold_gain=d.mean(), folds_up=f"{(d > 0).sum()}/{len(d)}",
                        test_like_gain=(tl - tl_base).mean(), draws_up=(tl > tl_base).mean()))
        print(f"[decision] {sname:10s} {cname:14s} fold gain {d.mean():+.4f} ({(d > 0).sum()}/{len(d)} folds up) | "
              f"test-like gain {(tl - tl_base).mean():+.4f}, improved in {(tl > tl_base).mean():.0%} of paired draws", flush=True)
D = pd.DataFrame(out)
D.to_csv(WORK / "36_decisions.csv", index=False)
print("\n" + D.round(4).to_string(index=False))
print("[reference] public: V5 0.7552 -> V6 (+dynamics) 0.751, i.e. -0.0042")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k, r in Hp.iterrows():
    ax[0].plot(Hp.columns / 12, r.values, "o-", label=f"validate {date_of(12 * k)}")
ax[0].set_xlabel("gap between last training month and validation year (years)")
ax[0].set_ylabel("AP")
ax[0].set_title(f"Drift: {slope:+.4f} AP per extra year of horizon")
ax[0].legend(fontsize=8)
ax[1].bar(list(win), [v.mean() for v in win.values()], color="tab:blue")
ax[1].set_ylim(min(v.mean() for v in win.values()) - .02, max(v.mean() for v in win.values()) + .01)
ax[1].set_title("Forward AP by training window (gap 12)")
plt.tight_layout()
plt.savefig(FIG / "36_drift_decisions.png", dpi=120)
print(f"[fig] {FIG / '36_drift_decisions.png'}")
