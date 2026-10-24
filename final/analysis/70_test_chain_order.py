"""70 - Which order do the test origins have? 67 pins two links for sure (81 -> 14 and 169 -> 94, 14 months each) but
the global correlation fit is ambiguous between chain A 169 -> 94 -> 81 -> 14 (gaps 14, 14, 14) and chain B
81 -> 14 -> 169 -> 94 (gaps 14, 18, 14). Independent evidence from slow trends instead of weather:
a 12-lag window is a full year, so its SUM of withdrawals has no seasonality; the per-HUC random factor of the test is
the same for all four origins, so log(sum at origin b / sum at origin a) of one HUC is a pure time difference.
(1) train: yearly trend of the annual withdrawal sums (public supply, thermoelectric, irrigation) and of baseflow;
(2) test: median pairwise log ratios; (3) fit the offset of each chain hypothesis to the train trend curve."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from common import FIG
from cvkit import te, tr, t
from prep import repair

FAM = {"ps": "q_ps_wd_inc_m3", "th": "q_th_wd_inc_m3", "ir": "q_ir_wd_inc_m3", "bf": "q_bf_inc_m3"}


def annual(df):
    x = repair(df)
    return pd.DataFrame({f: np.log1p(sum(x[f"{c}_lag{k}"].abs() for k in range(12))) for f, c in FAM.items()},
                        index=df.index)


S_tr = annual(tr)
S_tr["id"], S_tr["t"] = tr.id.values, t
# train trend: per HUC centre on its own mean, then median over HUCs per chain month (windows ending at t)
cen = S_tr.groupby("id")[list(FAM)].transform(lambda s: s - s.mean())


def tmean(v):
    """trimmed mean over HUCs whose value actually moves (most HUCs repeat the same withdrawal every year)"""
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (np.abs(v) > 1e-9)]
    return trim_mean(v, 0.1) if len(v) > 50 else 0.0


moving = {f: cen[f].abs().groupby(S_tr.id).max() > 1e-6 for f in FAM}
print("[70] share of HUCs whose annual sum ever changes: " + ", ".join(f"{f} {m.mean():.2f}" for f, m in moving.items()))
curve = pd.DataFrame({f: cen[f].groupby(t).apply(tmean) for f in FAM})
print("[70] train: median centred log annual sum by window end (every 12 months):")
print(curve.iloc[::12].round(3).to_string())
slope = {f: np.polyfit(curve.index[curve.index >= 100], curve[f][curve.index >= 100], 1)[0] * 12 for f in FAM}
print("[70] late-train slope per year (t >= 100): " + ", ".join(f"{f} {v:+.3f}" for f, v in slope.items()))

te_ = te.reset_index(drop=True)
S_te = annual(te_)
S_te["id"], S_te["o"] = te_.id.values, te_.origin_id.values
P = S_te.pivot_table(index="id", columns="o", values=list(FAM))
orig = sorted(te_.origin_id.unique())
print("\n[70] test: median over HUCs of log(annual sum at column origin / at row origin):")
for f in ["ps", "th", "ir", "bf"]:
    M = pd.DataFrame({b: {a: tmean(P[(f, b)] - P[(f, a)]) for a in orig} for b in orig})
    print(f"  {f}:\n" + M.round(3).to_string())

# fit: for a chain hypothesis with positions pos (months), predicted diff b-a = slope * (pos_b - pos_a) / 12
HYP = {"A 169-94-81-14 (14,14,14)": {169: 0, 94: 14, 81: 28, 14: 42},
       "B 81-14-169-94 (14,18,14)": {81: 0, 14: 14, 169: 32, 94: 46}}
print("\n[70] fit of each hypothesis to the public-supply and thermoelectric trends (sum of squared errors):")
res = {}
for name, pos in HYP.items():
    err = 0.0
    for f in ["ps", "th"]:
        for a, b in itertools.combinations(orig, 2):
            obs = tmean(P[(f, b)] - P[(f, a)])
            err += (obs - slope[f] * (pos[b] - pos[a]) / 12) ** 2
    res[name] = err
    print(f"  {name}: SSE {err:.5f}")
# model-free check: which origin is earliest / latest by public supply level
lvl = {o: tmean(P[("ps", o)] - P[("ps", orig[0])]) for o in orig}
print("[70] public-supply level relative to origin", orig[0], "(declining trend -> higher = earlier):",
      {o: round(v, 3) for o, v in sorted(lvl.items(), key=lambda kv: -kv[1])})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for f in FAM:
    ax[0].plot(curve.index, curve[f], label=f)
ax[0].set_xlabel("train chain month (window end)"); ax[0].set_ylabel("median centred log annual sum")
ax[0].set_title("Slow trends in annual sums (train)"); ax[0].legend()
for name, pos in HYP.items():
    xs = [pos[o] for o in orig]
    ax[1].scatter(xs, [lvl[o] for o in orig], label=name, s=60)
    for o in orig:
        ax[1].annotate(str(o), (pos[o], lvl[o]))
ax[1].set_xlabel("hypothesised position (months)"); ax[1].set_ylabel("public-supply level vs first origin")
ax[1].set_title("Test origins: public-supply level vs hypothesised time"); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "70_test_chain_order.png", dpi=120)
print(f"[fig] {FIG / '70_test_chain_order.png'}")
