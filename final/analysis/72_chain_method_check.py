"""72 - How reliable is the chain dating of 71? Run the exact same global search on train sets whose truth is known:
four train origins with the test's calendar months (10, 12, 2, 4) laid out like each top hypothesis of 71
(A: gaps 14,14,14 from October; B: 14,18,14 from February; C: 14,14,18 from December; D: 18,14,14 from April).
Templates are rebuilt without any instance that starts at one of the four chosen months (leave-set-out).
Accuracy = share of simulated sets whose best-scoring order and gaps equal the truth; also the margin distribution,
to read the test margin (9.21) against it."""
import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG
from cvkit import test_view, tr, t

FT, _ = test_view("v7")
ANOM = [f"anom{k}" for k in range(12)]
month_t = pd.Series(tr.month.values, index=t).groupby(level=0).first()
TR = {tt: g.set_index("id")[ANOM] for tt, g in pd.concat([tr[["id"]].reset_index(drop=True), FT[ANOM]], axis=1)
      .assign(t=t).groupby("t")}


def cmat(X1, X2):
    X2 = X2.reindex(X1.index)
    a = X1.to_numpy(float); b = X2.to_numpy(float)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[ok] - a[ok].mean(0), b[ok] - b[ok].mean(0)
    return (a.T @ b) / np.sqrt(np.outer((a * a).sum(0), (b * b).sum(0)))


INST = {}
for s in TR:
    for g in range(13, 61):
        if s + g in TR:
            INST[(s, g)] = cmat(TR[s], TR[s + g])


def template(m, g, exclude):
    I = [M for (s, gg), M in INST.items() if gg == g and month_t[s] == m and s not in exclude and s + g not in exclude]
    return np.mean(I, 0) if I else None


MODE = sys.argv[1] if len(sys.argv) > 1 else "raw"


def score(M, T):
    """raw: level-sensitive SSE (71). centered: SSE after removing each matrix's mean (year-level wetness).
    pearson: 1 - correlation of the 144 cells (shape only)."""
    if MODE == "centered":
        return (((M - M.mean()) - (T - T.mean())) ** 2).mean()
    if MODE == "pearson":
        return 1 - np.corrcoef(M.ravel(), T.ravel())[0, 1]
    return ((M - T) ** 2).mean()


def search(times):
    """times: dict label -> chain month. Returns sorted fits (score, order, gaps)."""
    mon = {k: month_t[v] for k, v in times.items()}
    ex = set(times.values())
    M = {(a, b): cmat(TR[times[a]], TR[times[b]]) for a, b in itertools.permutations(times, 2)}
    fits = []
    for order in itertools.permutations(times):
        opts = [[g for g in range(13, 61) if (mon[a] + g - mon[b]) % 12 == 0] for a, b in zip(order[:-1], order[1:])]
        for gaps in itertools.product(*opts):
            pos = dict(zip(order, np.concatenate([[0], np.cumsum(gaps)])))
            if max(pos.values()) > 61:
                continue
            pairs = [(a, b, pos[b] - pos[a]) for a, b in itertools.permutations(order, 2) if pos[b] > pos[a]]
            T = [template(mon[a], g, ex) for a, b, g in pairs]
            if any(x is None for x in T):
                continue
            fits.append((sum(score(M[(a, b)], x) for (a, b, _), x in zip(pairs, T)) * 1000, order, tuple(gaps)))
    return sorted(fits, key=lambda r: r[0])


LAYOUTS = {"A 10,12,2,4 gaps 14,14,14": (10, (14, 14, 14)), "B 2,4,10,12 gaps 14,18,14": (2, (14, 18, 14)),
           "C 12,2,4,10 gaps 14,14,18": (12, (14, 14, 18)), "D 4,10,12,2 gaps 18,14,14": (4, (18, 14, 14))}
rows = []
for name, (m0, gaps) in LAYOUTS.items():
    for s in [s for s in month_t.index if month_t[s] == m0]:
        ts = np.concatenate([[s], s + np.cumsum(gaps)])
        if ts[-1] not in TR:
            continue
        times = {f"o{i}": int(x) for i, x in enumerate(ts)}
        fits = search(times)
        truth = (tuple(times), gaps)
        best = (fits[0][1], fits[0][2])
        rank = next(i for i, f in enumerate(fits) if (f[1], f[2]) == truth)
        # does the best fit at least give the right set of 14-month successors?
        pos = dict(zip(best[0], np.concatenate([[0], np.cumsum(best[1])])))
        succ = {a for a in times for b in times if pos[b] - pos[a] == 14}
        true_succ = {f"o{i}" for i, g in enumerate(gaps) if g == 14}
        rows.append(dict(layout=name, start=s, correct=rank == 0, rank=rank, margin=fits[1][0] - fits[0][0],
                         succ_ok=succ == true_succ, n_hyp=len(fits)))
        print(f"  {name} start {s:3d}: truth rank {rank}, margin {fits[1][0] - fits[0][0]:6.2f}, "
              f"14-month successors right: {succ == true_succ}", flush=True)
R = pd.DataFrame(rows)
print(f"\n[72] MODE={MODE}: reliability of the calendar-template chain search on train sets with known order:")
print(R.groupby("layout").agg(n=("correct", "size"), exact=("correct", "mean"), successors=("succ_ok", "mean"),
                              median_margin=("margin", "median")).round(3).to_string())
print(f"[72] overall exact {R.correct.mean():.3f}, successor sets right {R.succ_ok.mean():.3f}; "
      f"among correct fits, share with margin >= 9.21 (the test's): {(R[R.correct].margin >= 9.21).mean():.2f}; "
      f"wrong fits with margin >= 9.21: {(~R.correct & (R.margin >= 9.21)).sum()} of {(~R.correct).sum()}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(R[R.correct].margin, bins=20, alpha=.6, label="best fit = truth")
ax.hist(R[~R.correct].margin, bins=20, alpha=.6, label="best fit wrong")
ax.axvline(9.21, color="r", ls="--", label="test margin 9.21")
ax.set_xlabel("score margin, best vs second hypothesis"); ax.set_ylabel("simulated train sets")
ax.set_title("Chain dating on train sets with known order")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / f"72_chain_method_check_{MODE}.png", dpi=120)
print(f"[fig] {FIG / f'72_chain_method_check_{MODE}.png'}")
