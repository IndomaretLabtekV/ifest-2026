"""78 - Post-mortem of V16 (public 0.733, V15 0.754; CV said +0.025). The Kaggle run reproduced the local chain and CV,
so the implementation matches the design; the premise must be wrong somewhere. Suspect: the chain evidence (67, 71, 74)
was correlation-based, and a PERSISTENT per-HUC anomaly in the test period (a HUC that stays wetter or drier than its
climatology through 2015-2020, or a HUC-specific error of the test climatology) correlates every pair of test origins
whatever their distance, inflating c_near and the template fits.
(1) cross-origin correlation at implied distances where the train decay curve is ~0: is the test far above it?
(2) share of anomaly variance that is a per-HUC constant over the 4 windows, test vs train windows laid out like chain A;
(3) confounder-free link test: remove each HUC's mean anomaly over the OTHER two origins' windows (same in train),
    then redo the gap-14 vs gap-26 likelihood ratio of 74;
(4) how V16 moved the ranking versus V15, per origin."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from common import FIG, ROOT
from cvkit import te, test_view, tr, t

FT, FE = test_view("v7")
ANOM = [f"anom{k}" for k in range(12)]
te_ = te.reset_index(drop=True)
orig = te_.groupby("origin_id").month.first().to_dict()
TE = {o: g.set_index("id")[ANOM] for o, g in pd.concat([te_[["id", "origin_id"]], FE[ANOM]], axis=1).groupby("origin_id")}
TR = {tt: g.set_index("id")[ANOM] for tt, g in pd.concat([tr[["id"]].reset_index(drop=True), FT[ANOM]], axis=1).assign(t=t).groupby("t")}
month_t = pd.Series(tr.month.values, index=t).groupby(level=0).first()
CHAIN_A = {169: 0, 94: 14, 81: 28, 14: 42}


def cmat(X1, X2):
    a, b = X1.to_numpy(float), X2.reindex(X1.index).to_numpy(float)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[ok] - a[ok].mean(0), b[ok] - b[ok].mean(0)
    return (a.T @ b) / np.sqrt(np.outer((a * a).sum(0), (b * b).sum(0)))


# (1) mean cross-origin correlation, test vs train at the chain-A distance of each pair
print("[78] (1) mean of the 12x12 cross-origin correlation matrix: test vs train at the same calendar layout and "
      "chain-A gap")
rows = []
for a, b in itertools.combinations(sorted(CHAIN_A, key=CHAIN_A.get), 2):
    g = CHAIN_A[b] - CHAIN_A[a]
    tr_means = [cmat(TR[s], TR[s + g]).mean() for s in month_t.index[month_t.values == orig[a]] if s + g in TR]
    rows.append(dict(pair=f"{a}->{b}", gap_chainA=g, test_mean=cmat(TE[a], TE[b]).mean(),
                     train_mean=np.mean(tr_means), train_sd=np.std(tr_means), n_train=len(tr_means)))
R1 = pd.DataFrame(rows)
R1["z"] = (R1.test_mean - R1.train_mean) / R1.train_sd
print(R1.round(3).to_string(index=False))

# (2) per-HUC persistent component: variance share of the HUC mean over the 4 windows (48 months)
def persistent_share(frames):
    X = pd.concat([f.add_prefix(f"w{i}_") for i, f in enumerate(frames)], axis=1).dropna()
    m = X.mean(1)
    return m.var() / X.stack().var(), len(X)


ps_te, n_te = persistent_share([TE[o] for o in sorted(CHAIN_A, key=CHAIN_A.get)])
shares = []
for s in month_t.index[month_t.values == 10]:
    if s + 42 in TR:
        shares.append(persistent_share([TR[s], TR[s + 14], TR[s + 28], TR[s + 42]])[0])
print(f"\n[78] (2) share of anomaly variance that is a per-HUC constant over the 4 windows: test {ps_te:.3f} (n={n_te}) | "
      f"train chain-A layouts {np.mean(shares):.3f} (sd {np.std(shares):.3f}, max {np.max(shares):.3f}, {len(shares)} layouts)")

# (3) confounder-free c_near: subtract each HUC's mean anomaly over the other two origins' windows
def residual(frames, i, j):
    others = [f for k, f in enumerate(frames) if k not in (i, j)]
    base = pd.concat(others, axis=1).mean(1)
    return frames[i].sub(base.reindex(frames[i].index), axis=0), frames[j].sub(base.reindex(frames[j].index), axis=0)


def c_near(X1, X2):
    x, z = X1.anom0, X2.anom11.reindex(X1.index)
    ok = x.notna() & z.notna()
    return np.corrcoef(x[ok], z[ok])[0, 1]


order = sorted(CHAIN_A, key=CHAIN_A.get)
frames_te = [TE[o] for o in order]
print("\n[78] (3) link test after removing the per-HUC persistent component (gap 14 vs 26, same calendar layout in train):")
rows = []
for i in range(3):
    a, b = order[i], order[i + 1]
    raw = c_near(TE[a], TE[b])
    r1, r2 = residual(frames_te, i, i + 1)
    cor = c_near(r1, r2)
    dist = {}
    for g in (14, 26):
        vals_raw, vals_res = [], []
        for s in month_t.index[month_t.values == orig[a]]:
            # train layout: the two windows at s, s+g plus two 'other' windows 14 months before/after as in the test
            others = [u for u in (s - 28, s - 14, s + g + 14, s + g + 28) if u in TR][:2]
            if s + g not in TR or len(others) < 2:
                continue
            fr = [TR[s], TR[s + g]] + [TR[u] for u in others]
            vals_raw.append(c_near(TR[s], TR[s + g]))
            x1, x2 = residual(fr, 0, 1)
            vals_res.append(c_near(x1, x2))
        dist[g] = (np.array(vals_raw), np.array(vals_res))
    lr_raw = norm.pdf(raw, dist[14][0].mean(), dist[14][0].std()) / norm.pdf(raw, dist[26][0].mean(), dist[26][0].std())
    lr_res = norm.pdf(cor, dist[14][1].mean(), dist[14][1].std()) / norm.pdf(cor, dist[26][1].mean(), dist[26][1].std())
    rows.append(dict(link=f"{a}->{b}", test_raw=raw, test_resid=cor, tr14_resid=dist[14][1].mean(),
                     tr26_resid=dist[26][1].mean(), LR_raw=lr_raw, LR_resid=lr_res))
R3 = pd.DataFrame(rows)
print(R3.round(3).to_string(index=False))

# (4) V16 vs V15 ranking, per origin
s15 = pd.read_csv(ROOT / "results/v15/submission.csv").set_index("row_id").label
s16 = pd.read_csv(ROOT / "results/v16/submission.csv").set_index("row_id").label
o = te_.set_index("row_id").origin_id.reindex(s15.index)
print("\n[78] (4) V16 vs V15 by origin: within-origin Spearman, and mean rank of the origin in the pooled ranking")
r15, r16 = s15.rank(pct=True), s16.rank(pct=True)
for oo in order:
    m = (o == oo).values
    print(f"  origin {oo:3d} (month {orig[oo]:2d}): spearman {spearmanr(s15[m], s16[m])[0]:.3f} | pooled mean rank "
          f"V15 {r15[m].mean():.3f} -> V16 {r16[m].mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(len(R1))
ax[0].bar(x - .2, R1.train_mean, .4, yerr=R1.train_sd, label="train, same layout (mean +- sd)")
ax[0].bar(x + .2, R1.test_mean, .4, label="test")
ax[0].set_xticks(x, [f"{p}\n(gap {g})" for p, g in zip(R1.pair, R1.gap_chainA)], fontsize=8)
ax[0].set_ylabel("mean cross-origin anomaly correlation"); ax[0].legend(fontsize=8)
ax[0].set_title("Test origins correlate at every distance")
x = np.arange(len(R3))
ax[1].bar(x - .3, R3.test_raw, .2, label="test raw c_near")
ax[1].bar(x - .1, R3.test_resid, .2, label="test after removing HUC constant")
ax[1].bar(x + .1, R3.tr14_resid, .2, label="train gap 14 (residual)")
ax[1].bar(x + .3, R3.tr26_resid, .2, label="train gap 26 (residual)")
ax[1].set_xticks(x, R3.link); ax[1].legend(fontsize=7); ax[1].set_title("Link evidence without the persistent component")
plt.tight_layout()
plt.savefig(FIG / "78_v16_postmortem.png", dpi=120)
print(f"[fig] {FIG / '78_v16_postmortem.png'}")
