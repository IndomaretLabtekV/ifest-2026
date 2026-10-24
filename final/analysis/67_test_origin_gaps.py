"""67 - How far apart are the four test origins, and does it matter?
The handover assumed "no leakage inside the test: origins are > 12 months apart, so no window holds another origin's
TARGET month". That only rules out the target month itself. The origins have calendar months 10, 12, 2, 4 and must sit
in Aug 2015 - Aug 2020, so gaps are 13 + something; a gap of 14 months means origin B's lag11 is month A+3, only two
months after A's target. Supply anomalies are persistent, so that month could say a lot about A's target.
(A) train chain: how much does the anomaly at target+d (d = 1..14) tell about the label?
(B) test: find each pair's gap by matching the cross-sectional correlation of their 12x12 lag anomalies with the decay
    curve rho(distance) measured on the train chain, over the gaps allowed by the calendar months."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import FIG, WORK
from cvkit import te, test_view, tr, y, t

FT, FE = test_view("v7")
te = te.reset_index(drop=True)  # FE rows follow this order (cvkit.test_view)

# ---------------- (A) information value of months after the target, train chain ----------------
A = pd.DataFrame({"id": tr.id.values, "t": t, "a": FT.anom0.values}).pivot_table(index="id", columns="t", values="a")
row_a = lambda d: A.stack().reindex(pd.MultiIndex.from_arrays([tr.id.values, t + 1 + d])).to_numpy()
print("[67A] label(t) = supply(t+1) <= 0.6 clim. Univariate skill of the anomaly at month t+1+d (train chain):")
print(f"   d=-1 (lag0, persistence) AP {ap(y, -np.nan_to_num(FT.anom0.values, nan=0)):.3f}")
rowsA = []
for d in [1, 2, 3, 4, 6, 9, 12, 14]:
    v = row_a(d)
    ok = np.isfinite(v)
    rowsA.append(dict(d=d, cover=ok.mean(), ap=ap(y[ok], -v[ok]), auc=auc(y[ok], -v[ok]),
                      corr_base=np.corrcoef(v[ok], np.nan_to_num(FT.anom0.values[ok]))[0, 1]))
    print(f"   d={d:2d} (target+{d}) AP {rowsA[-1]['ap']:.3f} AUC {rowsA[-1]['auc']:.3f}  (coverage {ok.mean():.2f})")

# cross-sectional decay curve rho(k): corr over HUCs of anomaly at month m and m+k, averaged over m
M = A.to_numpy()
rho = {}
for k in range(1, 73):
    cs = []
    for m in range(0, M.shape[1] - k, 3):
        a, b = M[:, m], M[:, m + k]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 200:
            cs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    rho[k] = np.mean(cs)
print("[67A] cross-sectional anomaly correlation vs distance (months): " +
      ", ".join(f"{k}:{rho[k]:.2f}" for k in [1, 2, 3, 6, 12, 13, 14, 15, 24, 26, 36]))

# ---------------- (B) test: gap per pair of origins ----------------
lagA = [f"anom{k}" for k in range(12)]
E = pd.concat([te[["id", "origin_id", "month"]], FE[lagA]], axis=1)
orig = E.groupby("origin_id").month.first().to_dict()
print(f"\n[67B] test origins and calendar months: {orig}")
res = []
for o1, o2 in itertools.permutations(orig, 2):
    X1 = E[E.origin_id == o1].set_index("id")[lagA]
    X2 = E[E.origin_id == o2].set_index("id")[lagA].reindex(X1.index)
    C = np.array([[pd.Series(X1[f"anom{i}"]).corr(pd.Series(X2[f"anom{j}"])) for j in range(12)] for i in range(12)])
    for g in range(13, 61):
        if (orig[o1] + g - orig[o2]) % 12:
            continue
        # o2 = o1 + g: lag i of o1 is month o1-i, lag j of o2 is month o1+g-j -> distance g - j + i
        pred = np.array([[rho[g - j + i] for j in range(12)] for i in range(12)])
        res.append(dict(first=o1, second=o2, gap=g, sse=((C - pred) ** 2).sum(), c_near=C[0, 11],
                        rho_near=pred[0, 11], mean_c=C.mean(), mean_pred=pred.mean()))
R = pd.DataFrame(res).sort_values("sse")
print("[67B] best gap hypotheses (o2 = o1 + gap months); c_near = corr(o1 lag0, o2 lag11), the closest months:")
print(R.head(12).round(3).to_string(index=False))
R.to_csv(WORK / "67_gap_fit.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
d = pd.DataFrame(rowsA)
ax[0].bar(d.d.astype(str), d.ap, color="#4C72B0")
ax[0].axhline(ap(y, -np.nan_to_num(FT.anom0.values, nan=0)), color="r", ls="--", label="persistence (lag0) AP")
ax[0].set_xlabel("months after the target month (d)")
ax[0].set_ylabel("univariate AP for label(t)")
ax[0].set_title("Information in months AFTER the target (train chain)")
ax[0].legend()
ax[1].plot(list(rho), list(rho.values()), marker=".")
ax[1].set_xlabel("distance between months"); ax[1].set_ylabel("cross-sectional corr of anomalies")
ax[1].set_title("Decay curve used to date the test origins")
plt.tight_layout()
plt.savefig(FIG / "67_test_origin_gaps.png", dpi=120)
print(f"[fig] {FIG / '67_test_origin_gaps.png'}")
