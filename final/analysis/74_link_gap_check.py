"""74 - Per-link check of the chain A 169 -> 94 -> 81 -> 14: is each link 14 months, or 26?
73 rejected every hypothesis where 169 follows 14 by 18 months (deterministic public-supply calendar-year test).
Remaining doubt per link: 14 vs 26 months. Statistic: c_near = corr over HUCs of the earlier origin's lag0 anomaly and
the later origin's lag11 anomaly (distance 3 months if the gap is 14, 15 if 26). Train gives the distribution of c_near
for EXACTLY the same calendar configuration and each gap; the test value is read as a likelihood ratio (normal fit)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import FIG
from cvkit import te, test_view, tr, t

FT, FE = test_view("v7")
month_t = pd.Series(tr.month.values, index=t).groupby(level=0).first()
a0 = pd.DataFrame({"id": tr.id.values, "t": t, "a0": FT.anom0.values, "a11": FT.anom11.values})
A0, A11 = a0.pivot_table(index="id", columns="t", values="a0"), a0.pivot_table(index="id", columns="t", values="a11")
te_ = te.reset_index(drop=True)
E = pd.concat([te_[["id", "origin_id"]], FE[["anom0", "anom11"]]], axis=1)
orig = te_.groupby("origin_id").month.first().to_dict()


def c_train(m, g):
    out = []
    for s in month_t.index[month_t.values == m]:
        if s + g in A11.columns:
            x, z = A0[s], A11[s + g]
            ok = x.notna() & z.notna()
            out.append(np.corrcoef(x[ok], z[ok])[0, 1])
    return np.array(out)


rows = []
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, (a, b) in enumerate([(169, 94), (94, 81), (81, 14)]):
    x = E[E.origin_id == a].set_index("id").anom0
    z = E[E.origin_id == b].set_index("id").anom11.reindex(x.index)
    ok = x.notna() & z.notna()
    c = np.corrcoef(x[ok], z[ok])[0, 1]
    d14, d26 = c_train(orig[a], 14), c_train(orig[a], 26)
    lr = norm.pdf(c, d14.mean(), d14.std()) / norm.pdf(c, d26.mean(), d26.std())
    rows.append(dict(link=f"{a}->{b}", test_c_near=c, gap14_mean=d14.mean(), gap14_sd=d14.std(), gap26_mean=d26.mean(),
                     gap26_sd=d26.std(), n14=len(d14), n26=len(d26), LR_14_vs_26=lr))
    ax[i].hist(d14, bins=10, alpha=.6, label="train gap 14"); ax[i].hist(d26, bins=10, alpha=.6, label="train gap 26")
    ax[i].axvline(c, color="r", lw=2, label=f"test {c:.2f}")
    ax[i].set_title(f"link {a} -> {b} (start month {orig[a]})"); ax[i].set_xlabel("c_near"); ax[i].legend(fontsize=8)
R = pd.DataFrame(rows)
print("[74] c_near of each chain-A link vs the train distribution for the same calendar configuration:")
print(R.round(3).to_string(index=False))
plt.tight_layout()
plt.savefig(FIG / "74_link_gap_check.png", dpi=120)
print(f"[fig] {FIG / '74_link_gap_check.png'}")

# whole 12x12 matrix: score = dist(M, template gap 14) - dist(M, template gap 26), leave-one-out on train instances
ANOM = [f"anom{k}" for k in range(12)]
TRW = {tt: g.set_index("id")[ANOM] for tt, g in pd.concat([tr[["id"]].reset_index(drop=True), FT[ANOM]], axis=1).assign(t=t).groupby("t")}
TEW = {o: g.set_index("id")[ANOM] for o, g in pd.concat([te_[["id", "origin_id"]], FE[ANOM]], axis=1).groupby("origin_id")}


def cm(X1, X2):
    X2 = X2.reindex(X1.index)
    a, b = X1.to_numpy(float), X2.to_numpy(float)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[ok] - a[ok].mean(0), b[ok] - b[ok].mean(0)
    return (a.T @ b) / np.sqrt(np.outer((a * a).sum(0), (b * b).sum(0)))


print("\n[74b] whole-matrix score = dist to gap-14 template - dist to gap-26 template (x1000), train LOO vs test:")
for a, b in [(169, 94), (94, 81), (81, 14)]:
    m = orig[a]
    inst = {g: {s: cm(TRW[s], TRW[s + g]) for s in month_t.index[month_t.values == m] if s + g in TRW} for g in (14, 26)}
    def score(M, skip=None):
        T14 = np.mean([v for s, v in inst[14].items() if s != skip], 0)
        T26 = np.mean([v for s, v in inst[26].items() if s != skip], 0)
        return (((M - T14) ** 2).mean() - ((M - T26) ** 2).mean()) * 1000
    s14 = np.array([score(M, s) for s, M in inst[14].items()])
    s26 = np.array([score(M, s) for s, M in inst[26].items()])
    st = score(cm(TEW[a], TEW[b]))
    lr = norm.pdf(st, s14.mean(), s14.std()) / norm.pdf(st, s26.mean(), s26.std())
    print(f"  {a}->{b}: train true-14 {s14.mean():+.1f} (sd {s14.std():.1f}, max {s14.max():+.1f}) | true-26 {s26.mean():+.1f} "
          f"(sd {s26.std():.1f}, min {s26.min():+.1f}) | test {st:+.1f} | LR(14:26) {lr:.3g}")
