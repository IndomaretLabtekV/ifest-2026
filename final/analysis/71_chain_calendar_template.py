"""71 - Date the test origins with CALENDAR-SPECIFIC templates. 67 compared every pair with one decay curve averaged
over all seasons, and the global fit could not separate chain A 169 -> 94 -> 81 -> 14 (gaps 14,14,14) from chain B
81 -> 14 -> 169 -> 94 (gaps 14,18,14); trends in withdrawals (70) are too flat or weather-driven to decide.
Anomaly persistence depends on the season (snowpack carries winter into spring), so here the template for a pair
(earlier origin in calendar month m, later origin g months after) is the 12 x 12 lag-anomaly correlation matrix
averaged over every train instance with exactly that month and gap. Each train instance's distance to its own
template (leave-one-out) versus to wrong-gap templates gives the null distribution, so a test distance can be read
as a likelihood ratio rather than a raw SSE."""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG
from cvkit import te, test_view, tr, t

FT, FE = test_view("v7")
ANOM = [f"anom{k}" for k in range(12)]
month_t = pd.Series(tr.month.values, index=t).groupby(level=0).first()
TR = {tt: g.set_index("id")[ANOM] for tt, g in pd.concat([tr[["id"]].reset_index(drop=True), FT[ANOM]], axis=1)
      .assign(t=t).groupby("t")}
te_ = te.reset_index(drop=True)
TE = {o: g.set_index("id")[ANOM] for o, g in pd.concat([te_[["id", "origin_id"]], FE[ANOM]], axis=1).groupby("origin_id")}
orig_month = te_.groupby("origin_id").month.first().to_dict()


def cmat(X1, X2):
    X2 = X2.reindex(X1.index)
    a = X1.to_numpy(float); b = X2.to_numpy(float)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[ok] - a[ok].mean(0), b[ok] - b[ok].mean(0)
    return (a.T @ b) / np.sqrt(np.outer((a * a).sum(0), (b * b).sum(0)))


def instances(m, g):
    return [cmat(TR[s], TR[s + g]) for s in month_t.index if month_t[s] == m and s + g in TR]


cache = {}
def template(m, g):
    if (m, g) not in cache:
        cache[(m, g)] = np.array(instances(m, g))
    return cache[(m, g)]


def dist(M, T):
    return ((M - T) ** 2).mean()


# null: how far is a train instance from its own leave-one-out template vs from a template with another gap
print("[71] null distribution on train (mean squared distance x1000):")
null_rows = []
for m, g in [(12, 14), (4, 18), (2, 14), (10, 14)]:
    I = template(m, g)
    own = [dist(I[i], np.delete(I, i, 0).mean(0)) for i in range(len(I))]
    alt = [dist(I[i], template(m, g2).mean(0)) for i in range(len(I)) for g2 in (g - 12, g + 12) if g2 > 12 and len(template(m, g2))]
    null_rows.append(dict(month=m, gap=g, n=len(I), own=np.mean(own) * 1000, own_max=np.max(own) * 1000, wrong_gap=np.mean(alt) * 1000))
    print(f"  month {m:2d} gap {g}: n={len(I)} own-template {np.mean(own) * 1e3:.2f} (max {np.max(own) * 1e3:.2f}) | "
          f"template of gap +-12 {np.mean(alt) * 1e3:.2f}")

print("\n[71] test pairs: distance to the calendar template of every admissible gap (x1000):")
rows = []
for a, b in itertools.permutations(TE, 2):
    M = cmat(TE[a], TE[b])
    for g in range(13, 61):
        if (orig_month[a] + g - orig_month[b]) % 12 or not len(template(orig_month[a], g)):
            continue
        rows.append(dict(first=a, second=b, gap=g, d=dist(M, template(orig_month[a], g).mean(0)) * 1000,
                         c_near=M[0, 11]))
D = pd.DataFrame(rows)
for (a, b), g in D.groupby(["first", "second"]):
    g = g.sort_values("d")
    print(f"  {a:3d} -> {b:3d}: " + ", ".join(f"gap {r.gap}: {r.d:.2f}" for r in g.head(3).itertuples()))

HYP = {"A 169-94-81-14 (14,14,14)": [(169, 94, 14), (94, 81, 14), (81, 14, 14)],
       "B 81-14-169-94 (14,18,14)": [(81, 14, 14), (14, 169, 18), (169, 94, 14)]}
print("\n[71] chain hypotheses, distance of each adjacent link (x1000) and the contested link:")
for name, links in HYP.items():
    ds = [D[(D["first"] == a) & (D.second == b) & (D.gap == g)].d.iloc[0] for a, b, g in links]
    print(f"  {name}: links {np.round(ds, 2).tolist()} total {sum(ds):.2f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for k, (a, b, g) in enumerate([(94, 81, 14), (14, 169, 18)]):
    im = ax[k].imshow(cmat(TE[a], TE[b]) - template(orig_month[a], g).mean(0), cmap="RdBu", vmin=-.2, vmax=.2)
    ax[k].set_title(f"test {a}->{b} minus template (month {orig_month[a]}, gap {g})", fontsize=9)
    ax[k].set_xlabel(f"lag of {b}"); ax[k].set_ylabel(f"lag of {a}")
plt.colorbar(im, ax=ax[1])
nr = pd.DataFrame(null_rows)
ax[2].bar(np.arange(len(nr)) - .2, nr.own, .4, label="own template (LOO)")
ax[2].bar(np.arange(len(nr)) + .2, nr.wrong_gap, .4, label="gap +-12 template")
ax[2].set_xticks(range(len(nr)), [f"m{r.month} g{r.gap}" for r in nr.itertuples()])
ax[2].set_ylabel("mean squared distance x1000"); ax[2].set_title("Train null: how separable the gaps are"); ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "71_chain_calendar_template.png", dpi=120)
print(f"[fig] {FIG / '71_chain_calendar_template.png'}")

# global search with calendar templates: every order and every admissible gap, all 6 pairs scored
print("\n[71] global search over orders and gaps (sum of all 6 pair distances to calendar templates, x1000):")
Mt = {(a, b): cmat(TE[a], TE[b]) for a, b in itertools.permutations(TE, 2)}
fits = []
for order in itertools.permutations(TE):
    opts = [[g for g in range(13, 61) if (orig_month[a] + g - orig_month[b]) % 12 == 0] for a, b in zip(order[:-1], order[1:])]
    for gaps in itertools.product(*opts):
        pos = dict(zip(order, np.concatenate([[0], np.cumsum(gaps)])))
        if max(pos.values()) > 61:
            continue
        pairs = [(a, b, pos[b] - pos[a]) for a, b in itertools.permutations(order, 2) if pos[b] > pos[a]]
        if any(not len(template(orig_month[a], g)) for a, b, g in pairs):
            continue
        fits.append((sum(dist(Mt[(a, b)], template(orig_month[a], g).mean(0)) for a, b, g in pairs) * 1000, order, gaps))
fits.sort(key=lambda r: r[0])
for f in fits[:6]:
    print(f"  total {f[0]:7.2f}  order {f[1]}  gaps {f[2]}")
print(f"  hypotheses scored: {len(fits)}; margin best vs second {fits[1][0] - fits[0][0]:.2f}")
