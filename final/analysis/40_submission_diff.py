"""40 - Where exactly does V7 differ from V5 on the test? (public: V5 0.7552, V6 0.751, V7 0.748)
No labels are needed: the rank change of every test row is related to the segments V6/V7 touched
(dynamics features for everyone, the v7 repair on rows whose lag-0 anomaly moved, seen vs unseen HUCs,
zone relative to the 0.6 threshold, origin). A change concentrated in one segment tells which change drove
the public difference, and designs the probe of 42."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import FIG, ROOT
from cvkit import te, test_view, tr

THR = np.log(0.6)
_, E7 = test_view("v7")
_, E6 = test_view("v6")
tt = te.reset_index(drop=True)
S = {v: pd.read_csv(ROOT / "results" / v / "submission.csv").set_index("row_id").label.reindex(tt.row_id).to_numpy()
     for v in ("v5", "v6", "v7")}
R = {v: rankdata(s) / len(s) for v, s in S.items()}
d = pd.DataFrame({
    "origin": tt.origin_id.values, "seen": tt.id.isin(set(tr.id)).values,
    "zone": pd.cut(E7.anom0.values - THR, [-np.inf, 0, .5, np.inf], labels=["dry", "onset", "safe"]),
    "repair_moved": (E7.anom0 - E6.anom0).abs().values > 0.01,
    "since_below": E7.since_below.values, "d57": R["v7"] - R["v5"], "d56": R["v6"] - R["v5"], "d67": R["v7"] - R["v6"],
})
print(f"[rows] {len(d)}; repair moved anom0 by >0.01 on {d.repair_moved.mean():.3%} of test rows "
      f"(dry zone {d.repair_moved[d.zone == 'dry'].mean():.3%})")
for a, b in [("v5", "v6"), ("v6", "v7"), ("v5", "v7")]:
    top_a, top_b = R[a] >= .8, R[b] >= .8
    print(f"[{a}->{b}] spearman {spearmanr(R[a], R[b]).correlation:.4f} | mean |rank change| {np.abs(R[b] - R[a]).mean():.4f} | "
          f"rows entering/leaving the top 20%: {(top_b & ~top_a).sum()}/{(top_a & ~top_b).sum()}")

print("\n[segments] mean |rank change| and mean signed change (V5->V6 = dynamics, V6->V7 = repair):")
rows = []
for col in ["origin", "seen", "zone", "repair_moved"]:
    for k, g in d.groupby(col, observed=True):
        rows.append(dict(segment=f"{col}={k}", n=len(g), abs_56=g.d56.abs().mean(), mean_56=g.d56.mean(),
                         abs_67=g.d67.abs().mean(), mean_67=g.d67.mean()))
T = pd.DataFrame(rows).set_index("segment")
print(T.round(4).to_string())

# which rows moved most from V5 to V7, and are they concentrated anywhere
big = d.d57.abs() >= d.d57.abs().quantile(.95)
print(f"\n[top 5% movers V5->V7] n={big.sum()}: zone {d.zone[big].value_counts(normalize=True).round(3).to_dict()} "
      f"vs all {d.zone.value_counts(normalize=True).round(3).to_dict()}")
print(f"  since_below median movers {d.since_below[big].median():.0f} vs all {d.since_below.median():.0f}; "
      f"repair_moved share {d.repair_moved[big].mean():.3f} vs {d.repair_moved.mean():.3f}")
print(f"  signed change of movers by zone: {d[big].groupby('zone', observed=True).d57.mean().round(4).to_dict()}")

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
ax[0].hist(d.d56, bins=100, alpha=.6, label="V5->V6 (dynamics)")
ax[0].hist(d.d67, bins=100, alpha=.6, label="V6->V7 (repair)")
ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Rank change per test row")
d.boxplot(column="d56", by="zone", ax=ax[1], showfliers=False)
ax[1].set_title("V5->V6 rank change by zone"); ax[1].set_xlabel("")
sb = d.groupby("since_below").d56.mean()
ax[2].bar(sb.index, sb.values, color="tab:blue")
ax[2].set_xlabel("since_below (months since last below 0.6)"); ax[2].set_title("V5->V6 mean rank change by since_below")
plt.suptitle("")
plt.tight_layout()
plt.savefig(FIG / "40_submission_diff.png", dpi=120)
print(f"[fig] {FIG / '40_submission_diff.png'}")
