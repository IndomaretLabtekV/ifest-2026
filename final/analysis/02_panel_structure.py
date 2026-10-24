"""02 - Panel structure: rows per origin/HUC, duplicates, train/test overlap,
and reconstruction of the true calendar order of the shuffled origin_id."""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, LAG_COLS, WORK, load_clean

tr, te = load_clean()
te["label"] = -1
df = pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True)

# 1. duplicates -------------------------------------------------------------------
key = ["id", "origin_id"]
print("[dup] exact duplicate feature rows (train):", tr.duplicated(subset=[c for c in tr.columns if c != "row_id"]).sum())
d = tr[tr.duplicated(key, keep=False)].sort_values(key)
print(f"[dup] (id, origin) pairs appearing >1 in train: {d.groupby(key).ngroups}  rows {len(d)}")
if len(d):
    g = d.groupby(key)
    print("[dup] label conflicts inside duplicated keys:", (g.label.nunique() > 1).sum())
    same = g[LAG_COLS].apply(lambda x: np.nanmax(np.abs(x.values[0] - x.values[1:])) if len(x) > 1 else 0)
    print("[dup] max |diff| of lag features within duplicate pair: median", np.nanmedian(same))
    print(d[["id", "origin_id", "month", "label", "q_bf_inc_m3_lag0", "q_ws_cum_m3_lag0", "clim_median_next"]].head(6).to_string())
print("[dup] (id, origin) duplicated in test:", te.duplicated(key).sum())

# 2. coverage -------------------------------------------------------------------
per_origin = tr.groupby("origin_id").size()
print(f"\n[panel] train origins {tr.origin_id.nunique()} (ids 0..{tr.origin_id.max()}), rows/origin median {per_origin.median():.0f}")
print(f"[panel] test origins {sorted(te.origin_id.unique())}, rows/origin {te.groupby('origin_id').size().to_dict()}")
print(f"[panel] HUC ids train {tr.id.nunique()} test {te.id.nunique()} | test ids seen in train {te.id.isin(tr.id).mean():.1%}"
      f" | test rows whose id is new {(~te.id.isin(tr.id)).sum()}")
print("[panel] origin overlap train/test:", set(tr.origin_id) & set(te.origin_id))

# 3. recover calendar order ----------------------------------------------------------
# For the same HUC, row at origin o+1 has lag{k+1} == lag{k} at origin o.
fam = "q_bf_inc_m3"
def keys(frame, lag):
    k = pd.DataFrame({"id": frame["id"], "v": frame[f"{fam}_lag{lag}"].round(1), "origin_id": frame["origin_id"]})
    return k[k.v > 0].drop_duplicates(["id", "v", "origin_id"])

cur = keys(df, 0).rename(columns={"origin_id": "earlier"})
parts = []
for k in range(1, 12):
    m = cur.merge(keys(df, k).rename(columns={"origin_id": "later"}), on=["id", "v"])
    parts.append(m[m.earlier != m.later].groupby(["earlier", "later"]).size().rename("n").reset_index().assign(gap=k))
edges = pd.concat(parts)
edges = edges[edges.n >= 200].sort_values("n", ascending=False)
nxt = edges[edges.gap == 1].drop_duplicates("earlier").drop_duplicates("later").set_index("earlier")["later"]
print(f"\n[time] confident 1-month successor links: {len(nxt)} (vote n median {edges[edges.gap == 1].n.median():.0f})")

# stitch chains
starts = sorted(set(nxt.index) - set(nxt.values))
chains = []
for s in starts:
    c = [s]
    while c[-1] in nxt.index:
        c.append(int(nxt[c[-1]]))
    chains.append(c)
chains.sort(key=len, reverse=True)
all_orig = set(df.origin_id)
print(f"[time] chains: {len(chains)}, lengths {[len(c) for c in chains][:10]}, covered {sum(map(len, chains))}/{len(all_orig)} origins")

month_of = df.groupby("origin_id").month.agg(lambda s: s.mode()[0])
ok_month = np.mean([(month_of[b] - month_of[a]) % 12 == 1 for a, b in nxt.items()])
print(f"[time] successor month == month+1 for {ok_month:.1%} of links")

order = pd.Series({o: i for c in chains for i, o in enumerate(c)}, name="t")
chain_id = pd.Series({o: j for j, c in enumerate(chains) for o in c}, name="chain")
tl = pd.concat([order, chain_id, month_of.rename("month")], axis=1).rename_axis("origin_id").reset_index()
tl["split"] = np.where(tl.origin_id.isin(te.origin_id), "test", np.where(tl.origin_id.isin(tr.origin_id), "train", "?"))
tl.to_csv(WORK / "timeline.csv", index=False)
chain = chains[0]
for o in sorted(te.origin_id.unique()):
    e = edges[(edges.earlier == o) | (edges.later == o)]
    # position relative to train chain: o is `gap` months before `later` / after `earlier`
    pos = [chain.index(r.later) - r.gap if r.earlier == o else chain.index(r.earlier) + r.gap
           for r in e.itertuples() if (r.later if r.earlier == o else r.earlier) in chain]
    print(f"[time] test origin {o} (month {month_of[o]}): links to train {len(e)}, implied chain position {sorted(set(pos))}"
          f" (train chain spans 0..{len(chain) - 1})")
# 4. figures ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
per_origin.plot.hist(bins=40, ax=ax[0], color="#4C72B0"); ax[0].set_title("Rows per origin (train)")
for j, c in enumerate(chains[:6]):
    ax[1].scatter(range(len(c)), [j] * len(c), c=["#C44E52" if o in set(te.origin_id) else "#4C72B0" for o in c], s=10)
ax[1].set_title("Recovered origin chains (red = test origin)"); ax[1].set_xlabel("position in chain (months)")
plt.tight_layout(); plt.savefig(FIG / "02_panel_timeline.png", dpi=110); plt.close()
