"""43 - Borrowed history for test-only HUCs.
V3-V7 score test-only HUCs (26% of test rows) with the plain model: their own history is 3 years, which 22
showed is too short for the transition features. But their river neighbours ARE in train with 14 years.
Month-to-month supply growth g = log(ws(m+1) / ws(m)) is scale-free and nested catchments share weather, so a
neighbour's history of growths into the target month can stand in, while the growth REQUIRED to stay above
0.6 x climatology is still computed from the HUC's own row.
Part A (EDA): how similar are g series of a HUC and its candidate donors (downstream, upstream, same basin)?
Part B (model): leave-one-year-out and blocked CV; the history model is fitted on seen HUCs with their own
history; held-out HUCs are scored by (1) the plain model (V7 hybrid) or (2) the history model fed with the
donor's history (new hybrid). The exclusion |T - (t + 1)| <= 12 keeps the target month out of every history.
OUTCOME: the 786 real test-only HUCs have NO donor at all (they form 47 basins disjoint from the train basins,
see 44), so this cannot be applied to the test. Part B is kept only as a documented dead end."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc

from common import FIG, WORK
from cvkit import DYN, UNS, base_cols, fit_predict, t, te, test_like, tr, views, y, year
from fe import basin_root, transition_features
from prep import network_fill, repair

THR = np.log(0.6)
TV, VV = views("v7")
cols = base_cols(TV) + DYN

# ---------------- donors ----------------
id_to = pd.concat([tr[["id", "to_id"]], te[["id", "to_id"]]]).drop_duplicates("id").set_index("id").to_id
basin = basin_root(id_to)
seen_ids = set(tr.id[~UNS])
area = tr.groupby("id").cum_aream2.median().abs()
ups = id_to.reset_index().groupby("to_id").id.apply(list)


def donor_of(h, seen_ids):
    d = id_to.get(h)
    if d in seen_ids:
        return d, "downstream"
    cand = [u for u in ups.get(h, []) if u in seen_ids]
    if cand:
        return max(cand, key=lambda u: area.get(u, 0)), "upstream"
    same = [u for u in basin.index[basin == basin.get(h)] if u in seen_ids and u != h]
    if same:
        la = np.log1p(area.get(h, 1))
        return min(same, key=lambda u: abs(np.log1p(area.get(u, 1)) - la)), "basin"
    return None, "none"


uns_ids = sorted(set(tr.id[UNS]))
D = pd.DataFrame([(h, *donor_of(h, seen_ids)) for h in uns_ids], columns=["id", "donor", "kind"]).set_index("id")
print(f"[donors] held-out HUCs {len(D)}: " + ", ".join(f"{k} {v}" for k, v in D.kind.value_counts().items()))
te_uns = sorted(set(te.id) - set(tr.id))
Dt = pd.DataFrame([(h, *donor_of(h, set(tr.id))) for h in te_uns], columns=["id", "donor", "kind"])
print(f"[donors] real test-only HUCs {len(Dt)}: " + ", ".join(f"{k} {v}" for k, v in Dt.kind.value_counts().items()))

# ---------------- Part A: growth similarity ----------------
rep = network_fill(tr, repair(tr))
g_long = pd.concat([pd.DataFrame({"id": tr.id.values, "T": t - k,
                                  "g": np.log((rep[f"q_ws_cum_m3_lag{k}"].abs().values + 1) /
                                              (rep[f"q_ws_cum_m3_lag{k + 1}"].abs().values + 1))}) for k in range(11)])
G = g_long.groupby(["id", "T"]).g.median().unstack()
sim = []
for h, r in D.iterrows():
    if r.donor is None or h not in G.index or r.donor not in G.index:
        continue
    a, b = G.loc[h], G.loc[r.donor]
    ok = a.notna() & b.notna()
    rand = G.loc[np.random.default_rng(len(sim)).choice(G.index)]
    ok2 = a.notna() & rand.notna()
    sim.append(dict(kind=r.kind, corr=np.corrcoef(a[ok], b[ok])[0, 1], corr_random=np.corrcoef(a[ok2], rand[ok2])[0, 1]))
S = pd.DataFrame(sim)
print("\n[A] correlation of monthly growth g between a held-out HUC and its donor (vs a random HUC):")
print(S.groupby("kind")[["corr", "corr_random"]].median().round(3).to_string())

# ---------------- transition features: own history for seen HUCs, donor history for held-out HUCs ----------------
path = WORK / "43_trans.parquet"
if path.exists():
    TF = pd.read_parquet(path)
else:
    hist = rep[~UNS].copy()
    hist["t"] = t[~UNS]
    q = rep.copy()
    q["t"] = t
    # clim_next consistent with the row's own scale, recovered from the scale-free margin of its view
    # (train view for seen HUCs, test-like view with the 8-month climatology for held-out HUCs)
    margin = VV.margin_raw_persist.to_numpy().copy()
    margin[~UNS] = TV.margin_raw_persist.reindex(np.flatnonzero(~UNS)).to_numpy()
    q["_clim_next"] = q.q_ws_cum_m3_lag0.abs().to_numpy() / np.exp(margin + THR)
    own = transition_features(q[~UNS], hist, min_hist=5)
    qd = q[UNS].copy()
    qd["id"] = qd.id.map(D.donor)
    qd = qd[qd.id.notna()]
    don = transition_features(qd, hist, min_hist=5)
    TF = pd.concat([own, don]).reindex(q.index)
    TF.to_parquet(path)
print(f"\n[coverage] trans_p_exceed available: seen {TF.trans_p_exceed.notna()[~UNS].mean():.3f}, "
      f"held-out via donor {TF.trans_p_exceed.notna()[UNS].mean():.3f}")
dz = (VV.anom0.values - THR) < 0
for nm, m in [("seen, own history", ~UNS), ("held-out, donor history", UNS)]:
    ok = m & TF.trans_p_exceed.notna().values & np.isfinite(VV.margin_raw_persist.values)
    print(f"  univariate AUC {nm:24s} all {auc(y[ok], -TF.trans_p_exceed.values[ok]):.3f} | "
          f"dry zone {auc(y[ok & dz], -TF.trans_p_exceed.values[ok & dz]):.3f} | "
          f"persistence baseline {auc(y[ok], -VV.margin_raw_persist.values[ok]):.3f}")

# ---------------- Part B: model ----------------
TVh = pd.concat([TV, TF.loc[TV.index]], axis=1)
VVh = pd.concat([VV, TF], axis=1)
hcols = cols + list(TF.columns)
edges = np.linspace(t.min(), t.max() + 1, 5).astype(int)
SCH = {"L5 blocked": [((t < lo - 12) | (t >= hi + 12), (t >= lo) & (t < hi)) for lo, hi in zip(edges[:-1], edges[1:])],
       "L7 LOYO": [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]}
for sname, folds in SCH.items():
    o_old, o_new = np.full(len(t), np.nan), np.full(len(t), np.nan)
    rows = []
    for fit, va in folds:
        pb = fit_predict(TV, VV, fit, va, cols)
        ph = fit_predict(TVh, VVh, fit, va, hcols)
        u = UNS[va]
        o_old[va] = np.where(u, pb, ph)
        o_new[va] = ph
        rows.append(dict(unseen_plain=ap(y[va][u], pb[u]), unseen_donor=ap(y[va][u], ph[u]),
                         all_old=ap(y[va], o_old[va]), all_new=ap(y[va], o_new[va])))
    R = pd.DataFrame(rows)
    du, da = R.unseen_donor - R.unseen_plain, R.all_new - R.all_old
    tl_o, tl_n = test_like(o_old, n=600), test_like(o_new, n=600)
    print(f"[B {sname}] held-out HUC AP plain {R.unseen_plain.mean():.4f} -> donor history {R.unseen_donor.mean():.4f} "
          f"({du.mean():+.4f}, up in {(du > 0).sum()}/{len(du)}) | all rows {R.all_old.mean():.4f} -> {R.all_new.mean():.4f} "
          f"({da.mean():+.4f}, {(da > 0).sum()}/{len(da)}) | test-like {tl_o.mean():.4f} -> {tl_n.mean():.4f} "
          f"({(tl_n - tl_o).mean():+.4f}, {(tl_n > tl_o).mean():.0%} draws up)", flush=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
S.boxplot(column=["corr", "corr_random"], by="kind", ax=ax)
ax.set_title("Monthly growth correlation: held-out HUC vs donor / random HUC")
plt.suptitle("")
plt.tight_layout()
plt.savefig(FIG / "43_donor_history.png", dpi=120)
print(f"[fig] {FIG / '43_donor_history.png'}")
