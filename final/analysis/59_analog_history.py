"""59 - History for new basins from CLIMATE ANALOGS (fair: train + test inputs only).
The transition features (V3) give seen HUCs their own 14-year distribution of month-to-month supply growth; the 47
new test basins have no own history and no river neighbour in train (43). Here each held-out HUC borrows the pooled
growth history of its k most similar seen HUCs: similarity on the seasonal climatology profile (12 log values minus
their mean, 8 months known + kNN fill exactly like the test), log catchment area and climatology CV. The growth
REQUIRED to stay above 0.6 x climatology comes from the HUC's own row, and target-year growths are excluded
(|T - (t + 1)| <= 12), as for the own-history features.
  A) does an analog's growth series track the HUC's own? (correlation, vs random seen HUC)
  B) univariate AUC of the analog feature on held-out basins vs the own-history feature on seen HUCs
  C) model: leave-one-year-out + basin holdout; new basins scored by the plain model (V9 hybrid) or by the history
     model fed with analog history."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.neighbors import NearestNeighbors

from common import FIG, WORK
from cvkit import HOLDOUTS, PARAMS, REPAIRS, TEST_MONTHS, base_cols, fit_predict, rescale, t, test_like, tr, views, y, year
from fe import clim_table_split, transition_features

THR = np.log(0.6)
K = 5
uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
P = {**PARAMS, "extra_trees": True}
HIST = {**P, "learning_rate": 0.0469, "num_leaves": 187, "min_data_in_leaf": 39, "feature_fraction": 0.316,
        "bagging_fraction": 0.539, "lambda_l2": 4.21}

# similarity space from the climatology the test would have (seen: full train table, new: 8 months + fill)
Ct = clim_table_split(tr[~uns])
sv = rescale(tr)
Cv = clim_table_split(sv[sv.month.isin(TEST_MONTHS)], ref=Ct)
area = tr.groupby("id").cum_aream2.median().abs()


def profile(C, ids):
    L = np.log1p(C.loc[ids].to_numpy())
    shape = L - L.mean(1, keepdims=True)
    return np.column_stack([shape, np.log1p(area.reindex(ids).fillna(area.median()).to_numpy()) / 3,
                            L.std(1, keepdims=True)])


seen_ids = np.array(sorted(set(tr.id[~uns])))
new_ids = np.array(sorted(set(tr.id[uns])))
Xs, Xn = profile(Ct, seen_ids), profile(Cv, new_ids)
# a few HUCs have invalid climatology cells (log of <= -1): standardise NaN-aware, then put them at the centre
mu, sd = np.nanmean(Xs, 0), np.nanstd(Xs, 0) + 1e-9
z = lambda X: np.nan_to_num((X - mu) / sd)
nn = NearestNeighbors(n_neighbors=K).fit(z(Xs))
_, idx = nn.kneighbors(z(Xn))
donors = {h: seen_ids[i] for h, i in zip(new_ids, idx)}

rep = REPAIRS["v7"](tr)
g_long = pd.concat([pd.DataFrame({"id": tr.id.values, "T": t - k,
                                  "g": np.log((rep[f"q_ws_cum_m3_lag{k}"].abs().values + 1) /
                                              (rep[f"q_ws_cum_m3_lag{k + 1}"].abs().values + 1))}) for k in range(11)])
Gs = g_long.groupby(["id", "T"]).g.median().unstack()
rng = np.random.default_rng(0)
sim = []
for h in new_ids[:400]:
    a = Gs.loc[h]
    for kind, d in [("analog 1", donors[h][0]), ("random", rng.choice(seen_ids))]:
        b = Gs.loc[d]
        ok = a.notna() & b.notna()
        sim.append(dict(kind=kind, corr=np.corrcoef(a[ok], b[ok])[0, 1]))
S = pd.DataFrame(sim)
print("[A] correlation of monthly growth, new basin HUC vs donor:", S.groupby("kind")["corr"].median().round(3).to_dict())

# transition features: own history for seen HUCs, pooled analog history for held-out HUCs
path = WORK / "59_trans.parquet"
if path.exists():
    TF = pd.read_parquet(path)
else:
    hist = rep[~uns].copy()
    hist["t"] = t[~uns]
    q = rep.copy()
    q["t"] = t
    margin = VV.margin_raw_persist.to_numpy().copy()
    margin[~uns] = TV.margin_raw_persist.reindex(np.flatnonzero(~uns)).to_numpy()
    q["_clim_next"] = q.q_ws_cum_m3_lag0.abs().to_numpy() / np.exp(margin + THR)
    own = transition_features(q[~uns], hist, min_hist=5)
    parts = []
    for j in range(K):
        qd = q[uns].copy()
        qd["id"] = qd.id.map({h: d[j] for h, d in donors.items()})
        parts.append(transition_features(qd, hist, min_hist=5))
    ana = sum(p_.fillna(p_.mean()) for p_ in parts) / K
    ana["trans_n"] = sum(p_.trans_n.fillna(0) for p_ in parts)
    TF = pd.concat([own, ana]).reindex(q.index)
    TF.to_parquet(path)
for nm, m in [("seen, own history", ~uns), ("new basins, analog history", uns)]:
    ok = m & TF.trans_p_exceed.notna().values & np.isfinite(VV.margin_raw_persist.values)
    print(f"[B] univariate AUC {nm:28s} trans_p_exceed {auc(y[ok], -TF.trans_p_exceed.values[ok]):.3f} | "
          f"persistence {auc(y[ok], -VV.margin_raw_persist.values[ok]):.3f}")

TVh, VVh = pd.concat([TV, TF.loc[TV.index]], axis=1), pd.concat([VV, TF], axis=1)
hcols = cols + list(TF.columns)
rows = []
o_old, o_new = np.full(len(t), np.nan), np.full(len(t), np.nan)
for k in range(14):
    fit, va = (t < 12 * k - 12) | (t >= 12 * k + 24), year == k
    pb = fit_predict(TV, VV, fit, va, cols, params=P, uns=uns)
    ph = fit_predict(TVh, VVh, fit, va, hcols, params=HIST, rounds=200, uns=uns)
    u = uns[va]
    o_old[va], o_new[va] = np.where(u, pb, ph), ph
    rows.append(dict(new_plain=ap(y[va][u], pb[u]), new_analog=ap(y[va][u], ph[u]),
                     all_old=ap(y[va], o_old[va]), all_new=ap(y[va], o_new[va])))
    print(f"  year {k}: new basins plain {rows[-1]['new_plain']:.4f} -> analog history {rows[-1]['new_analog']:.4f}", flush=True)
R = pd.DataFrame(rows)
dn, da = R.new_analog - R.new_plain, R.all_new - R.all_old
tl_o, tl_n = test_like(o_old, n=800), test_like(o_new, n=800)
print(f"\n[C] new basins AP {R.new_plain.mean():.4f} -> {R.new_analog.mean():.4f} ({dn.mean():+.4f}, up {(dn > 0).sum()}/14) | "
      f"all rows {R.all_old.mean():.4f} -> {R.all_new.mean():.4f} ({da.mean():+.4f}, up {(da > 0).sum()}/14) | "
      f"test-like {tl_o.mean():.4f} -> {tl_n.mean():.4f} ({(tl_n - tl_o).mean():+.4f}, {(tl_n > tl_o).mean():.0%} draws up)")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
S.boxplot(column="corr", by="kind", ax=ax[0]); ax[0].set_title("Growth correlation with the analog vs random HUC")
ax[1].bar(np.arange(14) - .2, R.new_plain, .4, label="plain model")
ax[1].bar(np.arange(14) + .2, R.new_analog, .4, label="history model + analogs")
ax[1].legend(); ax[1].set_xlabel("validation year"); ax[1].set_title("AP on held-out basins")
plt.suptitle("")
plt.tight_layout()
plt.savefig(FIG / "59_analog_history.png", dpi=120)
print(f"[fig] {FIG / '59_analog_history.png'}")
