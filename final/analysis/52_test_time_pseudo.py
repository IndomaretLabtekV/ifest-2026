"""52 - Labelled examples hidden inside the test rows (test-time self-training).
label(t) = 1  <=>  ws(t+1) <= 0.6 * clim(t+1)  (99.998%). A test row at origin T carries ws(T-11..T), so for a
'pseudo origin' T-k (k >= 1) the label is ws(T-k+1) <= 0.6 * clim(month of T-k+1): computed from the row's own
lags, never from the real target T+1. Checked on train: pseudo labels agree with the real labels 99.55-99.70%.
Shifting the window back by k leaves 12-k lags; the missing oldest lags are NaN.
Value: tens of thousands of labelled rows from the TEST period (2015-2020) and the 47 NEW basins.
Honest emulation: leave-one-year-out + whole-basin holdout + rescaled valid view; in each fold ONE evaluation origin
(October or April) is the 'test origin' and only it produces pseudo rows, so no pseudo window ever contains the
target month of another evaluated origin (the real test origins are > 12 months apart)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score as ap

from common import FIG, WORK
from cvkit import HOLDOUTS, PARAMS, REPAIRS, TEST_MONTHS, base_cols, rescale, t, te, tr, views, y, year
from fe import build, clim_table_split, network
from prep import FLOWS

uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
P = {**PARAMS, "extra_trees": True}
net = network(tr, te)
sv = rescale(tr)
rep_sv = REPAIRS["v7"](sv)
Cv = clim_table_split(sv[sv.month.isin(TEST_MONTHS)], ref=clim_table_split(tr[~uns]))
ci_all = Cv.index.get_indexer(tr.id)


def pseudo_rows(idx, ks):
    """Pseudo rows (features, labels) from the valid-view rows idx, window shifted back by each k in ks."""
    feats, labs = [], []
    src = rep_sv.loc[idx]
    for k in ks:
        sh = src.copy()
        for f in FLOWS:
            for j in range(12):
                sh[f"{f}_lag{j}"] = src[f"{f}_lag{j + k}"].values if j + k < 12 else np.nan
        sh["month"] = (src.month.values - 1 - k) % 12 + 1
        sh["origin_id"] = src.origin_id.values * 100 + k
        F = build(sh, Cv, net, do_repair=False, scale_free=True)
        tm = (src.month.values - 1 - (k - 1)) % 12 + 1
        clim_t = Cv.to_numpy()[ci_all[idx], tm - 1]
        lab = (src[f"q_ws_cum_m3_lag{k - 1}"].abs().values <= 0.6 * clim_t).astype(int)
        ok = np.isfinite(clim_t)
        feats.append(F.loc[ok, cols]); labs.append(lab[ok])
    return pd.concat(feats), np.concatenate(labs)


CONFIGS = {"base": None, "pseudo k=1..3": (1, 2, 3), "pseudo k=1..6": (1, 2, 3, 4, 5, 6),
           "pseudo k=1..3 x5 weight": (1, 2, 3)}
path = WORK / "52_results.parquet"
if path.exists():
    R = pd.read_parquet(path)
else:
    rows = []
    for k in range(14):
        for m in (10, 4):
            ev = np.flatnonzero((year == k) & (tr.month.values == m))
            if len(ev) == 0:
                continue
            fit = ((t < 12 * k - 12) | (t >= 12 * k + 24)) & ~uns
            Xf = TV.loc[np.flatnonzero(fit), cols]
            Xv = VV.loc[ev, cols].to_numpy("float32")
            for name, ks in CONFIGS.items():
                if ks is None:
                    X, Y, W = Xf, y[fit], None
                else:
                    Xp, Yp = pseudo_rows(ev, ks)
                    X, Y = pd.concat([Xf, Xp]), np.concatenate([y[fit], Yp])
                    W = np.r_[np.ones(fit.sum()), np.full(len(Yp), 5.0 if "x5" in name else 1.0)]
                mdl = lgb.train(P, lgb.Dataset(X.to_numpy("float32"), Y, weight=W), 300)
                p = mdl.predict(Xv)
                u = uns[ev]
                rows.append(dict(year=k, month=m, t=int(t[ev][0]), config=name, n_pseudo=0 if ks is None else len(Yp),
                                 ap=ap(y[ev], p), ap_new=ap(y[ev][u], p[u]), ap_seen=ap(y[ev][~u], p[~u]),
                                 prev=y[ev].mean()))
                np.save(WORK / f"52_pred_{k}_{m}_{name.replace(' ', '_').replace('.', '')}.npy", p)
            print(f"  year {k} month {m:2d}: " + " | ".join(f"{r['config']} {r['ap']:.4f}" for r in rows[-4:]), flush=True)
    R = pd.DataFrame(rows)
    R.to_parquet(path)

base = R[R.config == "base"].set_index(["year", "month"])
out = []
for name in list(CONFIGS)[1:]:
    g = R[R.config == name].set_index(["year", "month"])
    d, dn, ds = g.ap - base.ap, g.ap_new - base.ap_new, g.ap_seen - base.ap_seen
    out.append(dict(config=name, n_pseudo=int(g.n_pseudo.mean()), gain=d.mean(), up=f"{(d > 0).sum()}/{len(d)}",
                    new_gain=dn.mean(), new_up=f"{(dn > 0).sum()}/{len(dn)}", seen_gain=ds.mean(),
                    seen_up=f"{(ds > 0).sum()}/{len(ds)}"))
print("\n[52] single-origin AP change vs base (LOYO, basin holdout, one evaluation origin per model):")
print(pd.DataFrame(out).round(4).to_string(index=False))
print(f"[52] base AP per origin: mean {base.ap.mean():.4f}, new basins {base.ap_new.mean():.4f}, seen {base.ap_seen.mean():.4f}")

# pooled 4-origin AP (cross-origin level matters) from the saved predictions
preds = {}
for name in CONFIGS:
    tag = name.replace(" ", "_").replace(".", "")
    preds[name] = {(k, m): np.load(WORK / f"52_pred_{k}_{m}_{tag}.npy") for k, m in base.index}
keys = list(base.index)
rng = np.random.default_rng(0)
pool = {n: [] for n in CONFIGS}
while len(pool["base"]) < 1000:
    pick = [keys[i] for i in rng.choice(len(keys), 4, replace=False)]
    tt = [int(base.loc[p, "t"]) for p in pick]
    if min(abs(a - b) for i, a in enumerate(tt) for b in tt[i + 1:]) <= 12:
        continue
    ev = np.concatenate([np.flatnonzero((year == k) & (tr.month.values == m)) for k, m in pick])
    for n in CONFIGS:
        pool[n].append(ap(y[ev], np.concatenate([preds[n][p] for p in pick])))
pool = pd.DataFrame(pool)
for n in list(CONFIGS)[1:]:
    dd = pool[n] - pool.base
    print(f"[52] pooled 4-origin AP {n:24s} {pool[n].mean():.4f} vs base {pool.base.mean():.4f}: {dd.mean():+.4f}, "
          f"improved in {(dd > 0).mean():.0%} of draws")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for name in list(CONFIGS)[1:]:
    g = R[R.config == name].set_index(["year", "month"])
    ax[0].scatter(base.ap, g.ap - base.ap, s=14, label=name)
ax[0].axhline(0, color="k", lw=.8); ax[0].legend(fontsize=8)
ax[0].set_xlabel("base AP of the origin"); ax[0].set_ylabel("AP change with pseudo rows")
ax[0].set_title("Test-time pseudo rows: per-origin change")
(pool.drop(columns="base").sub(pool.base, axis=0)).boxplot(ax=ax[1], rot=10)
ax[1].axhline(0, color="k", lw=.8); ax[1].set_title("Pooled 4-origin AP change")
plt.tight_layout()
plt.savefig(FIG / "52_test_time_pseudo.png", dpi=120)
print(f"[fig] {FIG / '52_test_time_pseudo.png'}")
