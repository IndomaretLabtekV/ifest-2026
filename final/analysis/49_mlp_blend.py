"""49 - A different model family for the blend: an MLP trained from scratch (allowed by the rules).
Trees extrapolate as steps, an MLP smoothly; on new basins (44) that difference could decorrelate errors.
Same folds, views and holdout as 48 (leave-one-year-out + whole-basin holdout); features standardised on the fit
rows, NaN -> fit median. Judged alone and as a 20-30% probability blend with the V8-like LightGBM of 48."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap
from sklearn.neural_network import MLPClassifier

from common import FIG, WORK
from cvkit import HOLDOUTS, base_cols, t, test_like, views, y, year

uns = HOLDOUTS["basin"]
TV, VV = views("v7", "basin")
cols = base_cols(TV)
folds = [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)]
base = pd.read_parquet(WORK / "48_oof.parquet")["base (V8-like)"].to_numpy()

path = WORK / "49_oof.npy"
if path.exists():
    mlp = np.load(path)
else:
    mlp = np.full(len(t), np.nan)
    for i, (fit, va) in enumerate(folds):
        f, v = np.flatnonzero(fit & ~uns), np.flatnonzero(va)
        Xf = TV.loc[f, cols].replace([np.inf, -np.inf], np.nan)
        med = Xf.median()
        Xf = Xf.fillna(med)
        mu, sd = Xf.mean(), Xf.std().replace(0, 1)
        Xv = VV.loc[v, cols].replace([np.inf, -np.inf], np.nan).fillna(med)
        m = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                          max_iter=40, early_stopping=True, n_iter_no_change=4, random_state=42)
        m.fit(((Xf - mu) / sd).clip(-6, 6).to_numpy("float32"), y[f])
        mlp[v] = m.predict_proba(((Xv - mu) / sd).clip(-6, 6).to_numpy("float32"))[:, 1]
        print(f"  fold {i}: MLP AP {ap(y[v], mlp[v]):.4f} vs LGB {ap(y[v], base[v]):.4f} ({m.n_iter_} epochs)", flush=True)
    np.save(path, mlp)

tl_b = test_like(base, n=800)
rows = []
for name, p in [("LightGBM (V8-like)", base), ("MLP", mlp), ("80% LGB + 20% MLP", .8 * base + .2 * mlp),
                ("70% LGB + 30% MLP", .7 * base + .3 * mlp)]:
    per = np.array([(ap(y[va], p[va]), ap(y[va & uns], p[va & uns]), ap(y[va & ~uns], p[va & ~uns]), ap(y[va], base[va]))
                    for _, va in folds])
    tl = test_like(p, n=800)
    rows.append(dict(model=name, AP=per[:, 0].mean(), new_basins=per[:, 1].mean(), seen=per[:, 2].mean(),
                     folds_up=f"{(per[:, 0] > per[:, 3]).sum()}/14", test_like=tl.mean(), tl_gain=(tl - tl_b).mean(),
                     draws_up=(tl > tl_b).mean()))
R = pd.DataFrame(rows).set_index("model")
ok = ~np.isnan(mlp)
print(f"\n[49] correlation LGB vs MLP predictions (Spearman): {pd.Series(base[ok]).corr(pd.Series(mlp[ok]), method='spearman'):.3f}")
print(R.round(4).to_string())
fig, ax = plt.subplots(figsize=(8, 4.5))
R[["new_basins", "seen", "test_like"]].plot.bar(ax=ax, rot=15)
ax.set_ylim(R[["new_basins", "seen", "test_like"]].values.min() - .02, R[["new_basins", "seen", "test_like"]].values.max() + .01)
ax.set_title("MLP as a blend member (LOYO, basin holdout)")
plt.tight_layout()
plt.savefig(FIG / "49_mlp_blend.png", dpi=120)
print(f"[fig] {FIG / '49_mlp_blend.png'}")
