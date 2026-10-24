"""37 - Label-free per-origin prevalence calibration (no leaderboard probing).
28 showed that shifting each origin's scores to its TRUE prevalence is worth +2.5 AP on 4-origin draws.
Can part of it be earned from the data alone? Estimators of an origin's prevalence:
  mean-p     the row model's mean probability (what the raw scores already imply)
  EM         Saerens et al. (2002) prior-shift EM on the row probabilities
  origin-reg ridge regression of logit(prevalence) on label-free origin aggregates (share below the threshold,
             mean anomalies, trends, season), fitted on training-period origins only
  blend      average of mean-p and origin-reg in logit space
Each estimate drives a constant logit shift per origin. To keep the test honest ONE model scores all the
validation origins (like the real test): fit on Sep 2000 - Aug 2007, gap 12, validate Sep 2008 - Aug 2014.
The origin regressor is trained on origins before the gap using leave-one-year-out predictions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score as ap

from common import FIG
from cvkit import base_cols, fit_predict, run_scheme, t, tr, views, y, year

THR = np.log(0.6)
TV, VV = views()
cols = base_cols(TV)
cut = 84
va = t >= cut + 12
p_va = fit_predict(TV, VV, t < cut, va, cols)
p = np.full(len(t), np.nan)
p[va] = p_va
# leave-one-year-out predictions on the early origins -> training data for the origin regressor
early = [((t < 12 * k - 12) | ((t >= 12 * k + 24) & (t < cut)), year == k) for k in range(cut // 12)]
p_early, _ = run_scheme(TV, VV, early, cols)
P = np.where(va, p, p_early)
pi_train = y[t < cut].mean()


def origin_table():
    V = VV[["anom0", "anom_mean3", "anom_mean12", "ws_trend1", "margin_dry", "qf_share0", "m_sin", "m_cos"]].copy()
    V["below"] = (VV.anom0 < THR).astype(float)
    V["t"], V["y"], V["p"] = t, y, P
    g = V.groupby("t")
    O = g.mean()
    O["lp"] = logit(O.p.clip(1e-4, 1 - 1e-4))
    O["ly"] = logit(O.y.clip(1e-3, 1 - 1e-3))
    # origins inside the 12-month gap before the validation period have no honest prediction
    return O[np.isfinite(O.p)]


O = origin_table()
feat = ["anom0", "anom_mean3", "anom_mean12", "ws_trend1", "margin_dry", "qf_share0", "m_sin", "m_cos", "below"]
fit_o = O.index < cut
reg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(O.loc[fit_o, feat], O.loc[fit_o, "ly"])
reg_mp = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(O.loc[fit_o, feat + ["lp"]], O.loc[fit_o, "ly"])
O["reg"] = expit(reg.predict(O[feat]))
O["reg+meanp"] = expit(reg_mp.predict(O[feat + ["lp"]]))


def em(pp, pi0, it=100):
    pi = pi0
    for _ in range(it):
        a = pi / pi0 * pp
        q = a / (a + (1 - pi) / (1 - pi0) * (1 - pp))
        pi = q.mean()
    return pi


vo = O.index[O.index >= cut + 12]
est = pd.DataFrame(index=vo)
est["true"] = O.loc[vo, "y"]
est["mean-p"] = O.loc[vo, "p"]
est["EM"] = [em(p[t == o], pi_train) for o in vo]
est["origin-reg"] = O.loc[vo, "reg"]
est["reg+meanp"] = O.loc[vo, "reg+meanp"]
est["blend"] = expit((logit(est["mean-p"]) + logit(est["origin-reg"])) / 2)
print("[estimators] error of the origin prevalence estimate on the validation origins (one model):")
for c in est.columns[1:]:
    e = est[c] - est.true
    print(f"  {c:11s} MAE {e.abs().mean():.4f}  bias {e.mean():+.4f}  corr {np.corrcoef(est[c], est.true)[0, 1]:.3f}")


def shifted(sub, target):
    lg = logit(np.clip(sub, 1e-6, 1 - 1e-6))
    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expit(lg + mid).mean() < target else (lo, mid)
    return expit(lg + (lo + hi) / 2)


cal = {"raw": p.copy()}
for c in ["EM", "origin-reg", "reg+meanp", "blend", "true"]:
    q = p.copy()
    for o in vo:
        m = t == o
        q[m] = shifted(p[m], est.loc[o, c])
    cal["oracle" if c == "true" else c] = q

rng = np.random.default_rng(0)
by_m = {m: [o for o in vo if tr.month.values[t == o][0] == m] for m in (2, 4, 10, 12)}
S = {k: [] for k in cal}
while len(S["raw"]) < 1500:
    pick = [rng.choice(by_m[m]) for m in by_m]
    if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
        continue
    m = np.isin(t, pick)
    for k in cal:
        S[k].append(ap(y[m], cal[k][m]))
S = pd.DataFrame(S)
print(f"\n[single model] validation AP (all {len(vo)} origins pooled): raw {ap(y[va], p[va]):.4f}")
print("[4-origin draws] mean AP and paired gain vs raw:")
for k in S:
    d = S[k] - S.raw
    print(f"  {k:11s} AP {S[k].mean():.4f}  gain {d.mean():+.4f}  improved in {(d > 0).mean():.0%} of draws")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for c, mk in [("mean-p", "o"), ("origin-reg", "s"), ("EM", "^")]:
    ax[0].scatter(est.true, est[c], marker=mk, label=c, alpha=.7)
lim = [0, est.values.max() + .05]
ax[0].plot(lim, lim, "k:")
ax[0].set_xlabel("true origin prevalence"); ax[0].set_ylabel("estimate")
ax[0].legend(); ax[0].set_title("Per-origin prevalence estimates (validation origins)")
(S.drop(columns="raw").sub(S.raw, axis=0)).boxplot(ax=ax[1])
ax[1].axhline(0, color="k", lw=.8)
ax[1].set_title("Paired AP gain vs raw on 4-origin draws")
plt.tight_layout()
plt.savefig(FIG / "37_origin_prevalence.png", dpi=120)
print(f"[fig] {FIG / '37_origin_prevalence.png'}")
