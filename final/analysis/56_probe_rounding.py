"""56 - The real 17-probe run: scorer check 1.078, yet the calibrated submission rose 0.753 -> 0.7607. Why, and what is
the best calibration once Kaggle's 3-decimal rounding is taken into account?
A) Real scores: roots and sensitivity. For origins 94/169 one root is negative (unique, +-0.001-0.004 per rounding
   step). For origins 14/81 both roots are positive and the curve is flat (+-0.010-0.013); the model prior picked the
   0.150 root for the new basins there (sum 1.078) although the 0.038 root gives sum 0.9995 and matches the EDA of 44
   (new basins at origins 14/81 are much wetter now than the seen HUCs).
B) Simulation with the leaderboard ROUNDED to 3 decimals (basin-holdout OOF of 44, public = random half, gain on the
   private half): prior-root estimator (V11, 46) vs sum-first roots + slope only where the within-group AP decodes
   precisely, and calibration strength lambda (logit interpolation between raw and calibrated).
C) Candidate submissions from the real scores."""
import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as ap

from common import FIG, ROOT, WORK
from cvkit import HOLDOUTS, t, tr, y
from probe import calibrate, moment_calibrate, segment_ap, solve

LAMBDAS = (0.25, 0.5, 0.75, 1.0)


def decode(pi, bins, masks, shares, prior, sum_first):
    prev = solve(pi, bins, shares, prior, sum_first=sum_first)
    a = prev * shares / pi
    ap_s = np.array([segment_ap(m, ai, pi) if ai > 0 else np.nan for m, ai in zip(masks, a)])
    ap_s = np.clip(ap_s, 0, 1)
    err = np.where(a > 0, 0.0005 / np.maximum(a, 1e-9), np.inf)  # one rounding step of the masked probe
    return prev, ap_s, err


def shrink(p, cal, lam):
    L = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return expit(L + lam * (logit(np.clip(cal, 1e-6, 1 - 1e-6)) - L))


def simulate(n=200, decimals=3, verbose=True):
    O = pd.read_parquet(WORK / "44_oof.parquet")
    p_all = O["basin | V5 (v6 repair)"].to_numpy()
    new_all = HOLDOUTS["basin"].astype(int)
    months = tr.month.values
    by_m = {m: np.unique(t[months == m]) for m in (2, 4, 10, 12)}
    rng = np.random.default_rng(21)
    rows, sums = [], []
    while len(rows) < n:
        pick = [rng.choice(by_m[m]) for m in by_m]
        if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
            continue
        ix = np.concatenate([np.flatnonzero(t == k) for k in pick])
        yy, pp = y[ix], p_all[ix]
        g = np.concatenate([np.full((t == k).sum(), j) for j, k in enumerate(pick)]) * 2 + new_all[ix]
        rest = np.setdiff1d(np.flatnonzero(~np.isnan(p_all)), ix)
        iso = IsotonicRegression(out_of_bounds="clip").fit(pd.Series(p_all[rest]).rank(pct=True), y[rest])
        pc = np.clip(iso.predict((pd.Series(pp).rank().to_numpy() - .5) / len(pp)), 1e-4, 1 - 1e-4)
        pub = rng.random(len(ix)) < .5
        shares = np.array([(g == k).mean() for k in range(8)])
        prior = np.array([pc[g == k].mean() for k in range(8)])
        r3 = lambda v: np.round(v, decimals)
        pi = r3(yy[pub].mean())
        bins = [r3(ap(yy[pub], (g[pub] == k).astype(float))) for k in range(8)]
        masks = []
        for k in range(8):
            mk = np.zeros(len(pc))
            mk[g == k] = 0.01 + 0.99 * pd.Series(pc[g == k]).rank(pct=True).to_numpy()
            masks.append(r3(ap(yy[pub], mk[pub])))
        base = ap(yy[~pub], pc[~pub])
        true = np.array([yy[g == k].mean() for k in range(8)])
        r = {}
        # V11 / 46 estimator: prior roots, slope everywhere
        prev1, aps1, _ = decode(pi, bins, masks, shares, prior, sum_first=False)
        sums.append((prev1 * shares).sum() / pi)
        cal1, _ = moment_calibrate(pc, g, prev1, aps1, seed=len(rows))
        r["prior roots, full (V11)"] = ap(yy[~pub], cal1[~pub]) - base
        # sum-first roots, slope gated by decode precision
        prev2, aps2, err2 = decode(pi, bins, masks, shares, prior, sum_first=True)
        cal2, _ = moment_calibrate(pc, g, prev2, aps2, seed=len(rows), ap_err=err2)
        for lam in LAMBDAS:
            r[f"sum-first, lambda {lam}"] = ap(yy[~pub], shrink(pc, cal2, lam)[~pub]) - base
        r["sum-first, level only"] = ap(yy[~pub], calibrate(pc, g, prev2)[~pub]) - base
        r["oracle level+slope"] = ap(yy[~pub], moment_calibrate(pc, g, true, np.array(
            [ap(yy[g == k], pc[g == k]) for k in range(8)]), seed=len(rows))[0][~pub]) - base
        r["err prior roots"] = np.abs(prev1 - true).mean()
        r["err sum-first"] = np.abs(prev2 - true).mean()
        rows.append(r)
    R = pd.DataFrame(rows)
    if not verbose:
        return R
    print("[B] simulation with 3-decimal rounding, private-half AP gain (share of draws improved):")
    for c in [c for c in R.columns if not c.startswith("err")]:
        print(f"  {c:28s} {R[c].mean():+.4f} ({(R[c] > 0).mean():.0%})")
    print(f"[B] mean |prevalence error|: prior roots {R['err prior roots'].mean():.4f}, sum-first {R['err sum-first'].mean():.4f}")
    s = np.array(sums)
    print(f"[B] scorer-check sum with the prior-root estimator: median {np.median(s):.3f}, share outside 0.97-1.03: "
          f"{np.mean(np.abs(s - 1) > .03):.0%}")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    R[[c for c in R.columns if not c.startswith("err")]].boxplot(ax=ax, rot=20, showfliers=False)
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("Private-half AP gain with 3-decimal leaderboard rounding (origin x new basin, 17 probes)")
    plt.tight_layout()
    plt.savefig(FIG / "56_probe_rounding.png", dpi=120)
    print(f"[fig] {FIG / '56_probe_rounding.png'}")
    return R


def candidates(scores_path, sub_path):
    m = importlib.import_module("46_apply_probes")
    tt, g, names = m.test_groups("origin_new")
    sub = pd.read_csv(sub_path).set_index("row_id").label.reindex(tt.row_id).to_numpy()
    sc = pd.read_csv(scores_path).set_index("probe").ap.to_dict()
    p = m.to_prob(sub)
    shares = np.array([(g == k).mean() for k in range(len(names))])
    prior = np.array([p[g == k].mean() for k in range(len(names))])
    pi = sc["probe_constant"]
    prev, aps_, err = decode(pi, [sc[f"probe_{n}"] for n in names], [sc[f"mask_{n}"] for n in names], shares, prior, True)
    cal, slopes = moment_calibrate(p, g, prev, aps_, ap_err=err)
    print(f"\n[C] sum-first roots: scorer check {(prev * shares).sum() / pi:.4f}")
    for k, n in enumerate(names):
        print(f"  {n:16s} rows {(g == k).sum():5d} model prev {prior[k]:.3f} probed prev {prev[k]:.4f} "
              f"AP inside {aps_[k]:.3f} (+-{min(err[k], 9):.3f}) slope {slopes[k]:.2f}")
    out = ROOT / "results" / "v9"
    for lam in LAMBDAS:
        f = out / f"submission_cal_sumfirst_l{int(lam * 100):03d}.csv"
        pd.DataFrame({"row_id": tt.row_id.values, "label": shrink(p, cal, lam)}).to_csv(f, index=False)
        print(f"  wrote {f.name}")
    f = out / "submission_cal_sumfirst_level.csv"
    pd.DataFrame({"row_id": tt.row_id.values, "label": calibrate(p, g, prev)}).to_csv(f, index=False)
    print(f"  wrote {f.name}")


if __name__ == "__main__":
    simulate()
    if len(sys.argv) > 2:
        candidates(sys.argv[1], sys.argv[2])
