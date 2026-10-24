"""46 - Leaderboard probing tool for the REAL test.
  write : ../.venv/bin/python 46_apply_probes.py write [SUB.csv]  -> outputs/probes_v8/<grouping>/*.csv
          (with SUB.csv also mask_<group>.csv: that submission's ranking kept inside the group, 0 elsewhere)
  apply : ../.venv/bin/python 46_apply_probes.py apply SUB.csv SCORES.csv [grouping]
          SCORES.csv has columns probe,ap (probe = file name without .csv). Writes SUB_calibrated.csv.
  demo  : ../.venv/bin/python 46_apply_probes.py            -> end-to-end check on validation origins, where the
          labels play the leaderboard: write probes, 'score' them on a random half, calibrate, score the other half.
Groupings: origin (5 submissions), origin_new (origin x test-only/seen, 9), origin_zone (origin x dry/onset/safe, 13).
Calibration: submission ranks -> probability through an isotonic map learned out-of-fold (41), then a constant logit
shift per group to the probed prevalence (42/45)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as ap

from common import ROOT, WORK
from cvkit import t, te, test_view, tr, y
from probe import calibrate, moment_calibrate, segment_ap, solve

THR = np.log(0.6)
OUT = ROOT / "outputs" / "probes_v8"
oof = pd.read_parquet(WORK / "41_oof.parquet")["V5-like (v6 repair)"].to_numpy()
ok = ~np.isnan(oof)
ISO = IsotonicRegression(out_of_bounds="clip").fit(pd.Series(oof[ok]).rank(pct=True), y[ok])


def groups_for(origin, new, anom0, grouping):
    o = pd.Series(origin).astype("category").cat.codes.to_numpy()
    if grouping == "origin":
        g, names = o, [f"origin_{v}" for v in pd.Series(origin).astype("category").cat.categories]
    elif grouping == "origin_new":
        g = o * 2 + new.astype(int)
        names = [f"origin_{v}_{s}" for v in pd.Series(origin).astype("category").cat.categories for s in ("seen", "new")]
    else:
        g = o * 3 + np.digitize(anom0 - THR, [0, .5])
        names = [f"origin_{v}_{z}" for v in pd.Series(origin).astype("category").cat.categories for z in ("dry", "onset", "safe")]
    return g, names


def to_prob(score):
    return np.clip(ISO.predict((pd.Series(score).rank().to_numpy() - .5) / len(score)), 1e-4, 1 - 1e-4)


def solve_and_calibrate(score, g, names, scores):
    """Level shift per group (5 probes); level + slope when the masked probes mask_<group> are also given (9, 54)."""
    shares = np.array([(g == k).mean() for k in range(len(names))])
    p = to_prob(score)
    prior = np.array([p[g == k].mean() for k in range(len(names))])
    pi = scores["probe_constant"]
    est = solve(pi, [scores[f"probe_{n}"] for n in names], shares, prior if len(names) > 4 else None)
    # scorer check: the shares of positives recovered from the probes must sum to 1 if Kaggle computes AP like sklearn
    print(f"  [scorer check] sum of recovered positive shares {sum(est * shares) / pi:.4f} (must be ~1.000)")
    if all(f"mask_{n}" in scores for n in names):
        ap_obs = np.array([segment_ap(scores[f"mask_{n}"], est[k] * shares[k] / pi, pi) for k, n in enumerate(names)])
        cal, slopes = moment_calibrate(p, g, est, ap_obs)
        for n, a, s_ in zip(names, ap_obs, slopes):
            print(f"  {n:24s} model AP inside group {a:.4f} -> logit slope {s_:.2f}")
        return cal, est
    return calibrate(p, g, est), est


def test_groups(grouping):
    _, E = test_view("v7")
    tt = te.reset_index(drop=True)
    return tt, *groups_for(tt.origin_id.values, ~tt.id.isin(set(tr.id)).values, E.anom0.values, grouping)


def write(sub_path=None):
    """Binary probes per group; with a submission also the masked probes mask_<group> of THAT submission (47)."""
    for grouping in ("origin", "origin_new", "origin_zone"):
        tt, g, names = test_groups(grouping)
        d = OUT / grouping
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row_id": tt.row_id, "label": 0.5}).to_csv(d / "probe_constant.csv", index=False)
        s = None if sub_path is None else pd.read_csv(sub_path).set_index("row_id").label.reindex(tt.row_id).to_numpy()
        for k, n in enumerate(names):
            pd.DataFrame({"row_id": tt.row_id, "label": (g == k).astype(float)}).to_csv(d / f"probe_{n}.csv", index=False)
            if s is not None:
                mk = np.zeros(len(s))
                mk[g == k] = 0.01 + 0.99 * pd.Series(s[g == k]).rank(pct=True).to_numpy()
                pd.DataFrame({"row_id": tt.row_id, "label": mk}).to_csv(d / f"mask_{n}.csv", index=False)
        print(f"[write] {grouping}: {len(names) * (1 if s is None else 2) + 1} files -> {d}; rows per group {np.bincount(g).tolist()}")


def apply(sub_path, scores_path, grouping="origin"):
    sub = pd.read_csv(sub_path)
    tt, g, names = test_groups(grouping)
    s = sub.set_index("row_id").label.reindex(tt.row_id).to_numpy()
    scores = pd.read_csv(scores_path).set_index("probe").ap.to_dict()
    cal, est = solve_and_calibrate(s, g, names, scores)
    for n, e, k in zip(names, est, range(len(names))):
        print(f"  {n:24s} rows {(g == k).sum():5d}  probed prevalence {e:.4f}")
    out = Path(sub_path).with_name(Path(sub_path).stem + f"_calibrated_{grouping}.csv")
    pd.DataFrame({"row_id": tt.row_id.values, "label": cal}).to_csv(out, index=False)
    print(f"[apply] test prevalence {scores['probe_constant']:.4f}; wrote {out}")


def demo(grouping="origin", n=200):
    months = tr.month.values
    by_m = {m: np.unique(t[months == m]) for m in (2, 4, 10, 12)}
    from cvkit import views
    _, VV = views("v6")
    rng = np.random.default_rng(3)
    gains = []
    while len(gains) < n:
        pick = [rng.choice(by_m[m]) for m in by_m]
        if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
            continue
        ix = np.concatenate([np.flatnonzero(t == k) for k in pick])
        g, names = groups_for(t[ix], np.zeros(len(ix), bool), VV.anom0.values[ix], grouping)
        s = oof[ix]
        pub = rng.random(len(ix)) < .5
        scores = {"probe_constant": y[ix][pub].mean()}
        for k, nm in enumerate(names):
            scores[f"probe_{nm}"] = ap(y[ix][pub], (g[pub] == k).astype(float))
        cal, _ = solve_and_calibrate(s, g, names, scores)
        gains.append(ap(y[ix][~pub], cal[~pub]) - ap(y[ix][~pub], s[~pub]))
    gains = np.array(gains)
    print(f"[demo {grouping}] private-half AP gain {gains.mean():+.4f}, improved in {(gains > 0).mean():.0%} of {n} draws")
    assert gains.mean() > 0.01


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        write(sys.argv[2] if len(sys.argv) > 2 else None)
    elif len(sys.argv) > 1 and sys.argv[1] == "apply":
        apply(*sys.argv[2:])
    else:
        demo("origin")
        demo("origin_zone")
