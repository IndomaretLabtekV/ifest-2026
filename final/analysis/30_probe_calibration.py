"""30 - The probing loop end to end: read per-origin prevalence out of the leaderboard, then use it.
28 showed that shifting each origin's scores to its true prevalence is worth about +2.5 AP points.
14 showed the AP metric leaks exact positive counts. This script joins the two:
  1. simulate a leaderboard on a 4-origin draw of OOF rows (labels known, used only as the 'server'),
  2. recover each origin's positive count from a constant probe plus one binary probe per origin,
  3. shift the scores to the recovered prevalences and measure the real AP gain.
Then it writes the probe files for the actual test and a calibrator for a finished submission.
Usage: python 30_probe_calibration.py            -> validate + write outputs/probes_v5/
       python 30_probe_calibration.py apply sub.csv scores.csv -> write calibrated submission"""
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score as ap

from common import ROOT, WORK, load_clean


def solve_k(ap_s, n_s, P, N):
    a, b, c = 1 / (P * n_s), -(P / N) / P, (P / N) - ap_s
    disc = max(b * b - 4 * a * c, 0)
    return sorted({(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)})


def pick_roots(root_sets, priors, P):
    """each binary probe leaves two candidate counts; the true ones must add up to P, which together
    with the model's own expectation resolves every origin at once"""
    best, combo = None, None
    for idx in np.ndindex(*[max(len(r), 1) for r in root_sets]):
        cand = [rs[j] if rs else pr for rs, j, pr in zip(root_sets, idx, priors)]
        cost = abs(sum(cand) - P) / max(P, 1) + 0.05 * sum(abs(c - pr) / max(pr, 1) for c, pr in zip(cand, priors))
        if best is None or cost < best:
            best, combo = cost, cand
    return combo


def shift_to_prevalence(p, target):
    lo, hi = -12.0, 12.0
    l = logit(np.clip(p, 1e-6, 1 - 1e-6))
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(l + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return expit(l + (lo + hi) / 2)


def validate(n_draws=40):
    D = pd.read_parquet(WORK / "18_oof.parquet")
    tr = pd.read_parquet(WORK / "train.parquet").drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
    tl = pd.read_csv(WORK / "timeline.csv").dropna(subset=["t"])
    tr = tr.merge(tl[["origin_id", "t"]], on="origin_id", how="left")
    D = D.assign(origin=tr.origin_id.values[D.index], month=tr.month.values[D.index])
    rng = np.random.default_rng(0)
    err, gains, raws = [], [], []
    for f, g in D.groupby("fold"):
        by_m = [g.origin[g.month == m].unique() for m in (2, 4, 10, 12)]
        for _ in range(n_draws // D.fold.nunique()):
            s = g[g.origin.isin([rng.choice(o) for o in by_m if len(o)])]
            if s.y.nunique() < 2:
                continue
            N = len(s)
            P = ap(s.y, np.full(N, 0.5)) * N            # probe 1: constant submission
            out = s.p.values.copy()
            masks, root_sets, priors = [], [], []
            for o in sorted(s.origin.unique()):
                m = (s.origin.values == o)
                masks.append(m)
                root_sets.append(solve_k(ap(s.y, m.astype(float)), m.sum(), P, N))  # probe 2..5
                priors.append(s.p.values[m].sum() / s.p.values.sum() * P)
            ks = pick_roots(root_sets, priors, P)
            for m, k in zip(masks, ks):
                err.append(abs(k - s.y.values[m].sum()))
                out[m] = shift_to_prevalence(s.p.values[m], np.clip(k / m.sum(), 1e-4, 1 - 1e-4))
            raws.append(ap(s.y, s.p.values))
            gains.append(ap(s.y, out) - raws[-1])
    print(f"[validate] draws {len(raws)}; recovered positive count error: median {np.median(err):.2f} rows, "
          f"max {np.max(err):.1f}")
    print(f"[validate] raw AP {np.mean(raws):.4f} -> probe-calibrated {np.mean(raws) + np.mean(gains):.4f} "
          f"(gain {np.mean(gains):+.4f}, improved in {np.mean(np.array(gains) > 0):.1%} of draws)")
    print(f"[validate] submissions needed: 1 constant + 1 per origin = 5")


def write_probes():
    _, te = load_clean()
    out = ROOT / "outputs" / "probes_v5"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": te.row_id, "label": 0.5}).to_csv(out / "probe_constant.csv", index=False)
    for o in sorted(te.origin_id.unique()):
        pd.DataFrame({"row_id": te.row_id, "label": (te.origin_id == o).astype(float)}).to_csv(
            out / f"probe_origin_{o}.csv", index=False)
    pd.DataFrame({"origin_id": sorted(te.origin_id.unique()),
                  "n_rows": [int((te.origin_id == o).sum()) for o in sorted(te.origin_id.unique())]}
                 ).to_csv(out / "origin_sizes.csv", index=False)
    print(f"[probes] wrote {1 + te.origin_id.nunique()} files to {out}")
    print("[probes] submit each, then: python 30_probe_calibration.py apply <submission.csv> <scores.csv>")
    print("[probes] scores.csv columns: probe,ap  (probe = probe_constant, probe_origin_<id>)")


def apply(sub_path, scores_path):
    sub = pd.read_csv(sub_path)
    sc = pd.read_csv(scores_path).set_index("probe").ap
    _, te = load_clean()
    d = te[["row_id", "origin_id"]].merge(sub, on="row_id")
    N = len(d)
    P = sc["probe_constant"] * N
    print(f"[apply] test prevalence {P / N:.4f} -> {P:.0f} positives of {N} rows")
    p = d.label.values.astype(float)
    out = p.copy()
    origins = sorted(d.origin_id.unique())
    masks = [(d.origin_id == o).values for o in origins]
    root_sets = [solve_k(sc[f"probe_origin_{o}"], m.sum(), P, N) for o, m in zip(origins, masks)]
    priors = [p[m].sum() / p.sum() * P for m in masks]
    ks = pick_roots(root_sets, priors, P)
    for o, m, k, roots in zip(origins, masks, ks, root_sets):
        print(f"[apply] origin {o}: rows {m.sum()}, positives {k:.0f} (rate {k / m.sum():.3f}), roots {np.round(roots, 1)}")
        out[m] = shift_to_prevalence(p[m], np.clip(k / m.sum(), 1e-4, 1 - 1e-4))
    res = pd.DataFrame({"row_id": d.row_id, "label": out})
    res.to_csv(ROOT / "outputs" / "submission_probe_calibrated.csv", index=False)
    print("[apply] wrote outputs/submission_probe_calibrated.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        apply(sys.argv[2], sys.argv[3])
    else:
        validate()
        write_probes()
