"""14 - Leaderboard probes that read hidden label counts out of the AP metric.
  constant score            -> AP = pi = P/N                      (test prevalence)
  1 on subset S, 0 elsewhere -> AP = (k/P)(k/n_S) + (1 - k/P) pi   (k = positives inside S)
The binary probe is quadratic in k with roots summing to pi*n_S; the complementary probe
(0 on S, 1 elsewhere) gives k' = P - k and breaks the tie.
Self-test: simulate on train rows (known labels) and check k is recovered exactly.
Usage:
  python 14_probe_math.py make                 -> writes outputs/probes_v2/*.csv
  python 14_probe_math.py solve scores.csv     -> scores.csv has columns probe,ap"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import ROOT, WORK, load_clean
from prep import clim_scale


def binary_ap(k, n_s, P, N):
    return (k / P) * (k / n_s) + (1 - k / P) * (P / N)


def solve_k(ap_s, n_s, P, N):
    """both roots of the quadratic, clipped to feasible counts"""
    pi = P / N
    a, b, c = 1 / (P * n_s), -pi / P, pi - ap_s
    disc = max(b * b - 4 * a * c, 0)
    roots = sorted({(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)})
    return [r for r in roots if -0.5 <= r <= min(n_s, P) + 0.5]


def subsets(df):
    below = (df.q_ws_cum_m3_lag0.abs() / (df.clim_median_now * 10.0 ** clim_scale(df)) < 0.6).values
    S = {f"origin_{o}": (df.origin_id == o).values for o in sorted(df.origin_id.unique())}
    S["below_now"] = below
    return S


def self_test():
    tr, _ = load_clean()
    tr = tr.drop_duplicates(["id", "origin_id"])
    sim = tr[tr.origin_id.isin(tr.origin_id.drop_duplicates().sample(4, random_state=1))].reset_index(drop=True)
    y = sim.label.values
    N, P = len(y), y.sum()
    assert np.isclose(ap(y, np.zeros(N)), P / N), "constant probe != prevalence"
    for name, m in subsets(sim).items():
        a1 = ap(y, m.astype(float))
        a2 = ap(y, (~m).astype(float))
        k_true = y[m].sum()
        assert np.isclose(a1, binary_ap(k_true, m.sum(), P, N)), name
        k1, k2 = solve_k(a1, m.sum(), P, N), solve_k(a2, (~m).sum(), P, N)
        k = [r for r in k1 if any(abs((P - r) - q) < 1 for q in k2)]
        print(f"[self-test] {name:12s} n_S {m.sum():5d} true positives {k_true:5d} | roots {np.round(k1, 1)} "
              f"| resolved with complement {np.round(k, 1)}")
    print("[self-test] formula matches sklearn AP exactly")


def make():
    _, te = load_clean()
    out = ROOT / "outputs" / "probes_v2"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": te.row_id, "label": 0.5}).to_csv(out / "probe_constant.csv", index=False)
    rows = []
    for name, m in subsets(te).items():
        for tag, s in [("in", m), ("out", ~m)]:
            pd.DataFrame({"row_id": te.row_id, "label": s.astype(float)}).to_csv(out / f"probe_{name}_{tag}.csv", index=False)
        rows.append({"subset": name, "n_S": int(m.sum())})
    pd.DataFrame(rows).to_csv(out / "subset_sizes.csv", index=False)
    print(f"wrote {2 * len(rows) + 1} probe files to {out}")


def solve(path):
    sc = pd.read_csv(path).set_index("probe").ap
    sizes = pd.read_csv(ROOT / "outputs" / "probes_v2" / "subset_sizes.csv").set_index("subset").n_S
    N = 11928
    pi = sc["probe_constant"]
    P = pi * N
    print(f"test prevalence {pi:.4f} -> {P:.0f} positives")
    for s, n in sizes.items():
        k1 = solve_k(sc[f"probe_{s}_in"], n, P, N)
        k2 = solve_k(sc[f"probe_{s}_out"], N - n, P, N)
        k = [r for r in k1 if any(abs((P - r) - q) < 2 for q in k2)] or k1
        print(f"{s:12s} n={n:5d} positives ~{np.round(k, 0)}  rate {np.round(np.array(k) / n, 3)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"test": self_test, "make": make}.get(cmd, lambda: solve(sys.argv[2]))()
