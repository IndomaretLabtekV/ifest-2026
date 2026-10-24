"""47 - Diagnostic probes: WHERE on the real test does the model lose AP?
A 'masked' submission keeps the model's ranking inside a segment S and puts every other row at 0:
    AP_mask = a * AP_S + (1 - a) * pi          (exact; ties at 0 form one final threshold)
with a = share of all positives inside S (from the binary probe of 42/46), pi = test prevalence (constant probe)
and AP_S = the model's AP computed inside S alone. So two submissions per segment reveal the model's real test
AP in that segment: per origin, test-only basins vs seen HUCs, dry/onset/safe zone.
  demo : verify on validation draws (labels play the leaderboard, public = random half)
  write: ../.venv/bin/python 47_segment_ap_probes.py write SUB.csv   -> outputs/probes_diag/<segment>/
  read : ../.venv/bin/python 47_segment_ap_probes.py read SCORES.csv SUB.csv -> AP_S per segment
Roots of each binary probe come in pairs (a, s - a); the model's own group prevalence picks the right one (42)."""
import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

from common import FIG, ROOT, WORK
from cvkit import HOLDOUTS, t, te, test_view, tr, views, y
from probe import solve

THR = np.log(0.6)
OUT = ROOT / "outputs" / "probes_diag"


def masked(score, m):
    """Model ranking inside m mapped to (0.01, 1], everything else 0."""
    out = np.zeros(len(score))
    r = pd.Series(score[m]).rank(pct=True).to_numpy()
    out[m] = 0.01 + 0.99 * r
    return out


def segment_ap(ap_mask, a, pi):
    return (ap_mask - (1 - a) * pi) / a if a > 0 else np.nan


def partitions(origin, new, anom0):
    o = pd.Series(origin).astype("category")
    zone = np.digitize(anom0 - THR, [0, .5])
    return {
        "origin": (o.cat.codes.to_numpy(), [f"origin_{c}" for c in o.cat.categories]),
        "new": (new.astype(int), ["seen", "new"]),
        "zone": (zone, ["dry", "onset", "safe"]),
    }


def demo(n=300):
    O = pd.read_parquet(WORK / "44_oof.parquet")
    p_all = O["basin | V5 (v6 repair)"].to_numpy()
    new_all = HOLDOUTS["basin"]
    _, VV = views("v6", "basin")
    months = tr.month.values
    by_m = {m: np.unique(t[months == m]) for m in (2, 4, 10, 12)}
    rng = np.random.default_rng(5)
    rows = []
    while len(rows) < n:
        pick = [rng.choice(by_m[m]) for m in by_m]
        if min(abs(a - b) for i, a in enumerate(pick) for b in pick[i + 1:]) <= 12:
            continue
        ix = np.concatenate([np.flatnonzero(t == k) for k in pick])
        yy, pp = y[ix], p_all[ix]
        pub = rng.random(len(ix)) < .5
        pi = yy[pub].mean()
        for part, (g, names) in partitions(t[ix], new_all[ix], VV.anom0.values[ix]).items():
            shares = np.array([(g == k).mean() for k in range(len(names))])
            aps_bin = [ap(yy[pub], (g[pub] == k).astype(float)) for k in range(len(names))]
            prior = np.array([pp[g == k].mean() for k in range(len(names))])
            prev = solve(pi, aps_bin, shares, prior if part == "zone" else None)
            for k, nm in enumerate(names):
                a = prev[k] * shares[k] / pi
                est = segment_ap(ap(yy[pub], masked(pp, g == k)[pub]), a, pi)
                true_pub = ap(yy[pub & (g == k)], pp[pub & (g == k)])
                true_all = ap(yy[g == k], pp[g == k])
                rows.append(dict(part=part, seg=nm if part != "origin" else "origin", est=est, true_pub=true_pub,
                                 true_all=true_all, pooled=ap(yy, pp)))
    R = pd.DataFrame(rows)
    R["err_pub"], R["err_all"] = R.est - R.true_pub, R.est - R.true_all
    print("[demo] recovered segment AP vs truth (public rows / all rows of the segment):")
    print(R.groupby(["part", "seg"])[["err_pub", "err_all"]].agg(lambda s: np.abs(s).mean()).round(4).to_string())
    print(f"[demo] mean |error| on public rows {R.err_pub.abs().mean():.5f} (exact up to ties), on all rows "
          f"{R.err_all.abs().mean():.4f} (public half is a sample)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(R.true_all, R.est, s=4, alpha=.4)
    ax[0].plot([0, 1], [0, 1], "k:")
    ax[0].set_xlabel("true segment AP (all rows)"); ax[0].set_ylabel("recovered from 2 probes (public half)")
    ax[0].set_title("Masked-probe recovery of segment AP")
    R.boxplot(column="true_all", by=["part", "seg"], ax=ax[1], rot=45)
    ax[1].set_title("Segment AP on test-like draws (basin holdout)")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(FIG / "47_segment_ap_probes.png", dpi=120)
    print(f"[fig] {FIG / '47_segment_ap_probes.png'}")
    print("\n[demo] typical segment AP (basin holdout, all rows):")
    print(R.groupby(["part", "seg"]).true_all.describe()[["mean", "25%", "75%"]].round(3).to_string())


def test_parts():
    _, E = test_view("v7")
    tt = te.reset_index(drop=True)
    return tt, partitions(tt.origin_id.values, ~tt.id.isin(set(tr.id)).values, E.anom0.values)


def write(sub_path):
    tt, P = test_parts()
    s = pd.read_csv(sub_path).set_index("row_id").label.reindex(tt.row_id).to_numpy()
    for part, (g, names) in P.items():
        d = OUT / part
        d.mkdir(parents=True, exist_ok=True)
        for k, nm in enumerate(names):
            pd.DataFrame({"row_id": tt.row_id, "label": (g == k).astype(float)}).to_csv(d / f"bin_{nm}.csv", index=False)
            pd.DataFrame({"row_id": tt.row_id, "label": masked(s, g == k)}).to_csv(d / f"mask_{nm}.csv", index=False)
        print(f"[write] {part}: {2 * len(names)} files -> {d} (rows {np.bincount(g).tolist()})")
    pd.DataFrame({"row_id": tt.row_id, "label": 0.5}).to_csv(OUT / "probe_constant.csv", index=False)


def read(scores_path, sub_path):
    tt, P = test_parts()
    s = pd.read_csv(sub_path).set_index("row_id").label.reindex(tt.row_id).to_numpy()
    p = importlib.import_module("46_apply_probes").to_prob(s)
    sc = pd.read_csv(scores_path).set_index("probe").ap.to_dict()
    pi = sc["probe_constant"]
    print(f"[read] test prevalence {pi:.4f}")
    for part, (g, names) in P.items():
        if not all(f"bin_{nm}" in sc for nm in names):
            continue
        shares = np.array([(g == k).mean() for k in range(len(names))])
        prior = np.array([p[g == k].mean() for k in range(len(names))])
        prev = solve(pi, [sc[f"bin_{nm}"] for nm in names], shares, prior if part == "zone" else None)
        for k, nm in enumerate(names):
            a = prev[k] * shares[k] / pi
            msg = f"  {part:6s} {nm:12s} prevalence {prev[k]:.4f} share of positives {a:.3f}"
            if f"mask_{nm}" in sc:
                msg += f" | model AP inside segment {segment_ap(sc[f'mask_{nm}'], a, pi):.4f}"
            print(msg)


if __name__ == "__main__":
    {"write": lambda: write(sys.argv[2]), "read": lambda: read(sys.argv[2], sys.argv[3])}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", demo)()
