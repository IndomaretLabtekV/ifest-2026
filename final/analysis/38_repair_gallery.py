"""38 - Look at the repair, one corruption type at a time, and try to beat it where it is weakest.
Ground truth = median of every copy of the same (HUC, month) across the up-to-12 rows that carry it (02/04).
Part A: recovery per corruption type for q_ws_cum_m3 (the family that defines the label) + a gallery of
        random examples per type: raw, repaired, truth. Worst failures are printed to see WHY they fail.
Part B: network water-budget identity. The current repair fills a flagged ws cell with bf + qf + U where
        the upstream inflow U is INTERPOLATED along the lags. But the upstream HUCs' own rows at the same
        origin carry that inflow directly: ws(h) = bf(h) + qf(h) + sum of ws(upstream of h). The identity is
        checked per row on its clean lags and used only where it holds, so it works on the test too
        (upstream rows of the same test origin are in the test file)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG
from cvkit import tr
from prep import clean_flags, repair

FAM = "q_ws_cum_m3"
C = [f"{FAM}_lag{k}" for k in range(12)]
t = tr.t.values
long = pd.concat([pd.DataFrame({"id": tr.id.values, "m": t - k, "v": tr[c].values}) for k, c in enumerate(C)])
gt = long[long.v >= 0].groupby(["id", "m"]).v.median()
G = np.column_stack([gt.reindex(pd.MultiIndex.from_arrays([tr.id.values, t - k])).to_numpy() for k in range(12)])
raw = tr[C].to_numpy()
rep = repair(tr)
X = rep[C].to_numpy()


def rel(a, b):
    with np.errstate(all="ignore"):
        return np.abs(a - b) / np.abs(b)


ok = np.isfinite(G) & (G > 0)
bad = ok & ~(rel(raw, G) < 1e-3)
r = raw / G
kind = np.full(raw.shape, "", dtype=object)
kind[bad & np.isnan(raw)] = "nan"
kind[bad & np.isin(raw, [-9999, -9.999])] = "sentinel"
kind[bad & (raw < 0) & (rel(-raw, G) < 1e-3)] = "sign flip"
kind[bad & (raw == 0)] = "zero"
kind[bad & (r >= 5) & (r <= 200)] = "spike x10-100"
kind[bad & (kind == "")] = "other"
e = rel(X, G)
print(f"[A] {FAM}: {bad.sum()} corrupted cells of {ok.sum()} ({100 * bad.mean():.2f}%)")
tab = pd.DataFrame({"kind": kind[bad], "err": e[bad]}).groupby("kind").err.agg(
    n="size", within_5pct=lambda s: (s < .05).mean(), within_25pct=lambda s: (s < .25).mean(), median_err="median")
print(tab.round(3).to_string())
print(f"[A] clean cells damaged (>5% off): {(e[ok & ~bad] > .05).mean():.4%}")
oth = bad & (kind == "other")
if oth.any():
    print(f"[A] 'other' raw/truth ratio quantiles: {np.round(np.nanquantile(r[oth], [.05, .25, .5, .75, .95]), 3).tolist()}")

# gallery: 3 random examples per kind
kinds = [k for k in ["sign flip", "sentinel", "nan", "zero", "spike x10-100", "other"] if (kind == k).any()]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(len(kinds), 3, figsize=(15, 2.6 * len(kinds)))
for i, k in enumerate(kinds):
    rows, cols_ = np.where(kind == k)
    for j, s in enumerate(rng.choice(len(rows), 3, replace=False)):
        a, c = ax[i, j], cols_[s]
        rr = rows[s]
        a.plot(range(12), G[rr], "k-", lw=2, alpha=.35, label="truth")
        a.plot(range(12), np.where(raw[rr] > 0, raw[rr], np.nan), "o:", color="tab:red", ms=3, label="raw (>0)")
        a.plot(range(12), X[rr], "x--", color="tab:blue", label="repaired")
        a.axvline(c, color="tab:orange", alpha=.4)
        a.set_yscale("log")
        a.set_title(f"{k}: raw {raw[rr, c]:.4g} | truth {G[rr, c]:.4g} | fix {X[rr, c]:.4g}", fontsize=8)
        if i == 0 and j == 0:
            a.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "38_repair_gallery.png", dpi=110)
print(f"[fig] {FIG / '38_repair_gallery.png'}")

worst = np.argsort(-np.where(bad, np.nan_to_num(e, nan=0, posinf=1e9), -1).ravel())[:8]
print("\n[A] worst repaired cells (row, lag): raw / truth / repaired and the row's raw lags")
for w in worst:
    rr, c = divmod(w, 12)
    print(f"  ({rr}, lag{c}) {kind[rr, c]:9s} raw {raw[rr, c]:.4g} truth {G[rr, c]:.4g} fix {X[rr, c]:.4g} | "
          f"row raw {np.round(raw[rr], 1).tolist()}")

# ---------- Part B: network identity ----------
bf = rep[[f"q_bf_inc_m3_lag{k}" for k in range(12)]].to_numpy()
qf = rep[[f"q_qf_inc_m3_lag{k}" for k in range(12)]].to_numpy()
flag = np.isnan(clean_flags(tr[C].to_numpy(dtype="float64", copy=True), FAM))
key = pd.DataFrame({"to_id": tr.to_id.values, "o": tr.origin_id.values})
up = np.full(raw.shape, 0.0)
n_up = key.groupby(["to_id", "o"]).size()
has_up = pd.MultiIndex.from_arrays([tr.id.values, tr.origin_id.values]).isin(n_up.index)
for k in range(12):
    s = pd.Series(X[:, k]).groupby([key.to_id, key.o]).sum()
    up[:, k] = s.reindex(pd.MultiIndex.from_arrays([tr.id.values, tr.origin_id.values])).fillna(0).to_numpy()
ident = bf + qf + up
# does the identity hold for THIS row? check on its unflagged lags
chk = np.where(~flag, rel(ident, X), np.nan)
row_err = np.nanmedian(chk, axis=1)
for thr in (0.005, 0.01, 0.02, 0.05):
    use = (row_err < thr)[:, None] & flag & bad
    tgt = flag & bad
    new = np.where(use, ident, X)
    en = rel(new, G)
    print(f"[B] identity trusted where row median err < {thr:.3f}: applies to {use.sum() / tgt.sum():.1%} of flagged corrupt cells;"
          f" within 5% {(e[tgt] < .05).mean():.3f} -> {(en[tgt] < .05).mean():.3f}, within 25% {(e[tgt] < .25).mean():.3f}"
          f" -> {(en[tgt] < .25).mean():.3f}; on the cells it touches {(e[use] < .05).mean():.3f} -> {(en[use] < .05).mean():.3f}")
print(f"[B] rows whose upstream HUCs sit in the same origin: {has_up.mean():.3f}; headwater rows use bf + qf only")

# does it matter for the label-defining feature? anomaly error at lag0 before / after
thr_b = 0.01
use = (row_err < thr_b)[:, None] & flag
fixed = np.where(use, ident, X)
tgt0 = bad[:, 0]
d_old = np.abs(np.log((X[tgt0, 0] + 1) / (G[tgt0, 0] + 1)))
d_new = np.abs(np.log((fixed[tgt0, 0] + 1) / (G[tgt0, 0] + 1)))
print(f"[B] lag0 corrupted rows {tgt0.sum()}: median |log err| {np.median(d_old):.4f} -> {np.median(d_new):.4f}; "
      f"share crossing the 0.6 decision wrongly: {(d_old > np.log(1 / .6) / 3).mean():.3f} -> {(d_new > np.log(1 / .6) / 3).mean():.3f}")
clean_hurt = use & ~bad & ok
print(f"[B] flagged-but-actually-clean cells touched: {clean_hurt.sum()}, damaged >5%: {(rel(fixed, G)[clean_hurt] > .05).mean():.3f}")

# ---------- Part C: scale-robust network identity (prep.network_fill), on original AND test-like rescaled rows ----------
from cvkit import rescale
from prep import network_fill

sv = rescale(tr)
fac = (sv.clim_median_now / tr.clim_median_now).to_numpy()[:, None]
for tag, df in [("original scale", tr), ("rescaled like test", sv)]:
    R1 = repair(df)
    R2 = network_fill(df, R1)
    for fam in ("q_ws_cum_m3", "q_qf_inc_m3", "q_bf_inc_m3"):
        cc = [f"{fam}_lag{k}" for k in range(12)]
        lg = pd.concat([pd.DataFrame({"id": tr.id.values, "m": t - k, "v": tr[c].values}) for k, c in enumerate(cc)])
        g2 = lg[lg.v >= 0].groupby(["id", "m"]).v.median()
        Gf = np.column_stack([g2.reindex(pd.MultiIndex.from_arrays([tr.id.values, t - k])).to_numpy() for k in range(12)])
        Gf = Gf * (fac if tag != "original scale" else 1)
        rw = df[cc].to_numpy()
        okf = np.isfinite(Gf) & (Gf > 0)
        badf = okf & ~(rel(rw, Gf) < 1e-3)
        e1, e2 = rel(R1[cc].to_numpy(), Gf), rel(R2[cc].to_numpy(), Gf)
        print(f"[C] {tag:18s} {fam:12s} corrupt within 5%: {(e1[badf] < .05).mean():.3f} -> {(e2[badf] < .05).mean():.3f} | "
              f"within 25%: {(e1[badf] < .25).mean():.3f} -> {(e2[badf] < .25).mean():.3f} | "
              f"clean damaged >5%: {(e1[okf & ~badf] > .05).mean():.4f} -> {(e2[okf & ~badf] > .05).mean():.4f}", flush=True)

# ---------- Part D: the full ladder of repair versions, every family, against ground truth ----------
# v6 order bug: the spike test ran before the zero test, so a clean value whose only neighbour is an injected
# zero looked like a spike (neighbour max 0). Storm check: keep a ws 'spike' the water budget backs.
LADDER = {
    "v6 (spike then zero)": lambda d: repair(d, zero_first=False, storm=False),
    "+ zero first": lambda d: repair(d, storm=False),
    "+ storm check": lambda d: repair(d),
    "+ network budget (v7)": lambda d: network_fill(d, repair(d)),
}
print()
outs = {k: f(tr) for k, f in LADDER.items()}
rows = []
for fam in ("q_ws_cum_m3", "q_qf_inc_m3", "q_bf_inc_m3", "q_ps_wd_inc_m3"):
    cc = [f"{fam}_lag{k}" for k in range(12)]
    lg = pd.concat([pd.DataFrame({"id": tr.id.values, "m": t - k, "v": tr[c].values}) for k, c in enumerate(cc)])
    g2 = lg[lg.v >= 0].groupby(["id", "m"]).v.median()
    Gf = np.column_stack([g2.reindex(pd.MultiIndex.from_arrays([tr.id.values, t - k])).to_numpy() for k in range(12)])
    rw = tr[cc].to_numpy()
    okf = np.isfinite(Gf) & (Gf > 0)
    badf = okf & ~(rel(rw, Gf) < 1e-3)
    for k, o in outs.items():
        ee = rel(o[cc].to_numpy(), Gf)
        rows.append(dict(family=fam, version=k, corrupt_within_5=(ee[badf] < .05).mean(),
                         corrupt_within_25=(ee[badf] < .25).mean(), clean_damaged=(ee[okf & ~badf] > .05).mean(),
                         lag0_clean_damaged=(ee[:, 0][okf[:, 0] & ~badf[:, 0]] > .05).mean()))
L = pd.DataFrame(rows)
for col in ["corrupt_within_5", "clean_damaged", "lag0_clean_damaged"]:
    print(f"[D] {col}:")
    print(L.pivot(index="family", columns="version", values=col)[list(LADDER)].round(5).to_string())
