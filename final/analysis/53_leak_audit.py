"""53 - Last audit for hidden structure (is there any information left that the model does not use?)
  A) aux_twin_unit_*: they copy a lag column / 1000 with a different family mapping per split (01). Which exact
     (family, lag) does each copy, and could any of them copy the FUTURE value ws(t+1) (a planted leak)?
  B) row_id: does the row order carry the label, the time or the origin (train), or anything in the test?
  C) origin_id / id: any relation to time or label beyond what the timeline reconstruction found?"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as auc

from common import FIG, FLOWS, ROOT, load_raw, parse_num
from cvkit import t, te, tr, y

# ---------------- A: aux twins ----------------
raw_tr, raw_te = load_raw()
raw_tr = raw_tr.drop_duplicates(["id", "origin_id"]).reset_index(drop=True)
assert (raw_tr.id.values == tr.id.values).all() and (raw_te.id.values == te.id.values).all()
twins = [c for c in raw_tr.columns if c.startswith("aux_twin")]
nxt = tr[["id"]].assign(t=t + 1).merge(tr[["id"]].assign(t=t, ws_next=tr.q_ws_cum_m3_lag0.abs().values),
                                        on=["id", "t"], how="left").ws_next.to_numpy()
for split, R, D in [("train", raw_tr, tr), ("test", raw_te, te)]:
    for c in twins:
        v = parse_num(R[c]).to_numpy() * 1000
        best = []
        for f in FLOWS:
            for k in range(12):
                x = D[f"{f}_lag{k}"].abs().to_numpy()
                ok = np.isfinite(v) & np.isfinite(x) & (x > 0) & (v > 0)
                if ok.sum() > 1000:
                    best.append((np.mean(np.abs(np.log(v[ok] / x[ok])) < 0.01), f, k))
        if split == "train":
            ok = np.isfinite(v) & np.isfinite(nxt) & (v > 0) & (nxt > 0)
            best.append((np.mean(np.abs(np.log(v[ok] / nxt[ok])) < 0.01), "FUTURE ws(t+1)", -1))
        best.sort(reverse=True)
        print(f"[A] {split:5s} {c}: best match {best[0][1]} lag{best[0][2]} ({best[0][0]:.3f} of cells within 1%), "
              f"runner-up {best[1][1]} lag{best[1][2]} ({best[1][0]:.3f})")

# ---------------- B: row_id ----------------
rid = raw_tr.row_id.astype(int).to_numpy()
print(f"\n[B] train: spearman(row_id, t) {spearmanr(rid, t).correlation:+.3f}, AUC(row_id -> label) {auc(y, rid):.3f}")
within = pd.DataFrame({"o": tr.origin_id, "r": rid, "y": y}).groupby("o").apply(
    lambda g: auc(g.y, g.r) if g.y.nunique() > 1 else np.nan)
print(f"[B] train: within-origin AUC(row_id -> label) median {within.median():.3f} (p5 {within.quantile(.05):.3f}, "
      f"p95 {within.quantile(.95):.3f})")
trr = te.reset_index(drop=True)
blocks = (trr.origin_id != trr.origin_id.shift()).sum()
print(f"[B] test: row_id is {'contiguous' if (trr.row_id.diff().dropna() == 1).all() else 'not contiguous'}; "
      f"origin changes along row order {blocks} times (4 = sorted by origin)")
sub = pd.read_csv(ROOT / "results" / "v9" / "submission.csv").set_index("row_id").label
print(f"[B] test: spearman(row_id, V9 score) {spearmanr(trr.row_id, sub.reindex(trr.row_id)).correlation:+.3f}; "
      f"within origin median {trr.assign(s=sub.reindex(trr.row_id).values).groupby('origin_id').apply(lambda g: spearmanr(g.row_id, g.s).correlation).median():+.3f}")

# ---------------- C: origin_id and id ----------------
o_t = tr.groupby("origin_id").t.first()
print(f"\n[C] spearman(origin_id, t) {spearmanr(o_t.index, o_t.values).correlation:+.3f}; "
      f"origin prevalence vs origin_id spearman {spearmanr(o_t.index, pd.Series(y).groupby(tr.origin_id.values).mean().reindex(o_t.index)).correlation:+.3f}")
h = tr.id.map(lambda s: int(s[:8], 16))
print(f"[C] hashed id prefix vs HUC positive rate: spearman {spearmanr(h.groupby(tr.id).first(), pd.Series(y).groupby(tr.id.values).mean()).correlation:+.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(rid[::50], t[::50], s=1)
ax[0].set_xlabel("train row_id"); ax[0].set_ylabel("chain month t"); ax[0].set_title("Row order vs time (train)")
ax[1].plot(trr.row_id, trr.origin_id, lw=.5)
ax[1].set_xlabel("test row_id"); ax[1].set_ylabel("origin_id"); ax[1].set_title("Row order vs origin (test)")
plt.tight_layout()
plt.savefig(FIG / "53_leak_audit.png", dpi=120)
print(f"[fig] {FIG / '53_leak_audit.png'}")
