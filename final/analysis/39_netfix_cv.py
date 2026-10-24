"""39 - Does the repair of 38 translate into AP? Same model, same folds, same rows; only the repair differs:
v6 (spike before zero, no storm check) vs v7 (zero first + budget-backed storm check + river-network budget). Judged under L5 (blocked), L6 (forward) and L7 (leave-one-year-out),
with the per-fold sign and paired 4-origin test-like draws (26: paired differences are the precise tool)."""
import sys

import numpy as np
import pandas as pd

from common import WORK
from cvkit import base_cols, run_scheme, t, test_like, views, year

B = views(sys.argv[1] if len(sys.argv) > 1 else "v6")
N = views(sys.argv[2] if len(sys.argv) > 2 else "v7")
cols = base_cols(B[0])
edges = np.linspace(t.min(), t.max() + 1, 5).astype(int)
SCH = {
    "L5 blocked": [((t < lo - 12) | (t >= hi + 12), (t >= lo) & (t < hi)) for lo, hi in zip(edges[:-1], edges[1:])],
    "L6 forward": [(t < 12 * k - 12, year == k) for k in range(8, 14)],
    "L7 LOYO": [((t < 12 * k - 12) | (t >= 12 * k + 24), year == k) for k in range(14)],
}
d_an = (B[1].anom0 - N[1].anom0).abs()
print(f"[features] valid view rows whose anom0 changed by >0.01: {(d_an > .01).mean():.4f}; "
      f"by >log(1/0.6)/3: {(d_an > np.log(1 / .6) / 3).mean():.4f}")
rows = []
for s, folds in SCH.items():
    ob, tb = run_scheme(*B, folds, cols)
    on, tn = run_scheme(*N, folds, cols)
    d = tn.ap.values - tb.ap.values
    tlb, tln = test_like(ob, n=800), test_like(on, n=800)

    rows.append(dict(scheme=s, base=tb.ap.mean(), netfix=tn.ap.mean(), gain=d.mean(), folds_up=f"{(d > 0).sum()}/{len(d)}",
                     seen_gain=(tn.ap_seen - tb.ap_seen).mean(), unseen_gain=(tn.ap_unseen - tb.ap_unseen).mean(),
                     test_like_gain=(tln - tlb).mean(), draws_up=(tln > tlb).mean()))
    print(f"[{s}] base {tb.ap.mean():.4f} -> netfix {tn.ap.mean():.4f} ({d.mean():+.4f}, up in {(d > 0).sum()}/{len(d)} folds) | "
          f"test-like {tlb.mean():.4f} -> {tln.mean():.4f} ({(tln - tlb).mean():+.4f}, {(tln > tlb).mean():.0%} draws up) | "
          f"per-fold gain {np.round(d, 4).tolist()}", flush=True)
R = pd.DataFrame(rows)
R.to_csv(WORK / f"39_{sys.argv[1:] and '_'.join(sys.argv[1:]) or 'v6_v7'}.csv", index=False)
print("\n" + R.round(4).to_string(index=False))
