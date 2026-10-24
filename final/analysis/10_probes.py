"""10 - Diagnostic leaderboard probes.
AP is computed over the whole test, so one submission cannot say *where* the model fails.
Each probe below keeps the model's ranking on one subset and replaces the rest with the persistence
baseline (a known, fixed ranker). Comparing LB scores across probes isolates subset quality:
  LB(probe_S) - LB(all_persistence)  ~ how much the model beats persistence on subset S.
Usage: python 10_probes.py <model_submission.csv>   -> writes outputs/probes/*.csv"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import ROOT, WORK

sub = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else pd.read_csv(ROOT / "outputs" / "submission.csv")
Fte = pd.read_parquet(WORK / "feat_test.parquet")
tr_ids = set(pd.read_parquet(WORK / "feat_train.parquet", columns=["id"]).id)
d = Fte[["row_id", "id", "origin_id", "anom0"]].merge(sub, on="row_id")
out = ROOT / "outputs" / "probes"
out.mkdir(parents=True, exist_ok=True)

r = lambda s: rankdata(s) / len(s)
model = r(d.label)
persist = r(-d.anom0.fillna(0))
subsets = {f"origin_{o}": d.origin_id == o for o in sorted(d.origin_id.unique())}
subsets["seen_huc"] = d.id.isin(tr_ids)
subsets["unseen_huc"] = ~d.id.isin(tr_ids)
near = (d.anom0 - np.log(0.6)).abs() < 0.3
subsets["near_threshold"] = near
subsets["far_from_threshold"] = ~near

pd.DataFrame({"row_id": d.row_id, "label": persist}).to_csv(out / "probe_all_persistence.csv", index=False)
pd.DataFrame({"row_id": d.row_id, "label": model}).to_csv(out / "probe_all_model.csv", index=False)
for n, m in subsets.items():
    pd.DataFrame({"row_id": d.row_id, "label": np.where(m, model, persist)}).to_csv(out / f"probe_{n}.csv", index=False)
    print(f"probe_{n:20s} rows {m.sum():5d}")
print("wrote", len(subsets) + 2, "probe files to", out)
