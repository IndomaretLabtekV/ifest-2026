"""15 - Enhancements on the honest V2 harness (rescaled validation, per-split climatology).
X1 extra features (baseflow anomalies, de-seasonalised anomaly trend, richer basin state)
X2 X1 + monotone constraints on the persistence family (drier now -> never lower risk)
X3 X1 + 2x weight on origin months 2, 4, 10, 12 (the test months)
X4 X1 with stronger regularisation."""
import importlib
import sys

import numpy as np

h = importlib.import_module("13_v2_cv")
FOLDS, DROP, PARAMS, tr = h.FOLDS, h.DROP, h.PARAMS, h.tr
for i in range(len(FOLDS)):
    h.fold_features(i, extra=True)
cols = lambda F: [c for c in F.columns if c not in DROP]
which = sys.argv[1:] or ["x1", "x2", "x3", "x4"]
R = {}
if "x1" in which:
    R["x1"] = h.evaluate("X1 V2 + extra features", h.lgb_preds(cols, extra=True))
if "x2" in which:
    F0 = h.fold_features(0, extra=True)
    dec = {"anom0", "anom_mean3", "anom_mean6", "margin_raw_persist", "margin_dry", "margin_anom_persist",
           "basin_anom0", "bf_anom0", "bf_anom_next_dry", "basin_margin_dry"}
    mono = [-1 if c in dec else 0 for c in cols(F0)]
    R["x2"] = h.evaluate("X2 X1 + monotone persistence", h.lgb_preds(cols, {**PARAMS, "monotone_constraints": mono,
                                                                          "monotone_constraints_method": "advanced"}, extra=True))
if "x3" in which:
    wf = lambda fit: np.where(np.isin(tr.month.values[fit], [2, 4, 10, 12]), 2.0, 1.0)
    R["x3"] = h.evaluate("X3 X1 + test-month weight 2x", h.lgb_preds(cols, extra=True, weight_fn=wf))
if "x4" in which:
    reg = {**PARAMS, "learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 400, "feature_fraction": 0.5,
           "lambda_l2": 10.0}
    R["x4"] = h.evaluate("X4 X1 + strong regularisation", h.lgb_preds(cols, reg, extra=True))
