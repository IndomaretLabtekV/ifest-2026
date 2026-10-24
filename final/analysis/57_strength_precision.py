"""57 - After the real run (level 0.757, lambda 0.25 0.758, lambda 0.5/0.75/1.0 all 0.761 on a 3-decimal readout):
  A) paired per-draw comparison of lambda 0.5 / 0.75 / 1.0 on the private half: how often does each win, and by how
     much? Differences below 0.0005 are invisible on a 3-decimal leaderboard readout.
  B) value of reading the probe scores with 5 decimals instead of 3 (Kaggle shows 5, e.g. 0.75548, 0.76074):
     same simulation, only the rounding of the 17 probe answers changes."""
import importlib

import numpy as np

sim = importlib.import_module("56_probe_rounding").simulate
R3 = sim(n=200, decimals=3, verbose=False)
R5 = sim(n=200, decimals=5, verbose=False)
print("[A] private-half gain, probe answers rounded to 3 decimals (as recorded):")
for c in ["sum-first, lambda 0.5", "sum-first, lambda 0.75", "sum-first, lambda 1.0"]:
    print(f"  {c:24s} {R3[c].mean():+.4f}")
for a, b in [("0.5", "1.0"), ("0.75", "1.0"), ("0.5", "0.75")]:
    d = R3[f"sum-first, lambda {b}"] - R3[f"sum-first, lambda {a}"]
    print(f"  lambda {b} minus {a}: mean {d.mean():+.4f}, std {d.std():.4f}, {b} wins {(d > 0).mean():.0%}, "
          f"|diff| < 0.0005 (invisible at 3 decimals) in {(d.abs() < .0005).mean():.0%} of draws, "
          f"worst case for {b} {d.min():+.4f}")
print("\n[B] 3 vs 5 decimals of the probe answers, private-half gain:")
for c in ["sum-first, level only", "sum-first, lambda 0.5", "sum-first, lambda 1.0", "oracle level+slope"]:
    print(f"  {c:24s} 3 dec {R3[c].mean():+.4f} | 5 dec {R5[c].mean():+.4f}")
print(f"  mean |prevalence error| 3 dec {R3['err sum-first'].mean():.4f} | 5 dec {R5['err sum-first'].mean():.4f}")
