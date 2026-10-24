# IFEST Unpad 2026 — Data Analysis Competition

Team **IndomaretLabtekV** (Riantama Putra, Athilla Zaidan Zidna Fann, Nicholas Wise Saragih Sumbayak).

| Round | Task | Metric | Public | Private | Result |
|---|---|---|---|---|---|
| [Preliminary](https://www.kaggle.com/competitions/penyisihan-ifest-2026-dac/leaderboard) | Indonesian headline–article consistency | Macro F1 | 0.95681 | 0.94346 | Rank 11, advanced to final |
| [Final](https://www.kaggle.com/competitions/babak-final-ifest-dac-2026) | Next-month water-stress risk per sub-basin | Average Precision | 0.76181 | 0.76817 | Finalist |

## Repository layout

```
.
├── preliminary/
│   ├── notebooks/        v1–v17 Kaggle notebooks, one per iteration
│   └── docs/             competition overview and rules
└── final/
    ├── notebook/         Final_DAC2026_IndomaretLabtekV.ipynb (submitted notebook)
    ├── analysis/         01–80 analysis scripts + shared modules, indexed in analysis/README.md
    ├── outputs/figs/     figures produced by the analysis scripts
    ├── docs/             competition overview
    └── Presentasi_DAC2026_IndomaretLabtekV.pdf   final presentation deck
```

Data is not committed. Download it from each Kaggle competition page into `preliminary/data/` and `final/data/`.

## Final: GALON — water-stress early warning

**GALON** (Geospatially-Aware Learning for One-month-ahead water-stress across Networks) predicts the probability
that a HUC12 sub-basin enters water stress next month, from monthly US water-budget data (supply, withdrawals,
baseflow, quickflow, river-network links, 2000–2020).

**Data.** Train: 378,780 rows × 103 columns, 2,196 sub-basins, Sep 2000–Aug 2014. Test: 2,982 sub-basins across
4 forecast origins in 2015–2020; 26% of test sub-basins never appear in train.

### Key findings

1. **The label is a physical rule.** `label = 1` iff next-month supply ≤ 0.6 × that month's climatological median
   (USGS SUI definition). Matches 99.998% of train rows; every feature is built around this ratio.
2. **Test is rescaled per sub-basin** by a random factor of 0.25–4×. Absolute-scale features collapse
   (public 0.423); scale-free anomaly features and per-split climatology recover it (0.720).
3. **The data is deliberately corrupted** in 11 ways (mixed decimal/thousands separators, month formats, `-9999`
   sentinels, flipped signs, inserted zeros, ×10–100 spikes, ×1000 unit errors, inconsistent `to_id`, 17 trap
   `aux_*` columns). Gaps are repaired with the water-budget identity
   `supply = baseflow + quickflow + upstream inflow`, verified against duplicated rows as ground truth
   (corrupt-cell recovery 61% → 84%).
4. **Test is future and unseen space**, so validation is leave-one-year-out combined with whole-basin holdout.
5. **The remaining error is next month's weather**, which is not in the data. A regional weather oracle lifts AP
   from 0.690 to 0.918, so the model sits close to the information ceiling.

### Model

- Scale-free features: current anomaly vs 12-month climatology, distance to the 0.6 threshold, drought-run length,
  transition probabilities from each sub-basin's own history, baseflow recession (clipped 0.2–1.5),
  upstream/downstream basin state, cyclical month encoding.
- Hybrid ensemble of five GBDT families (LightGBM, LightGBM with sub-basin history, LightGBM DART, XGBoost,
  CatBoost), Optuna-tuned, 3 seeds, rank-averaged; selected over three basin-holdout splits.
- Validation AP 0.708 vs 0.589 for a persistence baseline. Monitoring the top 20% riskiest sub-basins each month
  catches 60% of stress events.

### Iterations

| Version | Change | Public AP |
|---|---|---|
| V1 | Full pipeline, climatology pooled across train and test | 0.423 |
| V2 | Per-sub-basin rescaling found, scale-free features | 0.720 |
| V3–V5 | Transition-probability features, hybrid model, Optuna, 3-seed average | 0.752–0.7552 |
| V8 | Whole-basin holdout in CV, catchment-dynamics features removed (regional overfit) | 0.75548 |
| V16 | Features from the next test origin's window | 0.733 (rejected) |
| V15 / V17 / Final | Five-family ensemble chosen via three basin splits | 0.754 |

### Business insights (from the deck)

- Sub-basins already below 60% of normal have a 63% chance of stress next month vs 9% otherwise (6.6×,
  significant in 14 of 14 years).
- Risk jumps in the first month of drought (9% → 49%) and saturates after ~6 months; recovery drops from 51%
  (1 month dry) to 18% (7–12 months dry). Intervene in the first 1–2 months.
- New droughts start most often in October, February and April.
- Water withdrawal intensity contributes <1% of model attribution (SHAP); supply deficit dominates.

## Preliminary: headline–article consistency

Binary classification of Indonesian news pairs: `1` if the headline is supported by the article, `0` otherwise.
Train has 14,397 pairs (1,437 negatives); test has 3,603. Most negatives are minimal edits of a true headline
(57% differ by one token: entity, location, number, role or polarity), so the task is fine-grained factual binding
rather than topic similarity.

| Version | Approach | Public F1 |
|---|---|---|
| v1 | Fine-tuned multilingual NLI cross-encoder (mDeBERTa-v3 XNLI), 3 folds, pairwise auxiliary loss | 0.93367 |
| v2 | Global + local evidence views, counterfactual supervision, calibrated meta layer, structural graph logic | 0.94856 |
| v4 | Same-content canonical matching: within a group of titles sharing one article, keep one positive | **0.95681** |
| v5 | v4 + 4 high-confidence positive rescues | 0.95681 |
| v6–v10 | Residual verifiers (span, retrieval, conditional likelihood, slot QA, meta-stacker); none transferred | ≤ 0.95408 |
| v12A–v17 | Rebuild without pretrained weights after the organizers clarified the rules: TF-IDF/BM25 features, LinearSVM, LightGBM, from-scratch word+char interaction network | 0.78284 (v12A) |

## Running

Notebooks are written for Kaggle (T4 ×2): attach the competition dataset and **Run All**.

Final-round analysis scripts run locally from the repo root after placing the data in `final/data/`:

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost optuna scipy statsmodels matplotlib seaborn pyarrow
python final/analysis/01_data_quality.py   # writes parsed caches to final/work/, required by later scripts
python final/analysis/<NN>_<name>.py      # each script prints its result and saves a figure to final/outputs/figs/
```
