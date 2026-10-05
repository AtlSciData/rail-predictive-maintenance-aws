# Predictive Maintenance on Sensor Time Series (AWS)

Failure prediction on multivariate sensor time series using the public
**NASA C-MAPSS turbofan** dataset, built as a proxy for locomotive
component-failure prediction (remaining useful life, and "fails within N cycles").

> **Status: in progress.** Phase 1 (data, leakage-safe features, baselines) is in
> this repo. AWS pipeline (Glue/PySpark, SageMaker, monitoring, CDK) is being added.
> C-MAPSS is simulated aircraft-engine data, not railroad data.

## Problem framing
- **Regression:** predict RUL (capped at 125 cycles, the usual piecewise-linear convention).
- **Classification:** predict whether an engine fails within 30 cycles.
- **Evaluation:** last observed cycle of each test engine; RMSE and the asymmetric NASA
  score (late predictions cost more) for RUL; recall, ROC-AUC and PR-AUC for failure.
- **No leakage:** cross-validation is grouped by engine; normalization, regime clustering
  and sensor selection are fit on training engines only; rolling features use past cycles only.

## Approach
1. Drop constant sensors.
2. Cluster operating settings into regimes (KMeans) and z-score each sensor within its regime
   (FD002 / FD004 have six operating conditions).
3. Per-engine rolling mean, std and slope features (windows 5 and 15).
4. Models: Random Forest, LightGBM, XGBoost. Explainability with SHAP (planned).

## Roadmap
- [x] Data loading, labels, grouped CV, baselines, unit tests
- [x] FD001 baseline results (below); [ ] FD002-FD004
- [x] Failure threshold chosen on out-of-fold validation predictions to meet a target recall (`--target-recall`, default 0.9); real-data results pending
- [x] PCA / t-SNE / DBSCAN exploration (`python -m rulpm.explore`)
- [ ] SHAP analysis (`python -m rulpm.explain`) and survival-model comparison
- [x] Pandas export step and Glue PySpark feature job written (`spark/glue_features.py`); parity test vs pandas included (not yet run)
- [ ] AWS: run the Glue job on S3, SageMaker training and endpoint, CloudWatch drift monitoring, CDK
- [x] Local threshold explorer page (`web/`); GitHub Pages / AWS hosting not set up yet
- [ ] Demo video and slide deck

## Results (FD001, baseline features, default hyperparameters)
Evaluated on the 100 FD001 test engines at their last observed cycle. With only 100 engines
(about 25 failing within 30 cycles), differences between models are within noise.

**Remaining useful life (RUL, capped at 125)**

| Model | Test RMSE | NASA score | CV RMSE (grouped by engine) |
|---|---|---|---|
| LightGBM | 15.7 | 397 | 16.0 |
| XGBoost | 16.2 | 444 | 16.1 |
| Random Forest | 16.7 | 476 | 16.6 |

**Failure within 30 cycles (threshold 0.5)**

| Model | Recall | ROC-AUC | PR-AUC | CV PR-AUC |
|---|---|---|---|---|
| Random Forest | 0.72 | 0.988 | 0.967 | 0.959 |
| LightGBM | 0.72 | 0.982 | 0.951 | 0.964 |
| XGBoost | 0.68 | 0.987 | 0.962 | 0.965 |

Notes: recall at the default 0.5 threshold misses roughly a quarter of impending failures;
choosing the threshold on validation data (not the test set) to favor recall is a planned
next step. FD002-FD004 (multiple operating regimes) have not been run yet.

## Quick start
```bash
pip install -e . pytest
pytest -q                      # tests run on synthetic data, no download needed

# get the real data: NASA Prognostics Data Repository, "Turbofan Engine Degradation
# Simulation Data Set". Put train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt in data/raw/
python -m rulpm.train_baseline --subset FD001 --task rul
python -m rulpm.train_baseline --subset FD001 --task fail --target-recall 0.9
```
The failure task reports metrics at the default 0.5 threshold and at a threshold chosen on
out-of-fold validation data (highest threshold with recall >= target). The threshold is never
tuned on the test set. Caveat: validation uses all cycles of the training engines while the
test set uses each engine's last cycle, so the tuned threshold may not transfer perfectly.
On Windows `python -m` works from the repo root after `pip install -e .`.

## Threshold explorer (local viewer)
After running the failure task, build the viewer data and open the page:
```bash
python -m rulpm.train_baseline --subset FD001 --task fail   # saves results/scores/*.json
python -m rulpm.build_site                                  # writes web/data/
python -m http.server 8000 --directory web                  # open http://localhost:8000
```
The page draws ROC and precision-recall curves for each model, with a threshold slider (or drag on
a chart) that moves the operating point and updates recall, precision, flagged engines, missed
failures and false alarms. It recomputes ROC-AUC and PR-AUC from the raw scores in the browser and
compares them with Python's values; each score file has a SHA-256 hash. Only derived data (scores,
labels, metrics) is published, not the NASA files. Run the build for each subset you want to show
(re-run the failure task per subset, then `build_site` once).

## Layout
```
src/rulpm/        data.py, features.py, metrics.py, models.py, train_baseline.py, explore.py, explain.py, export.py
spark/            glue_features.py (PySpark rolling features for AWS Glue)
web/              index.html, app.js, metrics.js (static threshold explorer; data in web/data/)
scripts/          make_synthetic.py (synthetic C-MAPSS-format data for tests only)
tests/            leakage, label, feature and end-to-end tests
```

## License
MIT
