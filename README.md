# ML Monitoring Template — Wine Quality (Red)
<!-- Badges -->
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-brightgreen)
![Evidently](https://img.shields.io/badge/Evidently-0.7.x-ff69b4)
![SHAP](https://img.shields.io/badge/SHAP-0.40%2B-informational)
![Status](https://img.shields.io/badge/status-stable-success)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A portfolio‑ready Jupyter notebook that demonstrates **production‑style monitoring** for a regression model on the UCI **Wine Quality (red)** dataset.  
It goes beyond default presets and combines **data/target/prediction drift** checks with **adversarial validation**, **PSI/JS effect sizes**, **SHAP/PDP explanations**, **slice analysis**, and a concrete **Alert Policy**.

> Verdict on sample run: **MAJOR_DRIFT** (multiple independent signals cross thresholds).

## What's inside
- **Reference vs Current split (70/30)** to emulate historical vs fresh data.
- **Model**: RandomForestRegressor trained on reference; predictions on both periods.
- **Evidently 0.7.12 (legacy)** reports:
  - Data Drift, Target Drift, Regression (Prediction), Data Quality.
- **Beyond presets**:
  - **Adversarial validation** (logistic regression) → single aggregated drift signal (AUC).
  - **Effect sizes**: **PSI**, **Jensen–Shannon**, **Hellinger** per feature.
  - **Correlation drift**: Δcorr heatmaps & top pairs.
  - **Global explanation drift**: **TreeSHAP** (mean |SHAP| comparison).
  - **PDP/ICE drift** on 1–2 key features.
  - **Slice analysis** (MAE/MAPE per quantile bins).
  - **Residuals QC**: QQ‑plot/ECDF (optional cell).
- **Alert Policy** with explicit thresholds and **runbook actions**.
- **Artifacts** saved to `reports/`: HTML (Evidently), PNG (figures), CSV/JSON summaries, `SUMMARY.md` and `final_summary.md`.

## Results at a glance (sample run)
- **PSI‑share (>0.2)**: **0.55**
- **JS‑share (>0.1)**: **0.18**
- **Adversarial AUC**: **0.736**
- **Mean |Δcorr| (top‑3 pairs)**: **0.232**
- **Slice ΔMAE% bins ≥ 20**: **8**
- **Verdict**: **MAJOR_DRIFT**

## Quick start
```bash
# 1) Create and activate environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r env/requirements.txt

# 2) Open the notebook
jupyter notebook quality_monitoring_notebook.ipynb
# or
jupyter lab quality_monitoring_notebook.ipynb
```

> The notebook will generate HTML reports and figures under `reports/`.

## Alert Policy (example)
- **PSI > 0.2** for **≥ 30%** features → drift.
- **JS > 0.1** for **≥ 30%** features → drift.
- **Adversarial AUC ≥ 0.70** → aggregated drift.
- **Mean |Δcorr| in top‑3 pairs ≥ 0.30** → structural drift.
- **ΔMAE ≥ 20%** in **≥ 2** slices → performance issue.

**Recommended actions:** re‑train on latest data (rolling window or full retrain), check sources for unstable features (e.g., `density`, `fixed acidity`), enable PSI/slice alerts, optionally apply **importance reweighting** as a quick stabilizer.

## API quirks & caveats (Evidently 0.7.12 legacy)
- **Explicit `column_mapping` is a must.** Auto‑detection may misclassify columns when new fields/renames appear.
  ```python
  from types import SimpleNamespace
  column_mapping = SimpleNamespace(
      target="quality",
      prediction="y_pred",
      numerical_features=[...],   # list of features
      categorical_features=[],
  )
  report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)
  ```
- **Legacy imports.** Use `evidently.legacy.report import Report` and `evidently.legacy.metric_preset`. API changed in ≥0.8.
- **RegressionPreset expects predictions.** Ensure `y_pred` exists in `*_pred` DataFrames (preds computed by the model).
- **Numerical stability for effect sizes.** Add a small `eps` (e.g., `1e-12`) to probabilities in **JS/Hellinger**; **PSI** uses quantile binning.
- **Adversarial validation.** `Pipeline(StandardScaler, LogisticRegression)` with stratification; AUC + feature importances are used as an aggregated drift signal.
- **SHAP on a sample.** Sample ~500 rows per period for speed; TreeExplainer for tree models.
- **Corr drift with the same color scale.** Heatmaps compared on `[-1, 1]` for honest “before/after” visuals.
- **Data Quality first.** Check for missing/constant columns and duplicates (we observed ~15% stable duplicates in both splits) to avoid mistaking DQ issues for drift.
- **Determinism.** Fix `random_state` in RF, CV splits, and SHAP sampling.
- **Paths/OS.** Notebook uses Windows paths in examples; prefer `pathlib.Path` for portability.

## Why Evidently *and* Grafana/Prometheus?
**Evidently** computes ML‑specific metrics/reports (drift, PSI/JS, explanations).  
**Prometheus/Grafana** store and visualize time‑series metrics & alerts.  
Typical production flow: compute metrics in Python → push numbers to Prometheus → Grafana dashboards; keep Evidently HTML for deep‑dive incident reviews.

## Files
- `quality_monitoring_notebook.ipynb` — the full monitoring notebook.
- `env/requirements.txt` — pinned deps.
- `LICENSE` — MIT.
- `.gitignore` — Python + notebooks + local env artifacts.

## Dataset
UCI Machine Learning Repository — Wine Quality (red).

## Badges / topics (GitHub)
`ml-monitoring`, `data-drift`, `model-drift`, `evidently`, `shap`, `psi`, `pdp`, `wine-quality`, `portfolio-project`