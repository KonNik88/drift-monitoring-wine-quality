# ML Monitoring Template — Wine Quality (Red)

A portfolio-ready Jupyter notebook that demonstrates **production-style monitoring** for a regression model using the UCI **Wine Quality (red)** dataset.  
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

## Files
- `quality_monitoring_notebook.ipynb` — the full monitoring notebook.
- `env/requirements.txt` — pinned deps.
- `LICENSE` — MIT.
- `.gitignore` — Python + notebooks + local env artifacts.

## Dataset
UCI Machine Learning Repository — Wine Quality (red).

## Badges / topics you can add on GitHub
`ml-monitoring`, `data-drift`, `model-drift`, `evidently`, `shap`, `psi`, `pdp`, `wine-quality`, `portfolio-project`