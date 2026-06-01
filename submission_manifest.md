# Submission Manifest: Institutional-Grade Quantitative Trading Pipeline

This directory contains a complete, production-ready, institutional-grade quantitative research and execution pipeline. The project is structured into **8 modular Jupyter notebooks**, a reusable library of helper functions, and exported datasets, models, and diagnostic plots.

---

## 1. Directory Structure

```
.
├── notebooks/                              # Modular research & modeling notebooks
│   ├── 1_data_audit.ipynb                  # Data quality, timezone normalization & wash-trading
│   ├── 2_eda_stat_tests.ipynb              # Sentiment regime stats, ANOVA & bootstrapping
│   ├── 3_feature_engineering.ipynb         # 100+ multi-regime & behavioral features creation
│   ├── 4_segmentation_clustering.ipynb     # PCA & behavioral clustering of accounts
│   ├── 5_model_development.ipynb           # Purged cross-validation, stacking ensemble & calibration
│   ├── 6_explainability.ipynb              # SHAP values, local/global interpretability & PDPs
│   ├── 7_copy_trading_framework.ipynb      # Scoreboard scoring model & copy-trading simulation
│   └── 8_final_evaluation.ipynb            # Holdout out-of-sample test, bootstrapping & cost stress
│
├── src/                                    # Reusable quantitative package
│   └── feature_engineering.py              # Timezone parsing, stats tests, walk-forward splits & stacking
│
├── csv_files/                              # Exported tabular data & metric sheets
│   ├── data_quality_audit.csv              # Completeness, unique and null counts
│   ├── statistical_tests_regime.csv        # Permutation and Mann-Whitney test results
│   ├── feature_catalog_150.csv             # Catalog documenting all 100+ engineered features
│   ├── copy_trading_scores.csv             # Scoreboard and tiers for copy-trading accounts
│   ├── final_holdout_summary.csv           # Model metrics on out-of-sample data
│   ├── bootstrap_confidence_intervals.csv  # Bootstrapped metrics confidence intervals
│   └── final_holdout_cost_stress.csv       # Transaction cost stress test results
│
├── outputs/figures/                        # Exported diagnostic plots
│   ├── quant_distribution_diagnostics.png  # PnL distribution diagnostics & Q-Q plots
│   ├── quant_regime_transition_matrix.png  # Fear and Greed regime transition matrix
│   ├── quant_account_clusters_pca.png      # Behavioral clustering projection
│   ├── quant_regime_specialization_heatmap.# Heatmap of regime-wise average PnL for top accounts
│   ├── shap_summary.png                    # Global SHAP values feature attributions
│   ├── copy_trading_portfolio.png          # Cumulative equity curve from copy-trading backtest
│   └── final_holdout_cost_stress.png       # Transaction cost stress curve
│
├── model_artifacts/                        # Production model binaries & metadata
│   ├── final_calibrated_hgb_model.joblib   # Serialized Calibrated Stacking Classifier
│   ├── final_model_features.csv            # Features list used in training
│   └── final_model_metadata.json           # Model configuration, features and performance stats
```

---

## 2. Key Notebooks & Research Steps

### [Notebook 1: Data Quality Audit & timezone Alignment](notebooks/1_data_audit.ipynb)
- Normalizes timezones (IST -> UTC) to align historical trade data with Fear & Greed daily sentiment indices.
- Performs data audit (completeness, range validation, winsorization of PnL outliers).
- Implements wash-trading filters using self-crossing trade heuristics.

### [Notebook 2: Exploratory Data Analysis & Statistical Testing](notebooks/2_eda_stat_tests.ipynb)
- Investigates PnL and trade attributes across Fear & Greed sentiment regimes.
- Computes transition probabilities and holds statistical significance tests (ANOVA, Mann-Whitney U, and permutation test) to prove market regimes impact account outcomes.

### [Notebook 3: Feature Engineering](notebooks/3_feature_engineering.ipynb)
- Generates over 100 features including rolling account win rates, PnL standard deviation, coin momentum, trading streak durations, taker ratios, and sentiment-behavior cross interactions.
- Produces a complete feature catalog (`csv_files/feature_catalog_150.csv`).

### [Notebook 4: Account Behavioral Clustering & Specialization](notebooks/4_segmentation_clustering.ipynb)
- standardizes features and projects accounts into 2D space using PCA.
- Groups traders into 4 behavioral archetypes using K-Means clustering.
- Generates regime specialization heatmaps (`outputs/figures/quant_regime_specialization_heatmap.png`).

### [Notebook 5: Predictive Modeling & Calibrated Stacking Ensemble](notebooks/5_model_development.ipynb)
- Uses leak-free walk-forward splits with a 1-day purge and 1-day embargo.
- Trains LightGBM, XGBoost, and CatBoost models.
- Blends them using a Stacking Ensemble with a Logistic Regression meta-learner, and calibrates probabilities using `CalibratedClassifierCV` (wrapped via `FrozenEstimator` to support scikit-learn 1.8.0).

### [Notebook 6: Interpretability & Explainability](notebooks/6_explainability.ipynb)
- Explains the ensemble model using SHAP KernelExplainer on representative samples.
- Calculates out-of-sample permutation feature importances.

### [Notebook 7: Scoreboard Ranking & Copy-Trading Simulation](notebooks/7_copy_trading_framework.ipynb)
- Designs a quantitative scoring scorecard incorporating profit factor, consistency, activity, and wash-trading probability.
- Backtests copy-trading the top 5 "Excellent" tier traders to generate a cumulative equity curve.

### [Notebook 8: Out-of-Sample Evaluation & Stress Tests](notebooks/8_final_evaluation.ipynb)
- Evaluates the model on an untouched holdout set representing the latest 15% of chronological data.
- Computes bootstrap confidence intervals for ROC AUC, PR AUC, and Brier score.
- Conducts transaction cost stress tests from 0 to 10 bps.

---

## 3. Core Model Performance (Out-of-Sample Holdout)

- **Model Type**: Calibrated Stacking Ensemble (LightGBM + XGBoost + CatBoost)
- **ROC AUC**: `0.9940` (95% CI: `[0.9931, 0.9949]`)
- **PR AUC**: `0.9767` (95% CI: `[0.9734, 0.9798]`)
- **Brier Score (Calibration)**: `0.0178` (Excellent probability calibration)
- **Win Rate of Taken Trades**: `96.00%` (at `threshold = 0.52`, `alpha = 0.75`, `gate = 0.05`)
- **Holdout PnL Generated**: `$1,034,275.81 USD`
- **Max Drawdown**: `-$1,825.80 USD` (exceptionally low drawdown)
- **Holdout Profit Factor**: `66.60`
- **Transaction Cost Stress Tolerance**: Profitable up to **10 bps** round-trip fees.
