# Crypto Copy-Trading: Quantitative Research & Modeling Pipeline

Welcome! This repository contains an end-to-end quantitative research project designed to build a smart, machine-learning-driven copy-trading system. 

Copy-trading sounds simple—find a successful trader and copy their trades. In reality, it is incredibly difficult:
1. Many traders have high raw profits but take extreme risks.
2. Some traders engage in "wash trading" (self-crossing trades) to artificially inflate their volume and stats.
3. A trader might perform exceptionally well in a bull market but lose everything when the market regime shifts.

To solve this, we built a modular pipeline that cleans trade logs, detects wash trading, groups traders by behavioral archetypes (using unsupervised learning), and trains a calibrated machine learning ensemble to predict whether a proposed trade will be profitable under current market conditions.

---

## How the Pipeline Works (The 8 Notebooks)

The project is split into 8 sequential Jupyter notebooks. Each notebook represents a logical step in the quantitative research process:

1. **`1_data_audit.ipynb` (Data Cleaning & Timezones)**  
   Aligns trade timestamps from local Indian Standard Time (IST) to UTC to prevent lookahead bias. It also runs heuristic checks to flag self-crossing wash trades where a trader is trading with themselves.
   
2. **`2_eda_stat_tests.ipynb` (Market Regimes & Stats)**  
   Explores how traders perform across different daily market regimes (Fear & Greed index classification). We run ANOVA, Mann-Whitney U, and permutation significance tests to prove statistically that market sentiment changes trader performance.

3. **`3_feature_engineering.ipynb` (Feature Store)**  
   Calculates over 100 features for each trade, including rolling win rates, account-level streaks, regime-specific averages, and trade size scaling features.

4. **`4_segmentation_clustering.ipynb` (Account Profiles)**  
   Uses PCA (Principal Component Analysis) and KMeans clustering to group accounts into 4 distinct behavioral clusters (e.g., highly consistent veteran traders vs. high-risk gamblers).

5. **`5_model_development.ipynb` (Model Training & Stacking)**  
   Trains a combination of LightGBM, XGBoost, and CatBoost models. We combine them using a meta-learner (Stacking) and calibrate the output probabilities so we can use them for Kelly-criterion position sizing.

6. **`6_explainability.ipynb` (SHAP & Interpretability)**  
   Peeks inside the black-box model using SHAP values and permutation importances to explain which features drive the model's decisions.

7. **`7_copy_trading_framework.ipynb` (Account Scoreboard & Portfolio)**  
   Scores and ranks accounts based on their historical profitability and consistency, then runs a backtest simulating a copy-trading portfolio that copies the best traders.

8. **`8_final_evaluation.ipynb` (Out-of-Sample Testing & Cost Stressing)**  
   Evaluates the final calibrated model on a completely untouched holdout dataset. It also stress-tests our trading performance under transaction fees ranging from 0 to 50 basis points.

---

## Directory Structure

Here is how the repository is laid out:

```
.
├── notebooks/                              # The 8 research notebooks
├── src/                                    # Reusable Python helper code (timezone parsing, model classes)
├── data/                                   # Raw CSV data and holdout slices
├── csv_files/                              # Exported spreadsheets and summary metrics
├── outputs/figures/                        # Key diagnostic plots and charts
├── model_artifacts/                        # Serialized models (.joblib) and metadata
├── README.md                               # This document
└── .gitignore                              # Git exclusion rules
```

---

## Quick Start (How to Run)

1. **Set up a virtual environment and install packages**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the notebooks**:  
   Open Jupyter Lab/Notebook and run the notebooks in order from `1_data_audit.ipynb` to `8_final_evaluation.ipynb`.

---

## Core Findings & Results

### Out-of-Sample Performance
On the completely untouched holdout dataset, the calibrated ensemble model achieved:
* **Area Under ROC (ROC AUC)**: `0.9940` (with a 95% bootstrap confidence interval of `[0.9931, 0.9949]`), meaning the model is excellent at distinguishing profitable trades from bad ones.
* **Calibrated Win Rate**: By setting a confidence threshold of `0.52`, the trades selected by the model had an actual win rate of **`96.00%`**.
* **Profitability**: Generated **`$1,034,275.81 USD`** in holdout PnL with a maximum drawdown of only **`-$1,825.80 USD`** due to confidence-based position sizing.

---

## Key Visualizations

Here are some of the key charts generated during our research:

### 1. Market Regimes & Distributions

**Trader Performance by Sentiment Regime**  
Shows that traders perform differently depending on whether the market is in Extreme Fear or Extreme Greed.
![Sentiment Regime Core Performance Metrics](outputs/figures/quant_regime_core_metrics.png)

**Regime Transition Matrix**  
The probability of the market transitioning from one daily Fear & Greed sentiment regime to another.
![Daily Sentiment Transition Matrix](outputs/figures/quant_regime_transition_matrix.png)

**PnL Distribution Diagnostics**  
A Q-Q plot and distribution curves showing that trader returns have extremely heavy tails (extreme wins and extreme losses).
![PnL Distribution Diagnostics](outputs/figures/quant_distribution_diagnostics.png)

---

### 2. Trader Profiles & Segments

**Behavioral Clustering (PCA)**  
Visualizes the 4 distinct behavioral clusters of traders projected onto 2D space.
![Behavioral Clustering Projection](outputs/figures/quant_account_clusters_pca.png)

**Skill vs. Risk Profile Map**  
Maps traders' average PnL against their standard deviation of returns, showing who generates stable profits vs. who takes wild gambles.
![Account Skill and Risk Map](outputs/figures/quant_account_skill_risk_map.png)

**Regime Specialization Heatmap**  
Shows that top traders specialize—some excel only during high-volatility greed regimes, while others perform best in quiet fear regimes.
![Regime Specialization Heatmap](outputs/figures/quant_regime_specialization_heatmap.png)

---

### 3. Model Interpretation & Predictions

**Global Feature Importance (SHAP)**  
Highlights which engineered features (like rolling PnLs and streak counts) the model relies on most to filter trades.
![Global SHAP Attributions](outputs/figures/shap_summary.png)

**Calibration Reliability Curve**  
Demonstrates that the model's predicted probabilities closely match the actual observed win rates, which is crucial for sizing trades.
![Calibration Reliability Curve](outputs/figures/final_holdout_calibration_curve.png)

**Transaction Cost Stress Test**  
Shows how the strategy's total PnL decay as round-trip transaction costs increase from 0 to 50 bps.
![Transaction Cost Sensitivity Stress Test](outputs/figures/final_holdout_cost_stress.png)

**Copy-Trading Portfolio Backtest**  
The simulated equity curve generated by copying the top 5 ranked traders according to our scoreboard scoring model.
![Copy-Trading Portfolio Backtest](outputs/figures/copy_trading_portfolio.png)
