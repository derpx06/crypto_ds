# SOTA Roadmap for the Primetrade.ai Assignment

## Goal
Build a submission that is:
- technically correct,
- leakage-safe,
- statistically defensible,
- commercially relevant for trading,
- and polished enough for a hiring review.

This is not a generic ML benchmark task. The best approach is a **financial tabular forecasting pipeline** with:
- regime-aware EDA,
- leakage-safe time-series validation,
- strong gradient-boosted tree models,
- calibrated probabilities,
- threshold and sizing optimization,
- and robust reporting.

## Why This Approach
The current research consensus for medium-sized tabular data is still strongly in favor of boosted trees over most deep learning alternatives.

Primary sources:
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
- [CatBoost: gradient boosting with categorical features support](https://arxiv.org/abs/1810.11363)
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
- [scikit-learn TimeSeriesSplit docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn probability calibration docs](https://scikit-learn.org/stable/modules/calibration.html)
- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

For finance-style leakage control, the important design pattern is purged/embargoed time-series validation, as popularized in financial ML practice:
- [mlfinlab cross-validation docs](https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html)

## What We Need To Deliver
1. `notebook_1.ipynb`
2. `notebook_2.ipynb`
3. `ds_report.pdf`
4. `csv_files/` with raw and processed CSVs
5. `outputs/` with publication-quality figures

Already built in the workspace:
- exploratory notebooks,
- advanced quant notebook,
- baseline and improved modeling notebooks,
- final champion summary,
- report PDF,
- raw CSV copies,
- major figures.

## Step-by-Step Execution Plan

### Phase 1. Lock the problem definition
1. Define the core prediction target.
   - Primary classification target: `Closed PnL > 0`.
   - Secondary objective: trade selection and sizing.
2. Define the business question.
   - How does trader behavior change with fear and greed?
   - Which execution styles and accounts are robust?
   - Can we build a profitable, leakage-safe trading signal?
3. Define the success criteria.
   - Strong out-of-sample performance.
   - Clear regime insights.
   - Honest validation.
   - A report a hiring manager can trust.

### Phase 2. Build the data foundation
1. Profile both datasets completely.
   - row counts,
   - date ranges,
   - nulls,
   - duplicates,
   - rare categories,
   - tail behavior.
2. Normalize timestamps.
   - Parse `Timestamp IST`.
   - Convert IST to UTC.
   - Derive `date_utc`.
3. Join sentiment correctly.
   - Join trades to daily fear/greed by `date_utc`.
   - Verify exact alignment.
4. Remove or flag noisy records.
   - dust conversions,
   - administrative edge cases,
   - obvious artifacts.
5. Winsorize heavy tails.
   - `Closed PnL`,
   - `Size USD`,
   - `Fee` if needed.

### Phase 3. Produce research-grade EDA
1. Compare regime performance.
   - mean/median PnL,
   - win rate,
   - traded volume,
   - fee behavior,
   - maker/taker behavior.
2. Analyze account heterogeneity.
   - identify top traders,
   - measure stability,
   - examine regime specialization.
3. Analyze coin and execution effects.
   - which assets are most active,
   - which regimes attract aggressive sizing,
   - how fees and crossing behavior change.
4. Build behavioral clusters.
   - maker-dominant,
   - taker-dominant,
   - regime contrarians,
   - high-volatility accounts.
5. Generate report-ready plots.
   - regime metrics,
   - transition matrix,
   - skill-risk maps,
   - microstructure plots.

### Phase 4. Design the feature store
1. Use only information available at or before the trade time.
2. Engineer lagged sentiment features.
   - `fg_value_l1`, `fg_value_l3`, `fg_value_l7`
   - rolling sentiment mean/std
   - regime transition flags
3. Engineer account-history features.
   - rolling win rate
   - rolling PnL mean/std
   - rolling taker rate
   - streak proxies
4. Engineer coin-history features.
   - coin win rate
   - coin PnL context
5. Engineer microstructure features.
   - side,
   - crossed flag,
   - fee transforms,
   - trade size transforms,
   - time-of-day features.
6. Add interaction features.
   - sentiment x taker,
   - sentiment x account skill,
   - sentiment x coin context.

### Phase 5. Build leakage-safe validation
1. Use chronological splits only.
2. Add purge and embargo windows.
3. Fit encoders/calibrators only inside the training fold.
4. Keep a final holdout slice untouched until the very end.
5. Track fold-by-fold stability, not only one aggregate score.

### Phase 6. Train the model stack
Recommended order:
1. Logistic Regression baseline.
2. Random Forest baseline.
3. HistGradientBoosting baseline.
4. XGBoost main model.
5. LightGBM main model.
6. CatBoost main model for categorical-heavy experiments.

Why this stack:
- XGBoost is a proven high-performing tree booster.
- CatBoost is especially attractive when categorical features are important.
- Gradient-boosted trees remain the strongest practical family for this kind of tabular problem.

### Phase 7. Calibrate probabilities
1. Check reliability curves.
2. Apply `CalibratedClassifierCV` with sigmoid or isotonic calibration.
3. Measure both discrimination and calibration.
4. Use Brier score and calibration plots.

Why this matters:
- A trading system needs well-calibrated probabilities to support thresholding and sizing.

### Phase 8. Optimize the trading decision
1. Search the threshold on out-of-sample predictions.
2. Add a safety gate for low-confidence trades.
3. Test confidence-based sizing.
4. Evaluate the objective on:
   - total PnL,
   - profit factor,
   - drawdown,
   - hit rate.
5. Prefer the configuration that is economically best, not just statistically best.

### Phase 9. Add robustness tests
1. Regime-only performance.
2. Top-coin-only performance.
3. Last-window holdout.
4. Cost stress test.
5. Lower-confidence skip test.
6. Bootstrap confidence intervals for key metrics.

### Phase 10. Explain the model
1. Use SHAP for global feature importance.
2. Use regime-specific SHAP summaries.
3. Inspect failure cases.
4. Report what actually drives good predictions.

Why this matters:
- The interview is about trust as much as accuracy.
- A strong explanation section makes the work feel senior.

### Phase 11. Package the submission
1. Keep a clean notebook for EDA.
2. Keep a clean notebook for modeling.
3. Keep the report concise, specific, and honest.
4. Include figures in the `outputs/` folder.
5. Mirror raw and processed CSVs in `csv_files/`.

## Recommended Final Modeling Choice
If we had to pick one champion path for this assignment:
1. Start with the best boosted-tree baseline.
2. Add calibrated probabilities.
3. Add confidence-based sizing.
4. Add a low-confidence safety gate.
5. Validate with purged walk-forward splits.

This is better than forcing a deep neural model here because the dataset is tabular, mixed-type, and not large enough to justify a complex sequence model as the default winner.

## Important Caveat
Calling the result "SOTA" is too strong unless we have:
- a public benchmark,
- an untouched holdout,
- ablation studies,
- and stability across multiple unseen periods.

So the right language is:
- "high-performing, leakage-safe quant trading model"
- "strong empirical champion on the provided dataset"
- "production-style regime-aware pipeline"

## Current Workspace Status
Already present:
- cleaned merged data,
- regime analysis,
- advanced EDA,
- modeling notebooks,
- final champion backtest,
- PDF report,
- key figures,
- raw data copies.

Still worth doing if you want the submission even cleaner:
1. Produce a final zipped bundle.
2. Simplify notebook naming.
3. Reduce duplicate intermediate artifacts.
4. Make the report language explicitly senior and honest about validation limits.

