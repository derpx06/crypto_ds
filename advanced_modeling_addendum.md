# Advanced Modeling Addendum

This addendum captures extra implementation ideas from the attached modeling notes and maps them to the current submission.

## Implemented Or Covered
- IST-to-UTC normalization and deterministic daily sentiment join.
- Winsorization for heavy-tailed `Closed PnL` and `Size USD`.
- Leakage-safe lagged and rolling account features.
- Lagged sentiment velocity features such as `fg_l1`, `fg_l3`, `fg_l7`, and `fg_d1`.
- Purged / embargoed walk-forward validation in the modeling notebook.
- Probability calibration with `CalibratedClassifierCV`.
- Confidence-scaled position sizing.
- Final untouched holdout.
- Permutation feature importance as the SHAP fallback because `shap` is not installed.
- Cost-stress testing on the untouched holdout.

## Useful Additions To Emphasize
- `recent_win_rate_10`: a shorter rolling account skill feature that can complement the existing `acc_wr20` and `acc_wr50`.
- `pnl_velocity`: exponentially weighted account PnL momentum.
- `realized_fee_yield`: rolling fee paid or rebate earned divided by traded USD volume.
- `date_id` grouping: useful for purged group time-series folds.
- `copy_confidence_score`: calibrated probability mapped into copy-trade sizing.

## Not Implemented Literally
- Triple-barrier labeling is not implemented because the provided dataset is fill-level and does not include a clean future price path per position needed to identify profit-take, stop-loss, and vertical-barrier events safely.
- Full graph-based wash-trading loop detection is not implemented because the dataset does not include transfer edges between wallets; the current work uses behavior-level heuristics instead.
- SHAP is not used because the local environment does not have the `shap` package installed. Permutation importance is used as a stable fallback.

## Recommended Interview Framing
The model is best described as a calibrated, leakage-safe, time-series boosted-tree pipeline with gated trade selection and a final untouched holdout. It is intentionally more defensible than a flashy model that leaks future information.
