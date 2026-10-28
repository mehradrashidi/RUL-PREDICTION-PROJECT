# Technical Report: Remaining Useful Life Prediction using C-MAPSS FD001

## Abstract
This study compares Ridge Regression, LightGBM, and an LSTM network for estimating Remaining Useful Life (RUL) from turbofan engine sensor data. Model-specific split-conformal intervals are evaluated alongside point-prediction metrics. Insert numerical findings only after executing the experiment.

## 1. Problem statement
RUL is the estimated number of operating cycles remaining before the defined end-of-life point. The task is regression over multivariate sensor data, with temporal patterns potentially carrying degradation information.

## 2. Dataset and target
FD001 provides run-to-failure training trajectories and truncated test trajectories. For training rows, RUL is calculated from the last observed training cycle for each engine minus the current cycle. The optional cap is 125 cycles in the default configuration. The official test label file contains one target for each test engine's final observed cycle.

## 3. Experimental design and leakage control
Engine IDs are split into mutually exclusive training, validation, and calibration groups. Validation data support tabular hyperparameter selection and LSTM early stopping. Calibration engines estimate each model's absolute-residual conformal quantile. Official test engines are reserved for final evaluation.

## 4. Models
### 4.1 Ridge Regression
A regularized linear baseline. Median imputation and standardization are encapsulated in the estimator pipeline and fit using training rows.

### 4.2 LightGBM
A gradient-boosted tree model that can capture nonlinear relationships between sensor/setting features and RUL. Hyperparameters are selected using validation MAE.

### 4.3 LSTM
A recurrent neural network trained on fixed-length windows within individual engines. Short trajectories are left-padded by repeating the first observed row. Early stopping, best-checkpoint saving, and learning-rate reduction are used.

## 5. Metrics
Point predictions: MAE, RMSE, and R². Prediction intervals: empirical coverage, mean interval width, median interval width, and interval score. Each model is evaluated on the same final observed cycle per official test engine.

## 6. Results
Populate from `outputs/tables/final_model_comparison.csv` after a successful run. Do not replace missing results with estimates.

| Model | MAE | RMSE | R² | Coverage | Mean interval width | Interval score |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | — | — | — | — | — | — |
| LightGBM | — | — | — | — | — | — |
| LSTM | — | — | — | — | — | — |

## 7. Discussion
Describe the measured model ranking, residual patterns, systematic over- or under-prediction, interval widths, and observed coverage versus the nominal 90% level. Clearly separate observed findings from hypotheses.

## 8. Limitations
- FD001 is only one operating-condition subset.
- A single split may have sampling variability.
- Conformal coverage depends on calibration/test exchangeability.
- RUL capping changes the prediction target.
- LSTM results depend on window size, random seed, architecture, and compute environment.

## 9. Conclusion
Write this section only after examining measured test metrics and plots. Include the actual values, limitations, and next experiments.

## Reproducibility checklist
- [ ] Record dataset source/subset and package versions.
- [ ] Record configuration and seed.
- [ ] Verify train/validation/calibration engine IDs are disjoint.
- [ ] Confirm official test data were not used for tuning or calibration.
- [ ] Save predictions, metrics, and plots.
