# C-MAPSS Remaining Useful Life (RUL) Prediction

Research scaffold comparing Ridge Regression, LightGBM, and LSTM for NASA C-MAPSS FD001, with split-conformal prediction intervals. It saves metrics, tuning tables, prediction tables, training history, and plots.

## Dataset setup

Download the official NASA C-MAPSS turbofan engine degradation dataset from NASA's Prognostics Center of Excellence. This project expects the matching FD001 files:

- `data/raw/train_FD001.txt`
- `data/raw/test_FD001.txt`
- `data/raw/RUL_FD001.txt`

The data are not bundled. Make sure all files come from the same FD001 subset.

## Environment and run

Python 3.10 or 3.11 is recommended, especially for TensorFlow compatibility. From the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Run unit tests with `pytest -q`.

## Methodology

1. Read and validate 26-column whitespace-separated C-MAPSS files.
2. Compute training RUL as each training engine's final observed cycle minus its current cycle; optionally cap at 125 (configurable).
3. Split training engines into disjoint train, validation, and calibration groups.
4. Tune Ridge and LightGBM using validation MAE.
5. Fit LSTM windows with preprocessing fit only on training-engine rows; use early stopping, best checkpoint, and learning-rate reduction.
6. Calibrate model-specific split-conformal intervals on the final observed cycle of each calibration engine.
7. Evaluate all three models on the final observed cycle of each official test engine using `RUL_FD001.txt`.

## Leakage and alignment safeguards

- Engine IDs do not overlap between train, validation, and calibration subsets.
- Test RUL labels are used only for final evaluation and are not assigned to every test row.
- For calibration and test evaluation, each model contributes exactly one prediction per engine: the final observed cycle.
- The calibration residual quantile is model-specific.
- LSTM imputation and scaling are fitted on training-engine rows only.

## Outputs

- `outputs/tables/data_quality.csv`
- `outputs/tables/ridge_tuning.csv`, `lightgbm_tuning.csv`
- `outputs/tables/lstm_training_history.csv`
- `outputs/tables/final_model_comparison.csv`
- `outputs/tables/best_params.json`
- `outputs/predictions/test_predictions_with_intervals.csv`
- `outputs/figures/model_comparison.png`, `actual_vs_predicted.png`, and residual plots
- `models/` fitted estimators and the best LSTM checkpoint

## Interpretation cautions

- This is a baseline research implementation, not a claim of state-of-the-art performance. Report only measured results from an actual run.
- The default RUL cap of 125 changes the target definition and must be stated in the report.
- Split-conformal coverage assumes calibration and test engines are exchangeable; distribution shift and small calibration sets can affect observed coverage.
- FD001 results alone do not establish generalization to FD002-FD004.
- Consider reporting the official asymmetric NASA scoring function in addition to MAE, RMSE, and R².

## Troubleshooting

- `FileNotFoundError`: place the three FD001 files at the configured paths.
- TensorFlow installation failure: use a Python version supported by your TensorFlow build.
- For a quick smoke test without the dataset, run `pytest -q`; it checks sequence construction and interval metrics only.
