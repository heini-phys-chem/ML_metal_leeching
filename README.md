# Machine Learning Models for Leaching Efficiency Prediction

This directory contains a comprehensive suite of machine learning models for predicting leaching efficiency from experimental data.

## Overview

The project includes **4 model types**:
1. **KRR** - Kernel Ridge Regression with custom CuPy Laplacian kernel
2. **RF** - Random Forest Regressor
3. **XGBoost** - XGBoost Regressor
4. **Ensemble-OOF** - Stacking ensemble combining KRR and XGBoost with linear meta-model using Out-of-Fold predictions

## Data Files

Three Excel files are used as input:
- `MS_samples_Al.xlsx` - Training dataset
- `MS_samples_Al_new.xlsx` - New test dataset for transfer learning
- `all_data.xlsx` - Combined dataset (training + test)

## Scripts Organization

### Group 1: LOOCV on Training Data (01-05)
Leave-One-Out Cross-Validation on `MS_samples_Al.xlsx`

- **01_loocv_krr.py** - KRR with LOOCV
- **02_loocv_rf.py** - Random Forest with LOOCV
- **03_loocv_xgb.py** - XGBoost with LOOCV
- **04_oof_loocv_ensemble.py** - Ensemble with LOOCV (OOF method)
- **05_plot_loocv.py** - Plot all LOOCV results (4 subplots)

**Outputs:**
- `loocv_{model}_results.npz` - Predictions, actuals, metal labels (KRR, RF, XGBoost)
- `loocv_ensemble_oof_results.npz` - Predictions, actuals, metal labels (Ensemble-OOF)
- `loocv_{model}_hyperparameters.csv` - Optimized hyperparameters
- `loocv_ensemble_oof_hyperparameters.csv` - Optimized hyperparameters (Ensemble-OOF)
- `loocv_comparison.png` - Combined plot with 4 subplots

### Group 2: Repeated Train-Test Splits (06-09)
10 random train-test splits on `MS_samples_Al.xlsx` for transferability testing

- **06_repeated_krr.py** - KRR with 10 splits
- **07_repeated_rf.py** - Random Forest with 10 splits
- **08_repeated_xgb.py** - XGBoost with 10 splits
- **09_oof_repeated_ensemble.py** - Ensemble with 10 splits (OOF method)

**Outputs:**
- `repeated_{model}_results.csv` - MAE values and hyperparameters for each iteration (KRR, RF, XGBoost)
- `repeated_ensemble_oof_results.csv` - MAE values and hyperparameters for each iteration (Ensemble-OOF)

### Group 3: Transfer Learning (10-14)
Train on `MS_samples_Al.xlsx`, test on `MS_samples_Al_new.xlsx`

- **10_transfer_krr.py** - KRR transfer learning
- **11_transfer_rf.py** - RF transfer learning
- **12_transfer_xgb.py** - XGBoost transfer learning
- **13_oof_transfer_ensemble.py** - Ensemble transfer learning (OOF method)
- **14_plot_transfer.py** - Plot all transfer results (4 subplots)

**Outputs:**
- `transfer_{model}_results.npz` - Predictions, actuals, metal labels on test set
- `transfer_{model}_hyperparameters.csv` - Hyperparameters used
- `transfer_comparison.png` - Combined plot with 4 subplots

### Group 4: LOOCV on All Data (15-19)
Leave-One-Out Cross-Validation on `all_data.xlsx`

- **15_alldata_loocv_krr.py** - KRR with LOOCV
- **16_alldata_loocv_rf.py** - RF with LOOCV
- **17_alldata_loocv_xgb.py** - XGBoost with LOOCV
- **18_alldata_loocv_ensemble.py** - Ensemble with LOOCV
- **19_plot_alldata_loocv.py** - Plot all results (4 subplots)

**Outputs:**
- `alldata_loocv_{model}_results.npz` - Predictions, actuals, metal labels
- `alldata_loocv_{model}_hyperparameters.csv` - Optimized hyperparameters
- `alldata_loocv_comparison.png` - Combined plot with 4 subplots

### Group 5: Repeated Train-Test on All Data (20-23)
10 random train-test splits on `all_data.xlsx`

- **20_alldata_repeated_krr.py** - KRR with 10 splits
- **21_alldata_repeated_rf.py** - RF with 10 splits
- **22_alldata_repeated_xgb.py** - XGBoost with 10 splits
- **23_oof_alldata_repeated_ensemble.py** - Ensemble with 10 splits (OOF method)

**Outputs:**
- `alldata_repeated_{model}_results.csv` - MAE values and hyperparameters for each iteration (KRR, RF, XGBoost)
- `alldata_repeated_ensemble_oof_results.csv` - MAE values and hyperparameters for each iteration (Ensemble-OOF)

## Requirements

Install dependencies:
```bash
pip install numpy pandas cupy-cuda12x scikit-learn xgboost matplotlib tqdm openpyxl
```

**Note:** Replace `cupy-cuda12x` with the appropriate version for your CUDA installation (e.g., `cupy-cuda11x` for CUDA 11.x).

## Execution Instructions

### Quick Start - Run All Scripts

To run the entire pipeline sequentially:

```bash
cd ml_models

# Automated execution (recommended)
bash run_all.sh
```

Or run manually:

```bash
# Group 1: LOOCV on training data
python 01_loocv_krr.py
python 02_loocv_rf.py
python 03_loocv_xgb.py
python 04_oof_loocv_ensemble.py
python 05_plot_loocv.py

# Group 2: Repeated train-test splits
python 06_repeated_krr.py
python 07_repeated_rf.py
python 08_repeated_xgb.py
python 09_oof_repeated_ensemble.py

# Group 3: Transfer learning
python 10_transfer_krr.py
python 11_transfer_rf.py
python 12_transfer_xgb.py
python 13_oof_transfer_ensemble.py
python 14_plot_transfer.py

# Group 4: LOOCV on all data
python 15_alldata_loocv_krr.py
python 16_alldata_loocv_rf.py
python 17_alldata_loocv_xgb.py
python 18_alldata_loocv_ensemble.py
python 19_plot_alldata_loocv.py

# Group 5: Repeated train-test on all data
python 20_alldata_repeated_krr.py
python 21_alldata_repeated_rf.py
python 22_alldata_repeated_xgb.py
python 23_oof_alldata_repeated_ensemble.py
```

### Run by Group

Each group can be executed independently:

**Group 1 (LOOCV on training data):**
```bash
for i in {01..05}; do python ${i}_*.py; done
```

**Group 2 (Repeated splits):**
```bash
for i in {06..09}; do python ${i}_*.py; done
```

**Group 3 (Transfer learning):**
```bash
for i in {10..14}; do python ${i}_*.py; done
```

**Group 4 (LOOCV on all data):**
```bash
for i in {15..19}; do python ${i}_*.py; done
```

**Group 5 (Repeated on all data):**
```bash
for i in {20..23}; do python ${i}_*.py; done
```

### Run Individual Models

To run a specific model across all experiments:

**KRR only:**
```bash
python 01_loocv_krr.py
python 06_repeated_krr.py
python 10_transfer_krr.py
python 15_alldata_loocv_krr.py
python 20_alldata_repeated_krr.py
```

**Random Forest only:**
```bash
python 02_loocv_rf.py
python 07_repeated_rf.py
python 11_transfer_rf.py
python 16_alldata_loocv_rf.py
python 21_alldata_repeated_rf.py
```

**XGBoost only:**
```bash
python 03_loocv_xgb.py
python 08_repeated_xgb.py
python 12_transfer_xgb.py
python 17_alldata_loocv_xgb.py
python 22_alldata_repeated_xgb.py
```

**Ensemble-OOF only:**
```bash
python 04_oof_loocv_ensemble.py
python 09_oof_repeated_ensemble.py
python 13_oof_transfer_ensemble.py
python 18_alldata_loocv_ensemble.py
python 23_oof_alldata_repeated_ensemble.py
```

## Expected Runtime

Approximate execution times (depends on hardware) using 8 CPUs:

- **KRR scripts**: 2 minutes each (GPU accelerated)
- **RF scripts**: 10 minutes each
- **XGBoost scripts**: 10 minutes each
- **Ensemble scripts**: 15 minutes each
- **Plotting scripts**: < 1 minute each

**Total estimated time for full pipeline: ~10-15 hours**

## Output Summary

After running all scripts, you will have:

### Results Files (NPZ format)
- 12 NPZ files containing predictions, actuals, and metal labels

### Hyperparameters (CSV format)
- 20 CSV files with optimized hyperparameters and MAE values

### Figures (PNG format)
- 3 comparison plots (4 subplots each):
  - `loocv_comparison.png`
  - `transfer_comparison.png`
  - `alldata_loocv_comparison.png`

## Model Details

### KRR (Kernel Ridge Regression)
- Custom Laplacian kernel implementation using CuPy (GPU accelerated)
- Hyperparameters: sigma (kernel bandwidth), lambda (regularization)
- Grid search over 20 sigma values and 6 lambda values
- 3-fold cross-validation for hyperparameter optimization

### Random Forest
- Sklearn RandomForestRegressor
- Hyperparameters: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf
- GridSearchCV with 3-fold cross-validation

### XGBoost
- XGBRegressor with objective='reg:squarederror'
- Hyperparameters: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- GridSearchCV with 3-fold cross-validation

### Ensemble-OOF (Stacking with Out-of-Fold Predictions)
- Base models: KRR + XGBoost
- Meta-model: Linear Regression
- Out-of-Fold training procedure:
  1. Generate OOF predictions using 5-fold cross-validation for each base model
  2. Train meta-model on OOF predictions from base models
  3. Train final base models on full training data with averaged hyperparameters
  4. Make final predictions by combining base model outputs through meta-model
- **Advantage**: Maximizes training data usage and avoids overfitting in meta-model

## Utility Module

`utils.py` contains shared functions:
- Data loading and preprocessing
- Feature extraction and one-hot encoding
- CuPy Laplacian kernel implementation
- Plotting functions
- Test data transformation

## Notes

- All scripts use random seed 667 for reproducibility
- Predictions are clipped to [0.0, 1.0] range
- Metal labels preserved for plotting
- Progress bars display via tqdm
- GPU required for KRR and Ensemble models (CUDA compatible)
- **Ensemble models use Out-of-Fold (OOF) predictions** to prevent overfitting and maximize training data usage

## Troubleshooting

**CUDA/CuPy errors:**
- Ensure CUDA is properly installed
- Install correct CuPy version for your CUDA version
- KRR models will fail without GPU support

**Memory errors:**
- Reduce batch size or number of hyperparameters to search
- Close other applications using GPU memory

**Import errors:**
- Ensure all dependencies are installed
- Check Python version compatibility (tested on Python 3.8+)

## Citation

If you use this code, please cite the associated paper on leaching efficiency prediction.

---

**Last Updated:** November 2025
**Author:** ML Models Suite for Leaching Efficiency Prediction
