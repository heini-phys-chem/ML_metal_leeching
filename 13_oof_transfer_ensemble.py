"""
Script 13_oof: Ensemble Transfer Learning with Out-of-Fold predictions
Train on MS_samples_Al.xlsx with OOF, predict on MS_samples_Al_new.xlsx
This version uses OOF predictions to train the meta-model, avoiding data splitting
and maximizing training data for all models.
"""

import sys
import numpy as np
import cupy as cp
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
import pandas as pd

sys.path.append('.')
from utils import (get_dataframe, get_features_and_target, transform_test_data,
                   laplacian_kernel_gpu)


def get_krr_hyperparams(X_gpu: cp.ndarray, y_gpu: cp.ndarray) -> tuple:
    """Find optimal KRR hyperparameters."""
    np.random.seed(667)
    sigmas = [0.01 * 2**i for i in range(20)]
    lambdas = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
    
    best_mae = float('inf')
    best_sigma = None
    best_lambda = None
    
    kf = KFold(n_splits=3, shuffle=True, random_state=667)
    
    for sigma in sigmas:
        for lambda_reg in lambdas:
            current_mae = 0.0
            for train_idx, val_idx in kf.split(X_gpu):
                X_train, X_val = X_gpu[train_idx], X_gpu[val_idx]
                y_train, y_val = y_gpu[train_idx], y_gpu[val_idx]
                
                K_train = laplacian_kernel_gpu(X_train, X_train, sigma)
                C = K_train + cp.eye(len(X_train)) * lambda_reg
                alpha = cp.linalg.solve(C, y_train)
                
                K_val = laplacian_kernel_gpu(X_train, X_val, sigma)
                predictions_gpu = cp.dot(K_val.T, alpha)
                
                error = cp.mean(cp.abs(predictions_gpu - y_val))
                current_mae += error.get()
            
            avg_mae = current_mae / kf.get_n_splits()
            
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_sigma = sigma
                best_lambda = lambda_reg
    
    return best_sigma, best_lambda


def get_krr_predictions(X_train_gpu: cp.ndarray, X_test_gpu: cp.ndarray,
                        y_train_gpu: cp.ndarray, sigma: float, lambda_reg: float) -> np.ndarray:
    """Train KRR and make predictions."""
    K_train = laplacian_kernel_gpu(X_train_gpu, X_train_gpu, sigma)
    C = K_train + cp.eye(K_train.shape[0]) * lambda_reg
    alpha = cp.linalg.solve(C, y_train_gpu)
    
    K_test = laplacian_kernel_gpu(X_train_gpu, X_test_gpu, sigma)
    predictions_gpu = cp.dot(K_test.T, alpha)
    
    return predictions_gpu.get()


def get_xgb_hyperparams(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Find optimal XGBoost hyperparameters."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
    }
    
    xgb_reg = xgb.XGBRegressor(random_state=667, objective='reg:squarederror', n_jobs=1)
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=3,
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_


def get_oof_predictions_krr(X_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5) -> tuple:
    """
    Generate out-of-fold predictions for KRR using K-Fold CV.
    Returns OOF predictions and averaged hyperparameters.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=667)
    oof_preds = np.zeros(len(X_train))
    sigma_list = []
    lambda_list = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train[train_idx]
        
        # Move to GPU
        X_fold_train_gpu = cp.asarray(X_fold_train)
        y_fold_train_gpu = cp.asarray(y_fold_train)
        X_fold_val_gpu = cp.asarray(X_fold_val)
        
        # Optimize hyperparameters on this fold
        best_sigma, best_lambda = get_krr_hyperparams(X_fold_train_gpu, y_fold_train_gpu)
        sigma_list.append(best_sigma)
        lambda_list.append(best_lambda)
        
        # Get predictions
        preds = get_krr_predictions(X_fold_train_gpu, X_fold_val_gpu,
                                    y_fold_train_gpu, best_sigma, best_lambda)
        oof_preds[val_idx] = preds
    
    # Return OOF predictions and average hyperparameters
    avg_sigma = np.mean(sigma_list)
    avg_lambda = np.mean(lambda_list)
    
    return oof_preds, avg_sigma, avg_lambda


def get_oof_predictions_xgb(X_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5) -> tuple:
    """
    Generate out-of-fold predictions for XGBoost using K-Fold CV.
    Returns OOF predictions and averaged hyperparameters.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=667)
    oof_preds = np.zeros(len(X_train))
    params_list = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train[train_idx]
        
        # Optimize hyperparameters on this fold
        best_params = get_xgb_hyperparams(X_fold_train, y_fold_train)
        params_list.append(best_params)
        
        # Train and predict
        model = xgb.XGBRegressor(random_state=667, objective='reg:squarederror', **best_params)
        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)
        
        oof_preds[val_idx] = preds
    
    # Average hyperparameters
    avg_params = {}
    for key in params_list[0].keys():
        if isinstance(params_list[0][key], (int, float)):
            values = [p[key] for p in params_list]
            avg_value = np.mean(values)
            if key in ['n_estimators', 'max_depth']:
                avg_params[key] = int(avg_value)
            else:
                avg_params[key] = float(avg_value)
        else:
            avg_params[key] = params_list[0][key]
    
    return oof_preds, avg_params


def main():
    print("="*60)
    print("Script 13_oof: Ensemble Transfer Learning with OOF")
    print("="*60)
    
    # Load training data
    df_train, scaler, encoder = get_dataframe('../MS_samples_Al.xlsx')
    X_train_full, y_train_full, _, _ = get_features_and_target(df_train)
    
    print(f"Training set size: {X_train_full.shape[0]} samples")
    print("Using Out-of-Fold predictions with 5-fold CV")
    
    # Load test data
    X_test, y_test, metal_labels_test, _ = transform_test_data(
        '../MS_samples_Al_new.xlsx', scaler, encoder
    )
    
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Generate OOF predictions for base models
    print("\nGenerating OOF predictions for KRR...")
    krr_oof_preds, avg_sigma, avg_lambda = get_oof_predictions_krr(X_train_full, y_train_full, n_folds=5)
    print(f"KRR - avg sigma: {avg_sigma:.4f}, avg lambda: {avg_lambda:.2e}")
    
    print("\nGenerating OOF predictions for XGBoost...")
    xgb_oof_preds, avg_xgb_params = get_oof_predictions_xgb(X_train_full, y_train_full, n_folds=5)
    print(f"XGBoost - avg params: {avg_xgb_params}")
    
    # Train meta-model on OOF predictions
    print("\nTraining meta-model on OOF predictions...")
    X_meta_features = np.column_stack((krr_oof_preds, xgb_oof_preds))
    
    meta_model = LinearRegression()
    meta_model.fit(X_meta_features, y_train_full)
    
    # Train final base models on full training data with averaged hyperparameters
    print("\nTraining final base models on full training data...")
    X_train_full_gpu = cp.asarray(X_train_full)
    y_train_full_gpu = cp.asarray(y_train_full)
    X_test_gpu = cp.asarray(X_test)
    
    # Final KRR predictions
    krr_test_preds = get_krr_predictions(X_train_full_gpu, X_test_gpu,
                                         y_train_full_gpu, avg_sigma, avg_lambda)
    
    # Final XGBoost predictions
    xgb_model = xgb.XGBRegressor(random_state=667, objective='reg:squarederror',
                                 **avg_xgb_params)
    xgb_model.fit(X_train_full, y_train_full)
    xgb_test_preds = xgb_model.predict(X_test)
    
    # Final ensemble predictions
    X_final_test_features = np.column_stack((krr_test_preds, xgb_test_preds))
    predictions = meta_model.predict(X_final_test_features)
    predictions_clipped = np.clip(predictions, 0.0, 1.0)
    
    # Save results
    np.savez('transfer_ensemble_oof_results.npz',
             predictions=predictions_clipped,
             actuals=y_test,
             metal_labels=metal_labels_test)
    
    hyperparams_df = pd.DataFrame([{
        'krr_sigma': avg_sigma,
        'krr_lambda': avg_lambda,
        **{'xgb_' + k: v for k, v in avg_xgb_params.items()},
        'meta_coef_krr': meta_model.coef_[0],
        'meta_coef_xgb': meta_model.coef_[1],
        'meta_intercept': meta_model.intercept_
    }])
    hyperparams_df.to_csv('transfer_ensemble_oof_hyperparameters.csv', index=False)
    
    mae = np.mean(np.abs(predictions_clipped - y_test))
    print(f"\nEnsemble Transfer Learning (OOF) MAE: {mae*100:.2f}%")
    print(f"Results saved to: transfer_ensemble_oof_results.npz")
    print(f"Hyperparameters saved to: transfer_ensemble_oof_hyperparameters.csv")


if __name__ == '__main__':
    main()
