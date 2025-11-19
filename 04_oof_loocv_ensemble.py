"""
Script 04_oof: Ensemble (KRR + XGBoost Stacking) with Out-of-Fold predictions
This version uses OOF predictions to train the meta-model, avoiding data splitting
and maximizing training data for all models.
"""

import sys
import numpy as np
import cupy as cp
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
import pandas as pd

sys.path.append('.')
from utils import get_dataframe, get_features_and_target, laplacian_kernel_gpu


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
            # Keep as int only for parameters that should be integers
            if key in ['n_estimators', 'max_depth']:
                avg_params[key] = int(avg_value)
            else:
                avg_params[key] = float(avg_value)
        else:
            # For non-numeric params, use mode
            avg_params[key] = params_list[0][key]
    
    return oof_preds, avg_params


def main():
    print("="*60)
    print("Script 04_oof: Ensemble with Out-of-Fold Predictions")
    print("="*60)
    
    # Load data
    df, _, _ = get_dataframe('../MS_samples_Al.xlsx')
    X, y, metal_labels, _ = get_features_and_target(df)
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print("Using Out-of-Fold predictions with 5-fold CV")
    
    predictions_all = []
    actuals_all = []
    hyperparams_list = []
    
    for i in tqdm(range(X.shape[0]), desc="LOOCV Progress"):
        # LOOCV split
        X_train_outer = np.vstack((X[:i], X[i+1:]))
        X_test_outer = X[i:i+1]
        y_train_outer = np.concatenate((y[:i], y[i+1:]))
        y_test_outer = y[i]
        
        # Step 1: Generate OOF predictions for KRR on train_outer
        krr_oof_preds, avg_sigma, avg_lambda = get_oof_predictions_krr(X_train_outer, y_train_outer, n_folds=5)
        
        # Step 2: Generate OOF predictions for XGBoost on train_outer
        xgb_oof_preds, avg_xgb_params = get_oof_predictions_xgb(X_train_outer, y_train_outer, n_folds=5)
        
        # Step 3: Train meta-model on OOF predictions (uses ALL train_outer data!)
        X_meta_features = np.column_stack([krr_oof_preds, xgb_oof_preds])
        meta_model = LinearRegression()
        meta_model.fit(X_meta_features, y_train_outer)
        
        # Step 4: Train base models on full train_outer for test prediction
        # KRR
        X_train_outer_gpu = cp.asarray(X_train_outer)
        y_train_outer_gpu = cp.asarray(y_train_outer)
        X_test_outer_gpu = cp.asarray(X_test_outer)
        
        krr_test_pred = get_krr_predictions(X_train_outer_gpu, X_test_outer_gpu,
                                           y_train_outer_gpu, avg_sigma, avg_lambda)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(random_state=667, objective='reg:squarederror', **avg_xgb_params)
        xgb_model.fit(X_train_outer, y_train_outer)
        xgb_test_pred = xgb_model.predict(X_test_outer)
        
        # Step 5: Meta-model combines base predictions
        X_test_meta = np.column_stack([krr_test_pred, xgb_test_pred])
        final_prediction = meta_model.predict(X_test_meta)
        prediction_clipped = np.clip(final_prediction[0], 0.0, 1.0)
        
        predictions_all.append(prediction_clipped)
        actuals_all.append(y_test_outer)
        
        hyperparams_list.append({
            'krr_sigma': avg_sigma,
            'krr_lambda': avg_lambda,
            **{'xgb_' + k: v for k, v in avg_xgb_params.items()},
            'meta_coef_krr': meta_model.coef_[0],
            'meta_coef_xgb': meta_model.coef_[1],
            'meta_intercept': meta_model.intercept_
        })
    
    predictions_all = np.array(predictions_all)
    actuals_all = np.array(actuals_all)
    
    # Save results
    np.savez('loocv_ensemble_oof_results.npz',
             predictions=predictions_all,
             actuals=actuals_all,
             metal_labels=metal_labels)
    
    hyperparams_df = pd.DataFrame(hyperparams_list)
    hyperparams_df.to_csv('loocv_ensemble_oof_hyperparameters.csv', index=False)
    
    mae = np.mean(np.abs(predictions_all - actuals_all))
    print(f"\nEnsemble (OOF) LOOCV MAE: {mae*100:.2f}%")
    print(f"Results saved to: loocv_ensemble_oof_results.npz")
    print(f"Hyperparameters saved to: loocv_ensemble_oof_hyperparameters.csv")


if __name__ == '__main__':
    main()
