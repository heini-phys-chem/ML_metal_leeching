"""
Script 09_oof: Ensemble (KRR+XGB) with OOF predictions - 10 repeated train-test splits
Training data: MS_samples_Al.xlsx
"""

import sys
import numpy as np
import cupy as cp
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

sys.path.append('.')
from utils import get_dataframe, get_features_and_target, laplacian_kernel_gpu


def get_krr_hyperparams(X_gpu: cp.ndarray, y_gpu: cp.ndarray) -> tuple:
    """Find optimal KRR hyperparameters using 3-fold CV."""
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


def get_xgb_hyperparams(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Find optimal XGBoost hyperparameters using GridSearch."""
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


def get_oof_predictions_krr(X_train_gpu: cp.ndarray, X_test_gpu: cp.ndarray,
                            y_train_gpu: cp.ndarray, sigma: float, 
                            lambda_reg: float, n_folds: int = 5) -> tuple:
    """
    Generate out-of-fold predictions for KRR.
    Returns: (oof_train_preds, test_preds)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=667)
    
    oof_preds = cp.zeros(len(X_train_gpu))
    test_preds_all = []
    
    for train_idx, val_idx in kf.split(X_train_gpu):
        X_fold_train = X_train_gpu[train_idx]
        y_fold_train = y_train_gpu[train_idx]
        X_fold_val = X_train_gpu[val_idx]
        
        # Train on fold
        K_train = laplacian_kernel_gpu(X_fold_train, X_fold_train, sigma)
        C = K_train + cp.eye(len(X_fold_train)) * lambda_reg
        alpha = cp.linalg.solve(C, y_fold_train)
        
        # Predict on validation fold (out-of-fold)
        K_val = laplacian_kernel_gpu(X_fold_train, X_fold_val, sigma)
        oof_preds[val_idx] = cp.dot(K_val.T, alpha)
        
        # Predict on test set
        K_test = laplacian_kernel_gpu(X_fold_train, X_test_gpu, sigma)
        test_preds_all.append(cp.dot(K_test.T, alpha))
    
    # Average test predictions from all folds
    test_preds = cp.mean(cp.stack(test_preds_all), axis=0)
    
    return oof_preds.get(), test_preds.get()


def get_oof_predictions_xgb(X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, best_params: dict,
                            n_folds: int = 5) -> tuple:
    """
    Generate out-of-fold predictions for XGBoost.
    Returns: (oof_train_preds, test_preds)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=667)
    
    oof_preds = np.zeros(len(X_train))
    test_preds_all = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        
        # Train on fold
        model = xgb.XGBRegressor(random_state=667, objective='reg:squarederror',
                                 **best_params)
        model.fit(X_fold_train, y_fold_train)
        
        # Predict on validation fold (out-of-fold)
        oof_preds[val_idx] = model.predict(X_fold_val)
        
        # Predict on test set
        test_preds_all.append(model.predict(X_test))
    
    # Average test predictions from all folds
    test_preds = np.mean(test_preds_all, axis=0)
    
    return oof_preds, test_preds


def main():
    print("="*60)
    print("Script 09_oof: Ensemble (OOF) Repeated Splits")
    print("="*60)
    
    # Load data
    df, _, _ = get_dataframe('../MS_samples_Al.xlsx')
    X, y, _, _ = get_features_and_target(df)
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print("Using Out-of-Fold predictions with 5-fold CV")
    
    num_iterations = 10
    test_size = 0.2
    n_folds = 5
    
    all_maes = []
    hyperparams_list = []
    
    for i in tqdm(range(num_iterations), desc="Repeated Splits"):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i
        )
        
        # Convert to GPU for KRR
        X_train_gpu = cp.asarray(X_train)
        y_train_gpu = cp.asarray(y_train)
        X_test_gpu = cp.asarray(X_test)
        
        # Optimize hyperparameters on full training set
        best_sigma, best_lambda = get_krr_hyperparams(X_train_gpu, y_train_gpu)
        best_xgb_params = get_xgb_hyperparams(X_train, y_train)
        
        # Get OOF predictions for both models
        krr_oof_train, krr_test = get_oof_predictions_krr(
            X_train_gpu, X_test_gpu, y_train_gpu, best_sigma, best_lambda, n_folds
        )
        
        xgb_oof_train, xgb_test = get_oof_predictions_xgb(
            X_train, X_test, y_train, best_xgb_params, n_folds
        )
        
        # Stack predictions
        X_meta_train = np.column_stack((krr_oof_train, xgb_oof_train))
        X_meta_test = np.column_stack((krr_test, xgb_test))
        
        # Train meta-model
        meta_model = LinearRegression()
        meta_model.fit(X_meta_train, y_train)
        
        # Final predictions
        final_predictions = meta_model.predict(X_meta_test)
        predictions_clipped = np.clip(final_predictions, 0.0, 1.0)
        
        # Calculate MAE
        mae = np.mean(np.abs(predictions_clipped - y_test))
        all_maes.append(mae)
        
        # Store hyperparameters
        hyperparams_list.append({
            'iteration': i + 1,
            'krr_sigma': best_sigma,
            'krr_lambda': best_lambda,
            'xgb_n_estimators': best_xgb_params['n_estimators'],
            'xgb_max_depth': best_xgb_params['max_depth'],
            'xgb_learning_rate': best_xgb_params['learning_rate'],
            'meta_coef_krr': meta_model.coef_[0],
            'meta_coef_xgb': meta_model.coef_[1],
            'meta_intercept': meta_model.intercept_,
            'mae': mae
        })
    
    # Save results
    results_df = pd.DataFrame(hyperparams_list)
    results_df.to_csv('repeated_ensemble_oof_results.csv', index=False)
    
    mean_mae = np.mean(all_maes)
    std_mae = np.std(all_maes)
    
    print(f"\nEnsemble (OOF) Repeated Splits MAE: {mean_mae*100:.2f}% ± {std_mae*100:.2f}%")
    print(f"Results saved to: repeated_ensemble_oof_results.csv")


if __name__ == '__main__':
    main()
