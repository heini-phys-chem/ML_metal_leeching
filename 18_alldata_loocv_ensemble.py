"""
Script 18: Ensemble (KRR+XGB) with LOOCV on all data (all_data.xlsx)
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


def get_krr_predictions(X_train_gpu, X_test_gpu, y_train_gpu, sigma, lambda_reg):
    K_train = laplacian_kernel_gpu(X_train_gpu, X_train_gpu, sigma)
    C = K_train + cp.eye(K_train.shape[0]) * lambda_reg
    alpha = cp.linalg.solve(C, y_train_gpu)
    
    K_test = laplacian_kernel_gpu(X_train_gpu, X_test_gpu, sigma)
    predictions_gpu = cp.dot(K_test.T, alpha)
    
    return predictions_gpu.get()


def get_xgb_hyperparams(X_train, y_train):
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


def main():
    print("="*60)
    print("Script 18: Ensemble LOOCV on All Data")
    print("="*60)
    
    df, _, _ = get_dataframe('../all_data.xlsx')
    X, y, metal_labels, _ = get_features_and_target(df)
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    
    predictions_all = []
    actuals_all = []
    hyperparams_list = []
    
    for i in tqdm(range(X.shape[0]), desc="LOOCV Progress"):
        X_train_outer = np.vstack((X[:i], X[i+1:]))
        X_test_outer = X[i:i+1]
        y_train_outer = np.concatenate((y[:i], y[i+1:]))
        y_test_outer = y[i]
        
        X_base_train, X_meta_train, y_base_train, y_meta_train = train_test_split(
            X_train_outer, y_train_outer, test_size=0.5, random_state=i
        )
        
        X_base_train_gpu = cp.asarray(X_base_train)
        y_base_train_gpu = cp.asarray(y_base_train)
        X_meta_train_gpu = cp.asarray(X_meta_train)
        X_test_outer_gpu = cp.asarray(X_test_outer)
        
        best_sigma, best_lambda = get_krr_hyperparams(X_base_train_gpu, y_base_train_gpu)
        
        krr_meta_preds = get_krr_predictions(X_base_train_gpu, X_meta_train_gpu,
                                             y_base_train_gpu, best_sigma, best_lambda)
        krr_test_preds = get_krr_predictions(X_base_train_gpu, X_test_outer_gpu,
                                             y_base_train_gpu, best_sigma, best_lambda)
        
        best_xgb_params = get_xgb_hyperparams(X_base_train, y_base_train)
        
        xgb_model = xgb.XGBRegressor(random_state=667, objective='reg:squarederror',
                                     **best_xgb_params)
        xgb_model.fit(X_base_train, y_base_train)
        
        xgb_meta_preds = xgb_model.predict(X_meta_train)
        xgb_test_preds = xgb_model.predict(X_test_outer)
        
        X_meta_features = np.column_stack((krr_meta_preds, xgb_meta_preds))
        X_final_test_features = np.column_stack((krr_test_preds, xgb_test_preds))
        
        meta_model = LinearRegression()
        meta_model.fit(X_meta_features, y_meta_train)
        
        final_prediction = meta_model.predict(X_final_test_features)
        prediction_clipped = np.clip(final_prediction[0], 0.0, 1.0)
        
        predictions_all.append(prediction_clipped)
        actuals_all.append(y_test_outer)
        
        hyperparams_list.append({
            'krr_sigma': best_sigma,
            'krr_lambda': best_lambda,
            **{'xgb_' + k: v for k, v in best_xgb_params.items()},
            'meta_coef_krr': meta_model.coef_[0],
            'meta_coef_xgb': meta_model.coef_[1],
            'meta_intercept': meta_model.intercept_
        })
    
    predictions_all = np.array(predictions_all)
    actuals_all = np.array(actuals_all)
    
    np.savez('alldata_loocv_ensemble_results.npz',
             predictions=predictions_all,
             actuals=actuals_all,
             metal_labels=metal_labels)
    
    hyperparams_df = pd.DataFrame(hyperparams_list)
    hyperparams_df.to_csv('alldata_loocv_ensemble_hyperparameters.csv', index=False)
    
    mae = np.mean(np.abs(predictions_all - actuals_all))
    print(f"\nEnsemble All Data LOOCV MAE: {mae*100:.2f}%")
    print(f"Results saved to: alldata_loocv_ensemble_results.npz")


if __name__ == '__main__':
    main()
