"""
Script 10: KRR Transfer Learning
Train on MS_samples_Al.xlsx, predict on MS_samples_Al_new.xlsx
"""

import sys
import numpy as np
import cupy as cp
from sklearn.model_selection import KFold
import pandas as pd

sys.path.append('.')
from utils import (get_dataframe, get_features_and_target, transform_test_data,
                   laplacian_kernel_gpu)


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


def get_krr_predictions(X_train_gpu: cp.ndarray, X_test_gpu: cp.ndarray,
                        y_train_gpu: cp.ndarray, sigma: float, lambda_reg: float) -> np.ndarray:
    """Train KRR and make predictions."""
    K_train = laplacian_kernel_gpu(X_train_gpu, X_train_gpu, sigma)
    C = K_train + cp.eye(K_train.shape[0]) * lambda_reg
    alpha = cp.linalg.solve(C, y_train_gpu)
    
    K_test = laplacian_kernel_gpu(X_train_gpu, X_test_gpu, sigma)
    predictions_gpu = cp.dot(K_test.T, alpha)
    
    return predictions_gpu.get()


def main():
    print("="*60)
    print("Script 10: KRR Transfer Learning")
    print("="*60)
    
    # Load training data
    df_train, scaler, encoder = get_dataframe('../MS_samples_Al.xlsx')
    X_train, y_train, _, _ = get_features_and_target(df_train)
    
    print(f"Training set size: {X_train.shape[0]} samples")
    
    # Load test data
    X_test, y_test, metal_labels_test, _ = transform_test_data(
        '../MS_samples_Al_new.xlsx', scaler, encoder
    )
    
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Move to GPU
    X_train_gpu = cp.asarray(X_train)
    y_train_gpu = cp.asarray(y_train)
    X_test_gpu = cp.asarray(X_test)
    
    # Optimize hyperparameters on training set
    print("Optimizing hyperparameters...")
    best_sigma, best_lambda = get_krr_hyperparams(X_train_gpu, y_train_gpu)
    
    print(f"Best sigma: {best_sigma:.4f}")
    print(f"Best lambda: {best_lambda:.2e}")
    
    # Predict on test set
    print("Making predictions on test set...")
    predictions = get_krr_predictions(X_train_gpu, X_test_gpu, y_train_gpu,
                                     best_sigma, best_lambda)
    predictions_clipped = np.clip(predictions, 0.0, 1.0)
    
    # Save results
    np.savez('transfer_krr_results.npz',
             predictions=predictions_clipped,
             actuals=y_test,
             metal_labels=metal_labels_test)
    
    hyperparams_df = pd.DataFrame([{
        'sigma': best_sigma,
        'lambda': best_lambda
    }])
    hyperparams_df.to_csv('transfer_krr_hyperparameters.csv', index=False)
    
    mae = np.mean(np.abs(predictions_clipped - y_test))
    print(f"\nKRR Transfer Learning MAE: {mae*100:.2f}%")
    print(f"Results saved to: transfer_krr_results.npz")
    print(f"Hyperparameters saved to: transfer_krr_hyperparameters.csv")


if __name__ == '__main__':
    main()
