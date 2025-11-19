"""
Script 01: KRR with LOOCV on training data (MS_samples_Al.xlsx)
Uses custom CuPy Laplacian kernel
"""

import sys
import numpy as np
import cupy as cp
from tqdm import tqdm
from sklearn.model_selection import KFold
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
    print("Script 01: KRR LOOCV on Training Data")
    print("="*60)
    
    # Load data
    df, _, _ = get_dataframe('../MS_samples_Al.xlsx')
    X, y, metal_labels, _ = get_features_and_target(df)
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    
    predictions_all = []
    actuals_all = []
    hyperparams_list = []
    
    for i in tqdm(range(X.shape[0]), desc="LOOCV Progress"):
        # LOOCV split
        X_train = np.vstack((X[:i], X[i+1:]))
        X_test = X[i:i+1]
        y_train = np.concatenate((y[:i], y[i+1:]))
        y_test = y[i]
        
        # Move to GPU
        X_train_gpu = cp.asarray(X_train)
        y_train_gpu = cp.asarray(y_train)
        X_test_gpu = cp.asarray(X_test)
        
        # Optimize hyperparameters
        best_sigma, best_lambda = get_krr_hyperparams(X_train_gpu, y_train_gpu)
        hyperparams_list.append({'sigma': best_sigma, 'lambda': best_lambda})
        
        # Predict
        prediction = get_krr_predictions(X_train_gpu, X_test_gpu, y_train_gpu, 
                                         best_sigma, best_lambda)
        prediction_clipped = np.clip(prediction[0], 0.0, 1.0)
        
        predictions_all.append(prediction_clipped)
        actuals_all.append(y_test)
    
    predictions_all = np.array(predictions_all)
    actuals_all = np.array(actuals_all)
    
    # Save results
    np.savez('loocv_krr_results.npz',
             predictions=predictions_all,
             actuals=actuals_all,
             metal_labels=metal_labels)
    
    hyperparams_df = pd.DataFrame(hyperparams_list)
    hyperparams_df.to_csv('loocv_krr_hyperparameters.csv', index=False)
    
    mae = np.mean(np.abs(predictions_all - actuals_all))
    print(f"\nKRR LOOCV MAE: {mae*100:.2f}%")
    print(f"Results saved to: loocv_krr_results.npz")
    print(f"Hyperparameters saved to: loocv_krr_hyperparameters.csv")


if __name__ == '__main__':
    main()
