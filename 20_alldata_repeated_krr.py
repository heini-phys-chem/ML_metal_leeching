"""
Script 20: KRR with 10 repeated train-test splits on all data
"""

import sys
import numpy as np
import cupy as cp
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
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


def main():
    print("="*60)
    print("Script 20: KRR Repeated Train-Test on All Data")
    print("="*60)
    
    df, _, _ = get_dataframe('../all_data.xlsx')
    X, y, _, _ = get_features_and_target(df)
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    
    num_iterations = 10
    test_size = 0.2
    
    all_maes = []
    hyperparams_list = []
    
    for i in tqdm(range(num_iterations), desc="Repeated Splits"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=i
        )
        
        X_train_gpu = cp.asarray(X_train)
        y_train_gpu = cp.asarray(y_train)
        X_test_gpu = cp.asarray(X_test)
        
        best_sigma, best_lambda = get_krr_hyperparams(X_train_gpu, y_train_gpu)
        
        predictions = get_krr_predictions(X_train_gpu, X_test_gpu, y_train_gpu,
                                         best_sigma, best_lambda)
        predictions_clipped = np.clip(predictions, 0.0, 1.0)
        
        mae = np.mean(np.abs(predictions_clipped - y_test))
        all_maes.append(mae)
        
        hyperparams_list.append({
            'iteration': i,
            'sigma': best_sigma,
            'lambda': best_lambda,
            'mae': mae
        })
    
    hyperparams_df = pd.DataFrame(hyperparams_list)
    hyperparams_df.to_csv('alldata_repeated_krr_results.csv', index=False)
    
    mean_mae = np.mean(all_maes)
    std_mae = np.std(all_maes)
    
    print(f"\nKRR All Data Repeated Train-Test Results:")
    print(f"Mean MAE: {mean_mae*100:.2f}% ± {std_mae*100:.2f}%")
    print(f"Results saved to: alldata_repeated_krr_results.csv")


if __name__ == '__main__':
    main()
