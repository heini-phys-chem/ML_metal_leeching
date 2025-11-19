"""
Script 12: XGBoost Transfer Learning
Train on MS_samples_Al.xlsx, predict on MS_samples_Al_new.xlsx
"""

import sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd

sys.path.append('.')
from utils import get_dataframe, get_features_and_target, transform_test_data


def get_xgb_hyperparams(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Find optimal XGBoost hyperparameters using GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_reg = xgb.XGBRegressor(random_state=667, objective='reg:squarederror')
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=3,
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_


def main():
    print("="*60)
    print("Script 12: XGBoost Transfer Learning")
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
    
    # Optimize hyperparameters on training set
    print("Optimizing hyperparameters...")
    best_hyperparams = get_xgb_hyperparams(X_train, y_train)
    
    print(f"Best hyperparameters: {best_hyperparams}")
    
    # Train on full training set and predict on test set
    print("Training model and making predictions...")
    model = xgb.XGBRegressor(random_state=667, objective='reg:squarederror',
                             **best_hyperparams)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions_clipped = np.clip(predictions, 0.0, 1.0)
    
    # Save results
    np.savez('transfer_xgb_results.npz',
             predictions=predictions_clipped,
             actuals=y_test,
             metal_labels=metal_labels_test)
    
    hyperparams_df = pd.DataFrame([best_hyperparams])
    hyperparams_df.to_csv('transfer_xgb_hyperparameters.csv', index=False)
    
    mae = np.mean(np.abs(predictions_clipped - y_test))
    print(f"\nXGBoost Transfer Learning MAE: {mae*100:.2f}%")
    print(f"Results saved to: transfer_xgb_results.npz")
    print(f"Hyperparameters saved to: transfer_xgb_hyperparameters.csv")


if __name__ == '__main__':
    main()
