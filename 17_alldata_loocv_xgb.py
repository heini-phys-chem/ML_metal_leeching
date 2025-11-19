"""
Script 17: XGBoost with LOOCV on all data (all_data.xlsx)
"""

import sys
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd

sys.path.append('.')
from utils import get_dataframe, get_features_and_target


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
    print("Script 17: XGBoost LOOCV on All Data")
    print("="*60)
    
    df, _, _ = get_dataframe('../all_data.xlsx')
    X, y, metal_labels, _ = get_features_and_target(df)
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    
    predictions_all = []
    actuals_all = []
    hyperparams_list = []
    
    for i in tqdm(range(X.shape[0]), desc="LOOCV Progress"):
        X_train = np.vstack((X[:i], X[i+1:]))
        X_test = X[i:i+1]
        y_train = np.concatenate((y[:i], y[i+1:]))
        y_test = y[i]
        
        best_hyperparams = get_xgb_hyperparams(X_train, y_train)
        hyperparams_list.append(best_hyperparams)
        
        model = xgb.XGBRegressor(random_state=667, objective='reg:squarederror',
                                 **best_hyperparams)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        prediction_clipped = np.clip(prediction[0], 0.0, 1.0)
        
        predictions_all.append(prediction_clipped)
        actuals_all.append(y_test)
    
    predictions_all = np.array(predictions_all)
    actuals_all = np.array(actuals_all)
    
    np.savez('alldata_loocv_xgb_results.npz',
             predictions=predictions_all,
             actuals=actuals_all,
             metal_labels=metal_labels)
    
    hyperparams_df = pd.DataFrame(hyperparams_list)
    hyperparams_df.to_csv('alldata_loocv_xgb_hyperparameters.csv', index=False)
    
    mae = np.mean(np.abs(predictions_all - actuals_all))
    print(f"\nXGBoost All Data LOOCV MAE: {mae*100:.2f}%")
    print(f"Results saved to: alldata_loocv_xgb_results.npz")


if __name__ == '__main__':
    main()
