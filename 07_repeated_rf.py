"""
Script 07: Random Forest with 10 repeated train-test splits
"""

import sys
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

sys.path.append('.')
from utils import get_dataframe, get_features_and_target


def get_rf_hyperparams(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Find optimal RF hyperparameters using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=667)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3,
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_


def main():
    print("="*60)
    print("Script 07: RF Repeated Train-Test Splits (10 iterations)")
    print("="*60)
    
    # Load data
    df, _, _ = get_dataframe('../MS_samples_Al.xlsx')
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
        
        # Optimize hyperparameters
        best_hyperparams = get_rf_hyperparams(X_train, y_train)
        
        # Train and predict
        model = RandomForestRegressor(random_state=667, **best_hyperparams)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions_clipped = np.clip(predictions, 0.0, 1.0)
        
        mae = np.mean(np.abs(predictions_clipped - y_test))
        all_maes.append(mae)
        
        hyperparams_list.append({
            'iteration': i,
            **best_hyperparams,
            'mae': mae
        })
    
    # Save results
    hyperparams_df = pd.DataFrame(hyperparams_list)
    hyperparams_df.to_csv('repeated_rf_results.csv', index=False)
    
    mean_mae = np.mean(all_maes)
    std_mae = np.std(all_maes)
    
    print(f"\nRF Repeated Train-Test Results:")
    print(f"Mean MAE: {mean_mae*100:.2f}% ± {std_mae*100:.2f}%")
    print(f"Results saved to: repeated_rf_results.csv")


if __name__ == '__main__':
    main()
