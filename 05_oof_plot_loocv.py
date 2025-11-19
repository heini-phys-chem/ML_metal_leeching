"""
Script 05_oof: Plot LOOCV results comparing KRR, RF, XGB, and Ensemble (OOF)
Replaces old ensemble with new OOF ensemble
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

sys.path.append('.')
from utils import create_subplot_scatterplot, setup_plot_style


def main():
    print("="*60)
    print("Script 05_oof: Plotting LOOCV Results (with OOF Ensemble)")
    print("="*60)
    
    setup_plot_style()
    
    # Load all results - using OOF ensemble instead of regular ensemble
    models = ['krr', 'rf', 'xgb', 'ensemble_oof']
    model_names = ['KRR', 'Random Forest', 'XGBoost', 'Ensemble-OOF (KRR+XGB)']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        # Load results - handle ensemble_oof special case
        if model == 'ensemble_oof':
            data = np.load('loocv_ensemble_oof_results.npz', allow_pickle=True)
        else:
            data = np.load(f'loocv_{model}_results.npz', allow_pickle=True)
        predictions = data['predictions']
        actuals = data['actuals']
        metal_labels = data['metal_labels']
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals)) * 100
        r2 = r2_score(actuals, predictions)
        
        # Plot
        create_subplot_scatterplot(axes[idx], actuals, predictions, metal_labels,
                                   mae, r2, model_name)
        
        print(f"{model_name}: MAE = {mae:.2f}%, R² = {r2:.3f}")
    
    plt.tight_layout()
    plt.savefig('loocv_comparison_oof.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: loocv_comparison_oof.png")


if __name__ == '__main__':
    main()
