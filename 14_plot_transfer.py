"""
Script 14: Plot Transfer Learning results for all 4 models
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

sys.path.append('.')
from utils import create_subplot_scatterplot, setup_plot_style


def main():
    print("="*60)
    print("Script 14: Plotting Transfer Learning Results")
    print("="*60)
    
    setup_plot_style()
    
    # Load all results
    models = ['krr', 'rf', 'xgb', 'ensemble']
    model_names = ['KRR', 'Random Forest', 'XGBoost', 'Ensemble (KRR+XGB)']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        # Load results
        data = np.load(f'transfer_{model}_results.npz', allow_pickle=True)
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
    plt.savefig('transfer_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: transfer_comparison.png")


if __name__ == '__main__':
    main()
