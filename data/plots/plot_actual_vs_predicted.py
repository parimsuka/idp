import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_deviation(y_true, y_pred):
    deviation = np.abs((y_pred - y_true) / y_true) * 100
    return np.mean(deviation)

def plot_actual_vs_predicted(y_test, results, base_and_tuned_pairs):
    cols = 2
    rows = len(base_and_tuned_pairs)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten()

    for idx, (base_model, tuned_model) in enumerate(base_and_tuned_pairs):
        ax_base = axes[idx * 2]
        y_pred_base = results[base_model]['y_pred']
        deviation_base = calculate_deviation(y_test, y_pred_base)
        
        ax_base.scatter(y_test, y_pred_base, alpha=0.5)
        ax_base.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        ax_base.set_title(f'{base_model}')
        ax_base.set_xlabel('Actual AOR')
        ax_base.set_ylabel('Predicted AOR')
        ax_base.text(0.05, 0.95, f'Deviation: {deviation_base:.2f}%',
                     transform=ax_base.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white'))
    
        # Tuned model plot
        ax_tuned = axes[idx * 2 + 1]
        y_pred_tuned = results[tuned_model]['y_pred']
        deviation_tuned = calculate_deviation(y_test, y_pred_tuned)
        
        ax_tuned.scatter(y_test, y_pred_tuned, alpha=0.5)
        ax_tuned.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        ax_tuned.set_title(f'{tuned_model}')
        ax_tuned.set_xlabel('Actual AOR')
        ax_tuned.set_ylabel('Predicted AOR')
        ax_tuned.text(0.05, 0.95, f'Deviation: {deviation_tuned:.2f}%',
                      transform=ax_tuned.transAxes, fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white'))

    fig.tight_layout()
    plt.show()

def plot_stacking_regressor(y_test, y_pred_stacking):
    deviation_stacking = calculate_deviation(y_test, y_pred_stacking)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_stacking, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.title('Stacking Regressor')
    plt.xlabel('Actual AOR')
    plt.ylabel('Predicted AOR')
    plt.text(
        0.05, 0.95, f'Deviation: {deviation_stacking:.2f}%', 
        transform=plt.gca().transAxes, 
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white')
    )
    plt.tight_layout()
    plt.show()

def plot_deviation_comparison(y_test, results):
    # Convert results dictionary to DataFrame
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

    # Apply the deviation calculation to each row
    metrics_df['Deviation'] = metrics_df.apply(lambda row: calculate_deviation(y_test, row['y_pred']), axis=1)

    # Sort the DataFrame by Deviation
    metrics_df_sorted = metrics_df.sort_values(by='Deviation')

    # Find the best model
    best_model = metrics_df_sorted.iloc[0]
    print(f"Best model: {best_model['Model']} with {best_model['Deviation']:.2f}% deviation")

    # Plot the deviation percentages
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df_sorted['Model'], metrics_df_sorted['Deviation'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Percentage Deviation (%)')
    plt.title('Model Percentage Deviation Comparison (Sorted)')
    plt.tight_layout()
    plt.show()
