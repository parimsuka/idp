import matplotlib.pyplot as plt

def plot_model_metrics(metrics_df):
    # Plot R² Scores
    metrics_df_sorted_r2 = metrics_df.sort_values(by='R2', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(metrics_df_sorted_r2['Model'], metrics_df_sorted_r2['R2'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('R² Score')
    plt.title('R² Scores of Models (Sorted)')
    plt.tight_layout()
    plt.show()

    # Plot MSE Comparison
    metrics_df_sorted_mse = metrics_df.sort_values(by='MSE', ascending=True)
    plt.figure(figsize=(12, 6))
    plt.bar(metrics_df_sorted_mse['Model'], metrics_df_sorted_mse['MSE'], color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE of Models (Sorted)')
    plt.tight_layout()
    plt.show()

    # Plot MAE Comparison
    metrics_df_sorted_mae = metrics_df.sort_values(by='MAE', ascending=True)
    plt.figure(figsize=(12, 6))
    plt.bar(metrics_df_sorted_mae['Model'], metrics_df_sorted_mae['MAE'], color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE of Models (Sorted)')
    plt.tight_layout()
    plt.show()
