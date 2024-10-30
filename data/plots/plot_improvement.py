import pandas as pd
import matplotlib.pyplot as plt

def calculate_percentage_change(base_value, tuned_value, reverse=False):
    if reverse:
        return ((tuned_value - base_value) / base_value) * 100 # For R² where higher is better
    else:
        return ((base_value - tuned_value) / base_value) * 100 # For MSE, MAE where lower is better

def plot_improvement(metrics_df, base_and_tuned_pairs):
    improvement_data = []
    for base_model, tuned_model in base_and_tuned_pairs:
        base_mse = metrics_df.loc[metrics_df['Model'] == base_model, 'MSE'].values[0]
        tuned_mse = metrics_df.loc[metrics_df['Model'] == tuned_model, 'MSE'].values[0]
        
        base_mae = metrics_df.loc[metrics_df['Model'] == base_model, 'MAE'].values[0]
        tuned_mae = metrics_df.loc[metrics_df['Model'] == tuned_model, 'MAE'].values[0]
        
        base_r2 = metrics_df.loc[metrics_df['Model'] == base_model, 'R2'].values[0]
        tuned_r2 = metrics_df.loc[metrics_df['Model'] == tuned_model, 'R2'].values[0]

        mse_change = calculate_percentage_change(base_mse, tuned_mse)
        mae_change = calculate_percentage_change(base_mae, tuned_mae)
        r2_change = calculate_percentage_change(base_r2, tuned_r2, reverse=True)
        
        improvement_data.append((base_model, mse_change, mae_change, r2_change))

    improvement_df = pd.DataFrame(improvement_data, columns=['Model', 'MSE Improvement (%)', 'MAE Improvement (%)', 'R² Improvement (%)'])

    # MSE Improvement
    plt.figure(figsize=(10, 6))
    plt.bar(improvement_df['Model'], improvement_df['MSE Improvement (%)'], color='skyblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MSE Improvement/Degradation (%)')
    plt.title('MSE Percentage Improvement After Tuning')
    plt.tight_layout()
    plt.show()

    # MAE Improvement
    plt.figure(figsize=(10, 6))
    plt.bar(improvement_df['Model'], improvement_df['MAE Improvement (%)'], color='lightgreen')
    plt.axhline(0, color='red', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('MAE Improvement/Degradation (%)')
    plt.title('MAE Percentage Improvement After Tuning')
    plt.tight_layout()
    plt.show()

    # R² Improvement
    plt.figure(figsize=(10, 6))
    plt.bar(improvement_df['Model'], improvement_df['R² Improvement (%)'], color='orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('R² Improvement/Degradation (%)')
    plt.title('R² Percentage Improvement After Tuning')
    plt.tight_layout()
    plt.show()
