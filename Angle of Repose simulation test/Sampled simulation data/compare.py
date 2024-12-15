import pandas as pd

def calculate_percent_deviation(truth, predicted):
    """
    Calculate the percent deviation between two columns.
    :param truth: List or Series of true values (ground truth).
    :param predicted: List or Series of predicted values.
    :return: Percent deviations as a Series.
    """
    return (abs(predicted - truth) / truth) * 100

# Replace 'your_file.csv' with the actual filename
file_path = 'Generated_validation_parameters_with_predictions.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# List of model columns
model_columns = [
    'AOR_XGBoost',
    'AOR_Gradient_Boosting',
    'AOR_Stacking_Regressor'
]

# Calculate percent deviation for each model
for model in model_columns:
    deviation_column_name = f'Percent Deviation ({model})'
    data[deviation_column_name] = calculate_percent_deviation(
        data['AOR from EDEM Simulation'], 
        data[model]
    )

# Print the results
columns_to_display = ['AOR from EDEM Simulation'] + model_columns + [
    f'Percent Deviation ({model})' for model in model_columns
]
print(data[columns_to_display])

# Calculate and print overall percentage deviation for each model
for model in model_columns:
    overall_deviation = data[f'Percent Deviation ({model})'].mean()
    print(f"Overall Percentage Deviation for {model}: {overall_deviation:.2f}%")

# Optionally save the updated DataFrame to a new CSV file
output_file_path = 'Generated_validation_parameters_with_percent_deviation.csv'
data.to_csv(output_file_path, index=False)
print(f"Updated data with percent deviations saved to {output_file_path}")
