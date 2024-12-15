import pandas as pd
import random
import os
from pyDOE import lhs

# Number of simulations to generate
N = 5

# Define parameter ranges
param_ranges = {
    'A': [0.1, 0.9],
    'B': [1e-8, 1e-4],
    'C': [0.1, 0.9],
    'D': [0.1, 0.92],
    'E': [1e-8, 8e-7],
    'F': [0.1, 0.9],
    'G': [0.1, 0.9]
}

# Generate and scale LHS samples to parameter ranges
lhs_samples = lhs(len(param_ranges), samples=N)
parameter_df = pd.DataFrame([
    {param: lhs_samples[i, idx] * (bounds[1] - bounds[0]) + bounds[0] for idx, (param, bounds) in enumerate(param_ranges.items())}
    for i in range(N)
])

# # Edge cases
# min_params = {param: bounds[0] for param, bounds in param_ranges.items()}
# max_params = {param: bounds[1] for param, bounds in param_ranges.items()}
# outliers = [
#     {'A': 1.0, 'B': 1e-7, 'C': 0.02, 'D': 1.0, 'E': 1e-7, 'F': 1.0, 'G': 0.02},
#     {'A': 0.02, 'B': 1e-3, 'C': 1.0, 'D': 0.02, 'E': 9e-7, 'F': 0.02, 'G': 1.0}
# ]
#
# # Generate random edge combinations from min, max, and outlier values
# random_combinations = []
# for _ in range(3):  # Adjust the number of random combinations here
#     random_combo = {param: random.choice([min_params[param], max_params[param], outliers[0][param], outliers[1][param]])
#                     for param in min_params.keys()}
#     random_combinations.append(random_combo)
#
# # Concatenate LHS samples, edge cases, and random combinations
# parameter_df = pd.concat([parameter_df, pd.DataFrame([min_params, max_params] + outliers + random_combinations)], ignore_index=True)

# Step 1: Save Parameters to a CSV File
# Define file path for saving parameters and results
file_path = "Generated_validation_parameters_extra.csv"

# Save the parameters to a CSV file if it doesn't exist, otherwise load it
if not os.path.exists(file_path):
    parameter_df.to_csv(file_path, index=False)
    print("Initial Parameters saved to CSV file.")
else:
    print("File already exists. Loading existing parameters.")
    parameter_df = pd.read_csv(file_path)

print("Parameter DataFrame:")
print(parameter_df)
