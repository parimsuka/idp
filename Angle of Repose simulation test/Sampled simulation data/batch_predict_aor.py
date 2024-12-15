import pandas as pd
import joblib

# Function to predict AOR from saved model and scaler
def predict_from_saved_model(model_path, scaler_path, input_values, feature_names):
    # Load the model and scaler from files
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare input DataFrame and scale it
    manual_input = pd.DataFrame([input_values], columns=feature_names)
    scaled_input = scaler.transform(manual_input)
    
    # Predict using the loaded model
    prediction = model.predict(scaled_input)
    return prediction[0]

# File paths for models and scalers
xgboost_model_path = "../../models/xgboost_model.joblib"
gradient_boosting_model_path = "../../models/gradient_boosting_model.joblib"
stacking_regressor_model_path = "../../models/stacking_regressor_model.joblib"

# File paths for scalers
minmax_scaler_path = "../../models/minmax_scaler.joblib"
standard_scaler_path = "../../models/standard_scaler.joblib"

# CSV input and output paths
csv_path = "Generated_validation_parameters_with_results_from_EDEM.csv"

# Define feature names in the correct order as they appear in the input CSV
feature_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Load the input CSV file
data = pd.read_csv(csv_path)

# Ensure the input file has the necessary columns
if not all(col in data.columns for col in feature_names):
    raise ValueError(f"The input CSV file must contain columns: {', '.join(feature_names)}")

# Predict AOR for each row using all three models and add to new columns
data['AOR_XGBoost'] = data[feature_names].apply(
    lambda row: predict_from_saved_model(xgboost_model_path, minmax_scaler_path, row.values, feature_names),
    axis=1
)

data['AOR_Gradient_Boosting'] = data[feature_names].apply(
    lambda row: predict_from_saved_model(gradient_boosting_model_path, minmax_scaler_path, row.values, feature_names),
    axis=1
)

data['AOR_Stacking_Regressor'] = data[feature_names].apply(
    lambda row: predict_from_saved_model(stacking_regressor_model_path, standard_scaler_path, row.values, feature_names),
    axis=1
)

# Save the result to a new CSV file
output_csv_path = "Generated_validation_parameters_with_predictions.csv"
data.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
