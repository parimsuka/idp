import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def predict_from_model(results, model_name, input_values, X_train):
    """
    Predict the outcome using a pre-trained model from the results dictionary.

    Args:
    - results (dict): Dictionary containing model results and trained models.
    - model_name (str): Name of the model in the results dictionary to use for prediction.
    - input_values (list): List of feature values for prediction.
    - feature_names (list): List of feature names corresponding to input values.

    Returns:
    - prediction: The model prediction for the given input values.
    """
    # Ensure the model exists in the results dictionary
    if model_name not in results:
        raise ValueError(f"Model '{model_name}' not found in the results.")

    # Initialize and fit the scaler on the training data (assumes StandardScaler was used)
    scaler = StandardScaler()
    scaler.fit_transform(X_train)

    # Prepare input DataFrame and scale it
    manual_input = pd.DataFrame([input_values], columns=X_train.columns)
    scaled_input = scaler.transform(manual_input)

    # Predict using the specified model
    model = results[model_name]['model']
    prediction = model.predict(scaled_input)

    return prediction


def predict_from_saved_model(model_path, scaler_path, input_values, feature_names):
    # Load the model from the file
    model = joblib.load(model_path)
    # print(f"Loaded model from {model_path}")

    # Load the scaler from the file
    scaler = joblib.load(scaler_path)
    # print(f"Loaded scaler from {scaler_path}")
    
    manual_input = pd.DataFrame([input_values], columns=feature_names)
    scaled_input = scaler.transform(manual_input)
    
    # Predict using the loaded model
    prediction = model.predict(scaled_input)
    return prediction
