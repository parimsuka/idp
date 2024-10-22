# stacking_model.py

from data.models.evaluate_model import evaluate_model
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def stacking_model(X_train, X_test, y_train, y_test):
    print('Stacking Regressor')
    # Define base models with tuned parameters if available
    base_models = [
        ('svr', SVR(C=10, gamma='auto', epsilon=1)),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)),
        ('bayesian', BayesianRidge())
    ]
    # Define the meta-model
    meta_model = LinearRegression()
    # Create the stacking regressor
    stacking_reg = StackingRegressor(estimators=base_models, final_estimator=meta_model, n_jobs=-1)
    # Evaluate the stacking model
    stacking_reg, y_pred, mse, mae, r2 = evaluate_model(
        stacking_reg, X_train, X_test, y_train, y_test, scaler=scaler)
    # Calculate percentage deviation
    percentage_deviation = (mae / y_test.mean()) * 100
    print(f"Percentage Deviation: {percentage_deviation:.2f}%")
    # You can return percentage_deviation if needed
    return stacking_reg, y_pred, mse, mae, r2
