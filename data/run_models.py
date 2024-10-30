import pandas as pd

# Import model functions
from data.models.bayesian_ridge import bayesian_ridge_model, bayesian_ridge_tuned
from data.models.svr_model import svr_model, svr_tuned
from data.models.random_forest import random_forest_model, random_forest_tuned
from data.models.gradient_boosting import gradient_boosting_model, gradient_boosting_tuned
from data.models.mlp_regressor_model import mlp_regressor_model, mlp_regressor_tuned
from data.models.xgboost_model import xgboost_model, xgboost_tuned
from data.models.stacking_model import stacking_model

def run_all_models(X_train, X_test, y_train, y_test):
    # Dictionary to store results
    results = {}
    
    # Bayesian Ridge Regression
    model, y_pred, mse, mae, r2 = bayesian_ridge_model(X_train, X_test, y_train, y_test)
    results['Bayesian Ridge'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    model, y_pred, mse, mae, r2 = bayesian_ridge_tuned(X_train, X_test, y_train, y_test)
    results['Bayesian Ridge Tuned'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    # Support Vector Regression
    model, y_pred, mse, mae, r2 = svr_model(X_train, X_test, y_train, y_test)
    results['SVR'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    model, y_pred, mse, mae, r2 = svr_tuned(X_train, X_test, y_train, y_test)
    results['SVR Tuned'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    # Random Forest Regression
    model, y_pred, mse, mae, r2 = random_forest_model(X_train, X_test, y_train, y_test)
    results['Random Forest'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    model, y_pred, mse, mae, r2 = random_forest_tuned(X_train, X_test, y_train, y_test)
    results['Random Forest Tuned'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    # Gradient Boosting Regression
    model, y_pred, mse, mae, r2 = gradient_boosting_model(X_train, X_test, y_train, y_test)
    results['Gradient Boosting'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    model, y_pred, mse, mae, r2 = gradient_boosting_tuned(X_train, X_test, y_train, y_test)
    results['Gradient Boosting Tuned'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    # MLP Regression
    model, y_pred, mse, mae, r2 = mlp_regressor_model(X_train, X_test, y_train, y_test)
    results['MLP Regression'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    model, y_pred, mse, mae, r2 = mlp_regressor_tuned(X_train, X_test, y_train, y_test)
    results['MLP Regression Tuned'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    # XGBoost Regression
    model, y_pred, mse, mae, r2 = xgboost_model(X_train, X_test, y_train, y_test)
    results['XGBoost'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    model, y_pred, mse, mae, r2 = xgboost_tuned(X_train, X_test, y_train, y_test)
    results['XGBoost Tuned'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    # Stacking Regressor
    model, y_pred, mse, mae, r2 = stacking_model(X_train, X_test, y_train, y_test)
    results['Stacking Regressor'] = {'model': model, 'MSE': mse, 'MAE': mae, 'R2': r2, 'y_pred': y_pred}
    
    return results
