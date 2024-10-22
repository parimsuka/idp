# xgboost_model.py

from data.models.evaluate_model import evaluate_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

scaler = MinMaxScaler()

def xgboost_model(X_train, X_test, y_train, y_test):
    print('XGBoost Regression')
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', 
                              colsample_bytree=0.3, learning_rate=0.1, 
                              max_depth=5, alpha=10, n_estimators=200, random_state=42)
    xg_reg, y_pred, mse, mae, r2 = evaluate_model(
        xg_reg, X_train, X_test, y_train, y_test, scaler=scaler)
    return xg_reg, y_pred, mse, mae, r2

def xgboost_tuned(X_train, X_test, y_train, y_test):
    print('Tuned XGBoost Regression')
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    # Parameters for XGBoost
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    grid_search_xgb = GridSearchCV(estimator=xg_reg, param_grid=param_grid_xgb, cv=5,
                                   scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train)
    print(f"Best Parameters for XGBoost: {grid_search_xgb.best_params_}")
    best_model = grid_search_xgb.best_estimator_
    best_model, y_pred, mse, mae, r2 = evaluate_model(
        best_model, X_train, X_test, y_train, y_test, scaler=scaler)
    return best_model, y_pred, mse, mae, r2
