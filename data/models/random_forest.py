# random_forest.py

from data.data_preprocessing import load_and_split_data
from data.models.evaluate_model import evaluate_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def random_forest_model(X_train, X_test, y_train, y_test):
    print('Random Forest Regression')
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf, y_pred, mse, mae, r2 = evaluate_model(rf, X_train, X_test, y_train, y_test)
    return rf, y_pred, mse, mae, r2

def random_forest_tuned(X_train, X_test, y_train, y_test):
    # Parameters for Random Forest
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f'Best parameters for Random Forest: {rf_grid.best_params_}')
    print(f'Best MSE: {-rf_grid.best_score_:.4f}')
    best_model = rf_grid.best_estimator_
    best_model, y_pred, mse, mae, r2 = evaluate_model(best_model, X_train, X_test, y_train, y_test)
    return best_model, y_pred, mse, mae, r2
