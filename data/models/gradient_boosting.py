# gradient_boosting.py

from data.data_preprocessing import load_and_split_data
from data.models.evaluate_model import evaluate_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

scaler = MinMaxScaler()

def gradient_boosting_model(X_train, X_test, y_train, y_test):
    print('Gradient Boosting Regression')
    gb = GradientBoostingRegressor(random_state=42)
    gb, y_pred, mse, mae, r2 = evaluate_model(gb, X_train, X_test, y_train, y_test, scaler=scaler)
    return gb, y_pred, mse, mae, r2

def gradient_boosting_tuned(X_train, X_test, y_train, y_test):
    # Parameters for Gradient Boosting
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 5]
    }
    # gb_params = {
    #     'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
    #     'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
    #     'max_depth': [1, 5, 10, 15, 20, 30],
    #     'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
    #     'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
    #     'max_features': list(range(1,X.shape[1]))
    # }
    gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    print(f'Best parameters for Gradient Boosting: {gb_grid.best_params_}')
    print(f'Best MSE: {-gb_grid.best_score_:.4f}')
    best_model = gb_grid.best_estimator_
    best_model, y_pred, mse, mae, r2 = evaluate_model(best_model, X_train, X_test, y_train, y_test, scaler=scaler)
    return best_model, y_pred, mse, mae, r2
