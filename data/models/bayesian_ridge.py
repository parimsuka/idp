# bayesian_ridge.py

from data.data_preprocessing import load_and_split_data
from data.models.evaluate_model import evaluate_model
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

scaler = MinMaxScaler()

def bayesian_ridge_model(X_train, X_test, y_train, y_test):
    print('Bayesian Ridge Regression')
    bayesian_ridge = BayesianRidge()
    bayesian_ridge, y_pred, mse, mae, r2 = evaluate_model(bayesian_ridge, X_train, X_test, y_train, y_test, scaler=scaler)
    return bayesian_ridge, y_pred, mse, mae, r2

def bayesian_ridge_tuned(X_train, X_test, y_train, y_test):
    # Parameters for Bayesian Ridge
    bayesian_params = {
        'n_iter': [300, 500, 1000],
        'alpha_1': [1e-6, 1e-5, 1e-4],
        'alpha_2': [1e-6, 1e-5, 1e-4],
        'lambda_1': [1e-6, 1e-5, 1e-4],
        'lambda_2': [1e-6, 1e-5, 1e-4],
        'alpha_init':[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9],
        'lambda_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9]
    }
    bayesian_grid = GridSearchCV(BayesianRidge(), bayesian_params, cv=5,
                                 scoring='neg_mean_squared_error', n_jobs=-1)
    bayesian_grid.fit(X_train, y_train)
    print(f'Best parameters for Bayesian Ridge: {bayesian_grid.best_params_}')
    print(f'Best MSE: {-bayesian_grid.best_score_:.4f}')
    best_model = bayesian_grid.best_estimator_
    best_model, y_pred, mse, mae, r2 = evaluate_model(best_model, X_train, X_test, y_train, y_test, scaler=scaler)
    return best_model, y_pred, mse, mae, r2
