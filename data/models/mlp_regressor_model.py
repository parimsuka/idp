# mlp_regressor_model.py

from data.data_preprocessing import load_and_split_data
from data.models.evaluate_model import evaluate_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()

def mlp_regressor_model(X_train, X_test, y_train, y_test):
    print('MLP Regression')
    mlp = MLPRegressor(random_state=42, max_iter=5000)
    mlp, y_pred, mse, mae, r2 = evaluate_model(mlp, X_train, X_test, y_train, y_test, scaler=scaler)
    return mlp, y_pred, mse, mae, r2

def mlp_regressor_tuned(X_train, X_test, y_train, y_test):
    # Parameters for MLP
    mlp_params = {
        'hidden_layer_sizes': [(50,50), (100,), (2,), (5,)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01]
    }
    mlp_grid = GridSearchCV(MLPRegressor(random_state=42, max_iter=5000), mlp_params, cv=5,
                            scoring='neg_mean_squared_error', n_jobs=-1)
    mlp_grid.fit(scaler.fit_transform(X_train), y_train)
    print(f'Best parameters for MLP: {mlp_grid.best_params_}')
    print(f'Best MSE: {-mlp_grid.best_score_:.4f}')
    best_model = mlp_grid.best_estimator_
    best_model, y_pred, mse, mae, r2 = evaluate_model(best_model, X_train, X_test, y_train, y_test, scaler=scaler)
    return best_model, y_pred, mse, mae, r2
