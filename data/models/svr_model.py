# svr_model.py

from data.data_preprocessing import load_and_split_data
from data.models.evaluate_model import evaluate_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

scaler = MinMaxScaler()

def svr_model(X_train, X_test, y_train, y_test):
    print('Support Vector Regression')
    svr = SVR(kernel='rbf')
    svr, y_pred, mse, mae, r2 = evaluate_model(svr, X_train, X_test, y_train, y_test, scaler=scaler)
    return svr, y_pred, mse, mae, r2

def svr_tuned(X_train, X_test, y_train, y_test):
    # Parameters for SVR
    svr_params = [
         {'C': np.logspace(-4, 3, 8), 'kernel': ['linear']},
         {'C': np.logspace(-4, 3, 8), 'gamma': np.logspace(-4, 3, 8), 'kernel': ['rbf']},
         {'C': np.logspace(-4, 3, 8), 'degree': [1, 2, 3, 4, 5], 'kernel': ['poly']},
    ]
    svr_grid = GridSearchCV(SVR(), svr_params, cv=5,
                            scoring='neg_mean_squared_error', n_jobs=-1)
    svr_grid.fit(scaler.fit_transform(X_train), y_train)
    print(f'Best parameters for SVR: {svr_grid.best_params_}')
    print(f'Best MSE: {-svr_grid.best_score_:.4f}')
    best_model = svr_grid.best_estimator_
    best_model, y_pred, mse, mae, r2 = evaluate_model(best_model, X_train, X_test, y_train, y_test, scaler=scaler)
    return best_model, y_pred, mse, mae, r2
