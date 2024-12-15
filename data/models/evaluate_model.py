# evaluate_model.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_train, X_test, y_train, y_test, scaler=None):
    if scaler:
        # Scale the data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    # Train the model
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2 Score: {r2:.4f}')
    return model, y_pred, mse, mae, r2



# def evaluate_model(model, X_train, X_test, y_train, y_test, scaler=None):
#     if scaler:
#         # Scale the data
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
#     # Train the model
#     model.fit(X_train, y_train)
#     # Predict
#     y_pred = model.predict(X_test)
#     # Evaluate
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     print(f'MSE: {mse:.4f}')
#     print(f'MAE: {mae:.4f}')
#     print(f'R2 Score: {r2:.4f}')
#     # Plot actual vs predicted
#     plt.figure(figsize=(6,6))
#     plt.scatter(y_test, y_pred, alpha=0.7)
#     plt.xlabel('Actual AOR')
#     plt.ylabel('Predicted AOR')
#     plt.title(f'Actual vs Predicted AOR - {model.__class__.__name__}')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
#     plt.show()
#     return model, y_pred