import numpy as np
import pandas as pd

class myLinearRegression:
    def __init__(self) -> None:
        self.weights = None
        self.intercept = None
    
    def fit(self, X_train, y_train, learning_rate = 0.01, epochs = 1000, tolerance = 1e-6) -> None:
        """
        Fit the linear regression model to the training data.
        
        Parameters:
        X_train (pd.DataFrame or np.array): Training data with shape (m, n).
        y_train (pd.Series or np.array): Target values with shape (m, 1).
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Maximum number of iterations for gradient descent.
        tolerance (float): Threshold for early stopping based on MSE change.
        
        Returns:
        None
        """
        # Ensure X_train and y_train are not empty
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        # Convert pandas objects to numpy arrays
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        
        # Ensure y_train is a column vector
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Get the number of features (n) and instances (m)
        m, n = X_train.shape
        
        # Initialize weights and intercept
        self.weights = np.random.randn(n, 1)  # (n, 1) for matrix multiplication
        self.intercept = np.random.randn()
        
        prev_mse = float('inf')
        
        # Gradient descent loop
        for epoch in range(epochs):
            # Make predictions
            y_pred = self.predict(X_train)
            
            # Compute error and MSE
            error = y_pred - y_train
            mse = np.mean(np.power(error, 2))
            
            # Early stopping if the change in MSE is below the tolerance
            if abs(prev_mse - mse) < tolerance:
                print(f"Converged at epoch {epoch}")
                break
            prev_mse = mse
            
            # Compute gradients
            dw = (2 / m) * np.dot(X_train.T, error)  # (n, m) * (m, 1) -> (n, 1)
            db = (2 / m) * np.sum(error)
            
            # Update weights and intercept
            self.weights -= learning_rate * dw
            self.intercept -= learning_rate * db

    def predict(self, X_test) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters:
        X_test (pd.DataFrame or np.array): Input data with shape (m, n).
        
        Returns:
        np.ndarray: Predictions with shape (m, 1).
        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        
        # Ensure correct matrix multiplication
        return np.dot(X_test, self.weights) + self.intercept
    