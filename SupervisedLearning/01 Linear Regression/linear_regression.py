import numpy as np
import pandas as pd

class myLinearRegression:
    def __init__(self) -> None:
        self.weights = None
        self.intercept = None
    
    def fit(self, X, y, learning_rate = 0.01, epochs = 1000, tolerance = 1e-6) -> None:
        """
        Fit the linear regression model to the training data.
        
        Parameters:
        X (pd.DataFrame or np.array): Training data with shape (m, n).
        y (pd.Series or np.array): Target values with shape (m, 1).
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Maximum number of iterations for gradient descent.
        tolerance (float): Threshold for early stopping based on MSE change.
        
        Returns:
        None
        """
        # Ensure X and y are not empty
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        # Convert pandas objects to numpy arrays
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        # Ensure y is a column vector
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Get the number of features (n) and instances (m)
        m, n = X.shape
        
        # Initialize weights and intercept
        self.weights = np.random.randn(n, 1)  # (n, 1) for matrix multiplication
        self.intercept = np.random.randn()
        
        prev_mse = float('inf')
        
        # Gradient descent loop
        for epoch in range(epochs):
            # Make predictions
            y_pred = self.predict(X)
            
            # Compute error and MSE
            error = y_pred - y
            mse = np.mean(np.power(error, 2))
            
            # Early stopping if the change in MSE is below the tolerance
            if abs(prev_mse - mse) < tolerance:
                print(f"Converged at epoch {epoch}")
                break
            prev_mse = mse
            
            # Compute gradients
            dw = (2 / m) * np.dot(X.T, error)  # (n, m) * (m, 1) -> (n, 1)
            db = (2 / m) * np.sum(error)
            
            # Update weights and intercept
            self.weights -= learning_rate * dw
            self.intercept -= learning_rate * db

    def predict(self, X) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters:
        X (pd.DataFrame or np.array): Input data with shape (m, n).
        
        Returns:
        np.ndarray: Predictions with shape (m, 1).
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        # Ensure correct matrix multiplication
        return np.dot(X, self.weights) + self.intercept
    