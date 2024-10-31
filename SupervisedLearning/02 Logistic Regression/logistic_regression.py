import numpy as np
import pandas as pd

'''
1. Create a Logistic Regression class
    1.1 Define weights, intercept, alpha, epochs, tolerance
2. Create a fit function
    2.1 Initialize weights and intercept
    2.2 Gradient Descent
3. Create a predit function
'''

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs=1000, tol=0.0001) -> None:
        self.weights = None
        self.intercept = None
        self.alpha = learning_rate
        self.epochs = epochs
        self.tol = tol

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

        
    def fit(self, X, y):
        # Ensure X_train and y_train are not empty
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        # Convert pandas objects to numpy arrays
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        # Ensure X has 2 dimensions
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # Initialize the weights and intercept
        m, n = X.shape
        self.weights = np.zeros(n)
        self.intercept = np.random.randn()
        
        old_loss = float('inf')
        for _ in range(self.epochs):
            # Predictions
            y_pred = self.predict(X)
            # Loss calculation
            loss = (-1/m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            
            # Check Tolerance
            if abs(old_loss - loss) < self.tol:
                break
            
            old_loss = loss  # Update old_loss for next iteration
            error = y_pred - y
            
            # Partial derivatives
            dw = (1/m) * np.dot(X.T, error)
            db = np.mean(error)      
                    
            # Update Weights
            self.weights -= self.alpha * dw
            self.intercept -= self.alpha * db

           
    def predict(self, X):
        z = np.dot(X, self.weights) + self.intercept #X -> (m,n) , weights -> (n,1)
        y_pred = self._sigmoid(z)
        return (y_pred > 0.5).astype(int) # Probabilities for every row 
            