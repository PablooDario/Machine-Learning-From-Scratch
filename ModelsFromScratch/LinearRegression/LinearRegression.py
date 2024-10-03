import numpy as np
import pandas as pd
from ..Losses.metrics import MeanSquaredError

class LinearRegression():
    def __init__(self) -> None:
        self.weights = None
        self.bias = None
    
    def fit(self, X_train, y_train, learning_rate=0.01, epochs=1000):
        # If X_train is a DataFraem cast it to Numpy to do math computations
        if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
            # Make a copy, so we do not alter the original one
            X_train = X_train.copy().to_numpy()
        # Get the shape
        if len(X_train.shape) == 1:
            n = X_train.shape[0]
        n = X_train.shape[1]
        
        # Initialize randomly the weights and the bias
        self.weights = np.random.randn(n)
        self.bias = np.random.randn(1)
        
        # Update the weights until convergence or we reach the maximum of epochs
        prev_mse = float('inf')
        for _ in range(epochs):
            y_pred = self.predict(X_train)
            error = y_pred - y_train
            
            dw = (2 / n) * np.dot(X_train.T, error)  
            db = (2 / n) * np.sum(error)     
            
            self.weights -= learning_rate * 
        
        
    def predict(self, X_test):
        # Return wx + b
        return (np.dot(X_test, self.weights) + self.bias)