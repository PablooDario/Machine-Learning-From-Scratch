import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, tol=0.0001, verbose=0):
        """
        Initialize the logistic regression model with specified hyperparameters.

        Parameters:
        - learning_rate (float): The step size for updating weights during training.
        - epochs (int): The maximum number of iterations to train the model.
        - tol (float): The tolerance for convergence, based on the change in loss.
        - verbose(int): Print the iteration in which the model converged
        """
        self.weights = None
        self.intercept = None
        self.alpha = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.verbose = verbose
        self._history = []

    def _sigmoid(self, z: np.array):
        # Apply the sigmoid function to an input.
        return 1 / (1 + np.exp(-z)) # return Sigmoid-transformed values, in the range (0, 1).

    def fit(self, X, y):
        """
        Train the logistic regression model on the given data.

        Parameters:
        - X (ndarray or DataFrame): Feature matrix where each row is a sample and each column is a feature.
        - y (ndarray or Series): Binary target values for each sample.

        Raises:
        - ValueError: If training data is empty.

        Updates:
        - self.weights (ndarray): Coefficients for the features.
        - self.intercept (float): Bias term.
        """
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Training data cannot be empty")
        
        # Transform the data to numpy if necessary
        X = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y

        # Reshape the input data if it only has 1 feature
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Initialize the weights and bias
        m, n = X.shape
        self.weights = np.zeros(n)
        self.intercept = 0.0
        
        old_loss = float('inf')
        for i in range(self.epochs):
            # Lienar Prediction
            z = np.dot(X, self.weights) + self.intercept
            # Probability
            y_pred = self._sigmoid(z)

            loss = (-1 / m) * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            self._history.append(loss)
            # Check convergence
            if abs(old_loss - loss) < self.tol:
                if self.verbose: print(f'Converged at {i} epoch')
                return 
            old_loss = loss
            
            # Get the parcial derivatives
            error = y_pred - y
            dw = (1 / m) * np.dot(X.T, error)
            db = np.mean(error)

            # Update weights
            self.weights -= self.alpha * dw
            self.intercept -= self.alpha * db
            
        if self.verbose: print('STOP. Max iterations reached')

    def predict_proba(self, X):
        # Compute the predicted probabilities for each sample in X.
        z = np.dot(X, self.weights) + self.intercept
        return self._sigmoid(z)

    def predict(self, X):
        # Predict binary class labels for each sample in X.
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int) # return Predicted binary labels (0 or 1) for each sample.
