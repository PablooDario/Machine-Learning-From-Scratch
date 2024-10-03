import numpy as np
import pandas as pd

def MeanSquaredError(y_true, y_pred):
    # Cast y_true and y_pred from Pandas.Series to Numpy.Array if necessary
    if isinstance(y_true, pd.Series):
        y_true = y_true.copy().to_numpy()
    if isinstance(y_pred, pd.Series):
        y_true = y_pred.copy().to_numpy()
    
    # Return 1/n * (y_true - y_pred)**2
    return np.mean(np.power(y_true - y_pred, 2))