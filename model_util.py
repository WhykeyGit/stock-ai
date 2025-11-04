import numpy as np
import pandas as pd

def train_test_split(data, test_size=0.2):
    """
    Splits data into training and testing sets.
    
    Args:
        data: pd.DataFrame or np.ndarray
        test_size: float, proportion of data to use for testing
    """
    if isinstance(data, pd.DataFrame):
        data = data.values  # Convert to numpy array for splitting
    
    n_samples = data.shape[0]
    n_test = int(n_samples * test_size)
    
    train_data = data[:-n_test]
    test_data = data[-n_test:]
    
    return train_data, test_data