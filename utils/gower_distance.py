import numpy as np
import pandas as pd
import gower

def gower_distance(data):
    """
    Calculate the Gower distance matrix for a dataset.
    :param data: Dataset for which to calculate distances (Pandas DataFrame).
    :return: Distance matrix (numpy array).
    """
    def gower_dist(x, y):
        sum_diff = 0
        for i in range(len(x)):
            if np.issubdtype(data.iloc[:, i].dtype, np.number):
                range_i = data.iloc[:, i].max() - data.iloc[:, i].min()
                sum_diff += np.abs(x[i] - y[i]) / range_i if range_i > 0 else 0
            else:
                sum_diff += x[i] != y[i]
        return sum_diff / len(x)
    
    n = len(data)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = gower_dist(data.iloc[i].values, data.iloc[j].values)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances