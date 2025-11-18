# utils.py
import numpy as np
import pandas as pd

def load_cities_from_csv(path):
    df = pd.read_csv(path)
    return df

def compute_distance_matrix(coords):
    # coords: numpy array shape (n,2)
    n = coords.shape[0]
    d = np.sqrt(((coords.reshape(n,1,2) - coords.reshape(1,n,2))**2).sum(axis=2))
    return d

def route_length(route, dist_matrix):
    # route: list/np.array of indices
    r = np.array(route)
    # distance along path + return to start
    total = dist_matrix[r[:-1], r[1:]].sum() + dist_matrix[r[-1], r[0]]
    return total
