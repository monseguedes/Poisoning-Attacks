"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Auxiliary module of package 'algorithm' with bounding procedure.
"""

from model.model_class import *
import model.instance_class as data

import numpy as np
from scipy.linalg import eigh

def find_bounds(dataset: data.InstanceData, model):
    """
    Finds bounds for the regression parameters taking into
    account data and bounds of the poisoning samples. More
    details can be founf in overleaf document. 
    """
    
    # Define matrix to be used (X^0^T X^0)
    data_matrix = dataset.x_train_dataframe.to_numpy()
    data_matrix = np.concatenate([data_matrix, np.ones((len(data_matrix), 1))], axis=1)
    transpose_data_matrix = np.transpose(data_matrix)
    matrix = transpose_data_matrix @ data_matrix

    # Find eigenvalues and singular values
    eigenvalues = eigh(matrix, eigvals_only=True)
    highest_eigenvalue = max(eigenvalues)
    highest_singular_value = np.sqrt(highest_eigenvalue)

    # Define response vector
    response_vector = dataset.y_train_dataframe.to_numpy()
    no_poisoning_samples = dataset.no_psamples

    # Bounding term
    bound = (highest_singular_value / (highest_singular_value**2 + model.regularization)) * (np.linalg.norm(response_vector, 2) + np.sqrt(no_poisoning_samples))

    return bound