"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Auxiliary module of package 'algorithm' with bounding procedure.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.linalg import norm


def find_bounds(
    x_train_dataframe: pd.DataFrame,
    y_train_dataframe: pd.DataFrame,
    no_poisoning_samples: int,
    config,
):
    """
    Finds bounds for the regression parameters taking into
    account data and bounds of the poisoning samples. More
    details can be founf in overleaf document.
    """

    # Define matrix to be used (X^0^T X^0)
    data_matrix = x_train_dataframe.to_numpy()
    transpose_data_matrix = np.transpose(data_matrix)
    matrix = transpose_data_matrix @ data_matrix

    # Find eigenvalues and singular values
    eigenvalues = eigh(matrix, eigvals_only=True)
    highest_eigenvalue = max(eigenvalues)
    highest_singular_value = np.sqrt(highest_eigenvalue)

    # Define response vector
    response_vector = y_train_dataframe.to_numpy()
    no_samples = len(response_vector) + no_poisoning_samples

    # Bounding weights
    sdv_component = highest_singular_value / (
        highest_singular_value**2 + config["regularization"]
    )
    response_component = (
        norm(response_vector)
        + np.sqrt(no_poisoning_samples)
        + (sum(response_vector) + no_poisoning_samples) / np.sqrt(no_samples)
    )
    bound_weights = sdv_component * response_component

    # Bounding intercept
    ubound_intercept = (
        sum(response_vector) + no_poisoning_samples
    ) / no_samples
    lbound_intercept = sum(response_vector) / no_samples

    return bound_weights, ubound_intercept, lbound_intercept
