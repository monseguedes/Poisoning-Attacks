"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

TODO:fill
"""

# Self-created imports
import instance_data_class 

import numpy as np
import yaml

# Python libraries
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score


def cross_validation_lambda(instance_data, values, print_results=False):
    """
    Function that performs cross validation for the regularization parameter
    """
    
    # Convert data into sklearn format (numpy)
    target = instance_data.validation_dataframe.pop(
        "target"
    ).values  # Numpy array with target values
    feature_matrix = (
        instance_data.validation_dataframe.to_numpy()
    )  # Numpy 2D array with feature matrix

    # Grid search with CV
    grid_space = {"alpha": values}
    ridge = Ridge()
    grid_search = GridSearchCV(
        ridge, grid_space, scoring="neg_mean_squared_error", cv=10
    )
    grid_search.fit(feature_matrix, target)

    if print_results:
        # Print results
        print(" Results from Grid Search ")
        print("-" * 50)
        print("All scores: ", grid_search.cv_results_)
        print(
            "\n The best estimator across ALL searched params:\n",
            grid_search.best_estimator_,
        )
        print(
            "\n The best score across ALL searched params:\n", grid_search.best_score_
        )
        print(
            "\n The best parameters across ALL searched params:\n",
            grid_search.best_params_,
        )

        print("Weights of the best estimator: ", grid_search.best_estimator_.coef_)

    return grid_search.best_params_


if __name__ == "__main__":
    # Define config
    # load config yaml
    with open("programs/minlp/config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    config["dataset"] = "house"
    config["seed"] = 2
    config["dataset_name"] = "allnumallcat"

    # Define values for the regularization parameter
    values = np.linspace(0.001, 10, 20)

    # Create data
    instance_data = instance_data_class.InstanceData(config, thesis=True, seed=config["seed"])

    # Perform cross validation
    best_params = cross_validation_lambda(instance_data, values)
    print(best_params)