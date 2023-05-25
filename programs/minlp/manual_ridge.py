# -*- coding: utf-8 -*-

"""Solve ridge regression by matrix inversion"""

import numpy as np
import pandas as pd

import instance_data_class
import ridge_regression


def run(config, instance_data):
    """Compute the weights of the ridge regression model using the formula

    This computes the weights and bias of the ridge regression model
    directly using the formula.
    """
    num_dataframe = pd.concat(
        [
            instance_data.get_num_x_train_dataframe(wide=True),
            instance_data.get_num_x_poison_dataframe(wide=True),
        ]
    )
    cat_dataframe = pd.concat(
        [
            instance_data.get_cat_x_train_dataframe(wide=True),
            instance_data.get_cat_x_poison_dataframe(wide=True),
        ]
    )
    X_df = pd.concat([num_dataframe, cat_dataframe], axis=1)
    X = X_df.to_numpy()
    y_df = pd.concat(
        [
            instance_data.get_y_train_dataframe(),
            instance_data.get_y_poison_dataframe(),
        ]
    )
    y = y_df.to_numpy()

    n_samples = X.shape[0]

    include_bias = True

    if include_bias:
        regularisation_parameter = instance_data.regularization
        regularisation_parameter_matrix = regularisation_parameter * np.eye(
            X.shape[1] + 1
        )
        regularisation_parameter_matrix[-1, -1] = 0
        X_with_one = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        theta = np.linalg.solve(
            X_with_one.T @ X_with_one + n_samples * regularisation_parameter_matrix,
            X_with_one.T @ y,
        )

        _weights_num = theta[: instance_data.no_numfeatures]
        _weights_cat = theta[instance_data.no_numfeatures : -1]
        _weights = theta[:-1]
        bias = theta[-1]

    else:
        regularisation_parameter = instance_data.regularization
        regularisation_parameter_matrix = regularisation_parameter * np.eye(X.shape[1])

        theta = np.linalg.solve(
            X.T @ X + n_samples * regularisation_parameter_matrix,
            X.T @ y,
        )

        _weights_num = theta[: instance_data.no_numfeatures]
        _weights_cat = theta[instance_data.no_numfeatures :]
        _weights = theta
        bias = 0

    y_pred = X @ _weights + bias
    mse = np.mean((y - y_pred) ** 2)

    return {
        "weights_num": _weights_num,
        "weights_cat": _weights_cat,
        "bias": bias,
        "mse": mse,
        "x_poison_num": instance_data.get_num_x_poison_dataframe(),
        "x_poison_cat": instance_data.get_cat_x_poison_dataframe(),
    }


def main():
    config = {
        # Dataset
        "dataset_name": "5num5cat",
        "training_samples": 20,
        "poison_rate": 50,
        "seed": 3,
        # Problem
        "function": "MSE",
        "regularization": 0.6612244897959183,
        "solver_name": "ipopt",
        # Solvers
        "solver_output": False,
        "feasibility": 0.00001,
        "time_limit": 20,
        # Numerical attack
        "numerical_attack_n_epochs": 1,
        "numerical_attack_mini_batch_size": 0.2,
        "numerical_attack_incremental": False,
        # Categorical attack
        "categorical_attack_n_epochs": 1,
        "categorical_attack_mini_batch_size": 0.1,
        "categorical_attack_no_nfeatures": 100,
        "categorical_attack_no_cfeatures": 100,
        # Iterative attack
        "iterative_attack_n_epochs": 10,
        # Flipping attack
        "flipping_attack_n_epochs": 1,
        # Solutions
        "datatype": "test",
    }
    instance_data = instance_data_class.InstanceData(config)

    scikit_learn_regression_parameters = ridge_regression.run(config, instance_data)

    def assert_solutions_are_close(sol1, sol2):
        def flatten(x):
            try:
                x = x.to_numpy()
            except AttributeError:
                pass
            try:
                return x.ravel()
            except AttributeError:
                return x

        for key in ["weights_num", "weights_cat", "bias", "mse"]:
            a = flatten(sol1[key])
            b = flatten(sol2[key])
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=key)

    manual_solution = run(config, instance_data)
    assert_solutions_are_close(manual_solution, scikit_learn_regression_parameters)


if __name__ == "__main__":
    import doctest

    n_failures, _ = doctest.testmod()
    if n_failures > 0:
        raise ValueError(f"{n_failures} tests failed")

    main()

# vimquickrun: python %
