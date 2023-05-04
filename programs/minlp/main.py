"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Main script for the paper of poisoning attacks of categorical variables.
"""

import categorical_attack
import instance_data_class
import iterative_attack
import numerical_attack
import numpy as np
import ridge_regression
import flipping_attack

config = {
    # Dataset
    "dataset_name": "10num10cat",
    "training_samples": 100,
    "poison_rate": 4,
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
    "numerical_attack_mini_batch_size": 0.1,
    "numerical_attack_incremental": False,
    # Categorical attack
    "categorical_attack_n_epochs": 1,
    "categorical_attack_mini_batch_size": 0.1,
    "categorical_attack_no_nfeatures": 0,
    "categorical_attack_no_cfeatures": 0,
    # Iterative attack
    "iterative_attack_n_epochs": 1,
    # Flipping attack
    "flipping_attack_n_epochs": 2,
    # Solutions
    "datatype": "test",
}

instance_data = instance_data_class.InstanceData(config)

# import random_categorical_attack
# random_categorical_attack.run(config, instance_data)
# raise SystemExit


# shape = (config["training_samples"], 5)
# np.testing.assert_equal(
#     instance_data.get_num_x_train_dataframe(wide=False).shape, (np.prod(shape),)
# )
# np.testing.assert_equal(instance_data.get_num_x_train_dataframe(wide=True).shape, shape)
# shape = (config["training_samples"], 24)
# np.testing.assert_equal(
#     instance_data.get_cat_x_train_dataframe(wide=False).shape, (np.prod(shape),)
# )
# np.testing.assert_equal(instance_data.get_cat_x_train_dataframe(wide=True).shape, shape)

numerical_model = None

_, instance_data, regression_parameters = flipping_attack.run(
    config, instance_data, numerical_model
)

# numerical_model = None

# _, instance_data, regression_parameters = iterative_attack.run(
#     config, instance_data, numerical_model
# )


## Only optimize numerical and categorical.
# numerical_model, instance_data, regression_parameters = numerical_attack.run(
#     config, instance_data, numerical_model
# )
# # Optimize numerical and categorical.
# categorical_model = None
# categorical_model, instance_data, regression_parameters = categorical_attack.run(
#     config, instance_data, categorical_model
# )


# Run the utitlity to check the results with scikitlearn.
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

    msg = []

    for key in ["weights_num", "weights_cat", "bias", "mse"]:
        a = flatten(sol1[key])
        b = flatten(sol2[key])

        try:
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=key)
        except AssertionError as e:
            # print(e)
            msg.append(str(e))

    if msg:
        raise AssertionError(("\n" + "- " * 30).join(msg))


assert_solutions_are_close(regression_parameters, scikit_learn_regression_parameters)
print("test passed")
