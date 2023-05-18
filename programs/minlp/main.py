"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Main script for the paper of poisoning attacks of categorical variables.
"""

import os

import binary_attack
import categorical_attack
import flipping_attack
import full_flipping_attack
import instance_data_class
import iterative_attack
import numerical_attack
import numpy as np
import ridge_regression
import testing
import computational_experiments

config = {
    # Dataset
    "dataset_name": "5num5cat",
    "training_samples": 100,
    "poison_rate": 20,
    "seed": 3,
    # Problem
    "function": "MSE",
    "binary": False,
    "regularization": 0.6612244897959183,
    "solver_name": "ipopt",
    # Solvers
    "solver_output": False,
    "feasibility": 0.00001,
    "time_limit": 600,
    # Numerical attack
    "numerical_attack_n_epochs": 1,
    "numerical_attack_mini_batch_size": 0.5,
    "numerical_attack_incremental": False,
    # Categorical attack
    "categorical_attack_n_epochs": 1,
    "categorical_attack_mini_batch_size": 0.1,
    "categorical_attack_no_nfeatures": 0,
    "categorical_attack_no_cfeatures": 0,
    # Iterative attack
    "iterative_attack_n_epochs": 1,
    # Flipping attack
    "flipping_attack_n_epochs": 1,
    "return_benchmark": False,
    # Solutions
    "datatype": "test",
}

for poisoning_rate in [4, 8, 12, 16, 20]:
    config["poison_rate"] = poisoning_rate
    computational_experiments.run(40, config)


instance_data = instance_data_class.InstanceData(config)


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

# numerical_model = None

# try:
#     os.mkdir("programs/minlp/attacks/" + config["dataset_name"])
# except:
#     pass
# for file in os.listdir(f"programs/minlp/attacks/{config['dataset_name']}"):
#     os.remove(f"programs/minlp/attacks/{config['dataset_name']}/{file}")
# instance_data.train_dataframe.to_csv(
#     f"programs/minlp/attacks/{config['dataset_name']}/training_data.csv"
# )
# scikit_learn_regression_parameters = ridge_regression.run_just_training(
#     config, instance_data
# )
# _, instance_data, regression_parameters = flipping_attack.run(
#     config, instance_data, numerical_model
# )

# numerical_model, instance_data, regression_parameters = binary_attack.run(
#     config, instance_data, numerical_model
# )


# # Run the utitlity to check the results with scikitlearn
# scikit_learn_regression_parameters = ridge_regression.run(config, instance_data)


# testing.assert_solutions_are_close(
#     regression_parameters, scikit_learn_regression_parameters
# )
# print("test passed")
