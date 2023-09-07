"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Main script for the paper of poisoning attacks of categorical variables.
"""

import os

import numpy as np

import all_attack
import binary_attack
import categorical_attack
import computational_experiments
import flipping_attack
import instance_data_class
import numerical_attack
import plots
import ridge_regression
import testing

config = {
    # Dataset
    "dataset_name": "5num5cat",
    "training_samples": 300,
    "poison_rate": 20,
    "seed": 3,
    # Problem
    "function": "MSE",
    "binary": False,
    # "regularization": 0.6612244897959183,
    "regularization": 0.1,
    "solver_name": "neos",
    # Solvers
    "solver_output": False,
    "feasibility": 0.00001,
    "time_limit": 300,
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
    # Results
    "runs": 50,
    "poison_rates": [4, 8, 12, 16, 20],
    "data_type": "train",
}

if __name__ == "__main__":
    pass

    # instance_data = instance_data_class.InstanceData(config)

    # all_attack.run(config, instance_data, solver='gurobi')

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

    # # Run the utitlity to check the results with scikitlearn
    # scikit_learn_regression_parameters = ridge_regression.run(config, instance_data)

    # testing.assert_solutions_are_close(
    #     regression_parameters, scikit_learn_regression_parameters
    # )
    # print("test passed")
