import collections
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

import binary_attack
import flipping_attack
import instance_data_class
import numerical_attack
import plots
import pyomo_model
import ridge_regression


def run(config):
    """Function to run computational experiment
    and save results to csv files.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration parameters.

    Returns
    -------
    None.

    """

    # Create directory to store results
    folder_name = f"{config['runs']}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_PR{config['poison_rate']}_lambda{config['regularization']}"
    isExist = os.path.exists(f"results/{config['dataset_name']}/{folder_name}")
    if not isExist:
        os.makedirs(f"results/{config['dataset_name']}/{folder_name}")

    # Create dictionaries to store results
    flipping_results = collections.defaultdict(list)
    benchmark_results = collections.defaultdict(list)
    unpoisoned_results = collections.defaultdict(list)
    binary_results = collections.defaultdict(list)

    # TODO fix not build model for each run

    for run in range(config["runs"]):
        print(f"Run {run+1} of {config['runs']}")
        print(f"Poisoning rate: {config['poison_rate']}")
        config["seed"] = run
        instance_data = instance_data_class.InstanceData(config)
        # model = pyomo_model.PyomoModel(instance_data, config)

        ## Unpoisoned model------------------------------------------------------
        regression_parameters = ridge_regression.run_not_poisoned(
            config, instance_data
        )
        # Save results to dictionary
        unpoisoned_results["mse_final"].append(regression_parameters["mse"])
        unpoisoned_results["weights_num"].append(
            regression_parameters["weights_num"]
        )
        unpoisoned_results["weights_cat"].append(
            regression_parameters["weights_cat"]
        )
        unpoisoned_results["bias"].append(regression_parameters["bias"])

        ## Flipping attack-------------------------------------------------------
        _, _, flipping_solutions = flipping_attack.run(config, instance_data)
        # Save results to dictionary
        flipping_results["mse_per_iteration"].append(
            flipping_solutions["mse_per_iteration"]
        )
        flipping_results["mse_final"].append(flipping_solutions["mse_final"])
        flipping_results["computational_time_final"].append(
            flipping_solutions["computational_time_final"]
        )
        flipping_results["weights_num"].append(
            flipping_solutions["weights_num"]
        )
        flipping_results["weights_cat"].append(
            flipping_solutions["weights_cat"]
        )
        flipping_results["bias"].append(flipping_solutions["bias"])

        ## Benchmark attack------------------------------------------------------
        config["iterative_attack_incremental"] = True
        _, _, benchmark_solution = numerical_attack.run(config, instance_data)
        config["iterative_attack_incremental"] = False
        # Benchmark results
        benchmark_results["mse_final"].append(benchmark_solution["mse"])
        benchmark_results["computational_time_final"].append(
            benchmark_solution["computational_time"]
        )
        benchmark_results["weights_num"].append(
            benchmark_solution["weights_num"]
        )
        benchmark_results["weights_cat"].append(
            benchmark_solution["weights_cat"]
        )
        benchmark_results["bias"].append(benchmark_solution["bias"])

        # # # Run binary attack---------------------------------------------------
        # _, _, binary_solutions = binary_attack.run(config, instance_data, model)
        # # Save results to dictionary
        # binary_results["mse_final"].append(binary_solutions["mse_final"])
        # binary_results["computational_time_final"].append(
        #     binary_solutions["computational_time_final"]
        # )
        # binary_results["weights_num"].append(binary_solutions["weights_num"])
        # binary_results["weights_cat"].append(binary_solutions["weights_cat"])
        # binary_results["bias"].append(binary_solutions["bias"])

    # Store numpy savez file
    np.savez(
        f"results/{config['dataset_name']}/{folder_name}/flipping_results.npz",
        **flipping_results,
    )
    np.savez(
        f"results/{config['dataset_name']}/{folder_name}/benchmark_results.npz",
        **benchmark_results,
    )
    np.savez(
        f"results/{config['dataset_name']}/{folder_name}/unpoisoned_results.npz",
        **unpoisoned_results,
    )
    np.savez(
        f"results/{config['dataset_name']}/{folder_name}/binary_results.npz",
        **binary_results,
    )

    # Store config as yaml file
    with open(
        f"results/{config['dataset_name']}/{folder_name}/config.yaml", "w"
    ) as file:
        documents = yaml.dump(config, file)


def choose_regularization(config, instance_data, possible_values):
    """Function to choose regularization parameter
    for ridge regression.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration parameters.
    instance_data : InstanceData
        InstanceData object.
    possible_values : list
        List of possible regularization parameters.

    Returns
    -------
    float
        Best regularization parameter.

    """
    regularization_results = []
    for regularization in possible_values:
        config["regularization"] = regularization
        regression_parameters = ridge_regression.run_not_poisoned(
            config, instance_data
        )
        regularization_results.append(regression_parameters["mse"])

    print(regularization_results)
    return max(regularization_results)
