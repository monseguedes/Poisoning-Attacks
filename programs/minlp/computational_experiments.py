import os
import pandas as pd

import flipping_attack
import instance_data_class
import numpy as np
import ridge_regression


def run(runs, config):
    """Function to run computational experiment
    and save results to csv files.

    Parameters
    ----------
    runs : int
        Number of runs to perform.

    Returns
    -------
    None.

    """

    folder_name = f"{runs}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_PR{config['poison_rate']}"
    # Create directory to store results
    isExist = os.path.exists(f"results/{config['dataset_name']}/{folder_name}")
    if not isExist:
        os.makedirs(f"results/{config['dataset_name']}/{folder_name}")

    # Create dictionary to store results
    results = {}
    benchmark_results = {}

    results["mse_per_iteration"] = []
    results["mse_final"] = []
    results["computational_time_per_iteration"] = []
    results["computational_time_final"] = []
    results["benchmark_improvement"] = []
    results["ridge_improvement"] = []
    benchmark_results["benchmark_mse_final"] = []
    benchmark_results["benchmark_computational_time"] = []

    for run in range(runs):
        config["seed"] = run
        instance_data = instance_data_class.InstanceData(config)
        regression_parameters = ridge_regression.run_just_training(
            instance_data, config
        )
        model = None

        # Run flipping attack
        (
            _,
            instance_data,
            solutions,
        ) = flipping_attack.run(config, instance_data, model)

        results["mse_per_iteration"].append(solutions["mse_per_iteration"])
        results["mse_final"].append(solutions["mse_final"])
        results["computational_time_per_iteration"].append(
            solutions["computational_time_per_iteration"]
        )
        results["computational_time_final"].append(
            solutions["computational_time_final"]
        )

        benchmark_improvement = (
            (solutions["mse"] - solutions["benchmark_mse_final"])
            / solutions["benchmark_mse_final"]
            * 100
        )
        ridge_improvement = (
            (solutions["mse"] - regression_parameters["mse"])
            / regression_parameters["mse"]
            * 100
        )

        results["benchmark_improvement"].append(benchmark_improvement)
        results["ridge_improvement"].append(solutions["ridge_improvement"])

        benchmark_results["benchmark_mse_final"].append(
            solutions["benchmark_mse_final"]
        )
        benchmark_results["benchmark_computational_time"].append(
            solutions["benchmark_computational_time"]
        )

    # Add average MSE to results
    results["average_mse"] = np.mean(results["mse_final"])
    results["average_computational_time"] = np.mean(results["computational_time_final"])
    results["benchmark_average_improvement"] = np.mean(results["benchmark_improvement"])
    benchmark_results["average_mse"] = np.mean(benchmark_results["mse_final"])
    benchmark_results["average_computational_time"] = np.mean(
        benchmark_results["benchmark_computational_time"]
    )
