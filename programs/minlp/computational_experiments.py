import os
import pandas as pd

import flipping_attack
import instance_data_class
import numpy as np
import ridge_regression
import binary_attack
import pyomo_model
import numerical_attack


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
    flipping_results = {}
    benchmark_results = {}
    unpoisoned_results = {}
    binary_results = {}

    flipping_results["mse_per_iteration"] = []
    flipping_results["mse_final"] = []
    flipping_results["computational_time_per_iteration"] = []
    flipping_results["computational_time_final"] = []
    flipping_results["benchmark_improvement"] = []
    flipping_results["ridge_improvement"] = []
    flipping_results["solutions_dict"] = []

    benchmark_results["mse_final"] = []
    benchmark_results["computational_time_final"] = []
    benchmark_results["solutions_dict"] = []

    unpoisoned_results["mse_final"] = []
    unpoisoned_results["solutions_dict"] = []

    binary_results["mse_final"] = []
    binary_results["computational_time_final"] = []
    binary_results["solutions_dict"] = []

    # TODO fix not build model for each run

    for run in range(runs):
        print(f"Run {run+1} of {runs}")
        config["seed"] = run
        instance_data = instance_data_class.InstanceData(config)
        model = pyomo_model.PyomoModel(instance_data, config)
        regression_parameters = ridge_regression.run_not_poisoned(config, instance_data)
        unpoisoned_results["mse_final"].append(regression_parameters["mse"])
        unpoisoned_results["solutions_dict"].append(regression_parameters)
        unpoisoned_results.update({key : value for key, value in regression_parameters.items() if key not in unpoisoned_results})

        # Run flipping attack
        (
            _,
            _,
            solutions,
        ) = flipping_attack.run(config, instance_data, model)
        # Run benchmark attack
        config["iterative_attack_incremental"] = True
        _, benchmark_data, benchmark_solution = numerical_attack.run(
            config, instance_data
        )
        config["iterative_attack_incremental"] = False

        # Save results to dictionary
        # Flipping algorithm results
        flipping_results["mse_per_iteration"].append(solutions["mse_per_iteration"])
        flipping_results["mse_final"].append(solutions["mse_final"])        
        flipping_results["computational_time_per_iteration"].append(
            solutions["computational_time_per_iteration"]
        )
        flipping_results["computational_time_final"].append(
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
        flipping_results["benchmark_improvement"].append(benchmark_improvement)
        flipping_results["ridge_improvement"].append(ridge_improvement)
        flipping_results["solutions_dict"].append(solutions)
        flipping_results.update({key : value for key, value in solutions.items() if key not in flipping_results})
        # Benchmark results
        benchmark_results["mse_final"].append(solutions["benchmark_mse_final"])
        benchmark_results["computational_time_final"].append(
            solutions["benchmark_computational_time"]
        )
        benchmark_results["solutions_dict"].append(benchmark_solution)
        benchmark_results.update({key : value for key, value in benchmark_solution.items() if key not in benchmark_results})

        # # Run binary attack
        # (
        #     _,
        #     _,
        #     binery_solutions,
        # ) = binary_attack.run(config, instance_data, model)

    # Add average MSE to results
    flipping_results["average_mse"] = np.mean(flipping_results["mse_final"])
    flipping_results["average_computational_time"] = np.mean(
        flipping_results["computational_time_final"]
    )
    flipping_results["benchmark_average_improvement"] = np.mean(
        flipping_results["benchmark_improvement"]
    )
    benchmark_results["average_mse"] = np.mean(benchmark_results["mse_final"])
    benchmark_results["average_computational_time"] = np.mean(
        benchmark_results["computational_time_final"]
    )

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


# folder_name = f"{5}_BS{0.5}_TS{100}_PR{20}"
# dictionary = f"results/{'5num5cat'}/{folder_name}"
# flipping_results = np.load(f"{dictionary}/flipping_results.npz")
# benchmark_results = np.load(f"{dictionary}/benchmark_results.npz")
# print(flipping_results["mse_final"])
# print(flipping_results["mse_per_iteration"])
# print(flipping_results["computational_time_final"])
# print(flipping_results["benchmark_improvement"])
# print(flipping_results["ridge_improvement"])
# print(benchmark_results["mse_final"])
# print(benchmark_results["computational_time_final"])
