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
    config["return_benchmark"] = True
    folder_name = f"{runs}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_PR{config['poison_rate']}"
    # Create directory to store results
    isExist = os.path.exists(f"results/{config['dataset_name']}/{folder_name}")
    if not isExist:
        os.makedirs(f"results/{config['dataset_name']}/{folder_name}")

    # Create dictionary to store results
    results = {}
    benchmark_results = {}
    for run in range(runs):
        config["seed"] = run
        instance_data = instance_data_class.InstanceData(config)
        model = None
        # Run flipping attack
        (
            _,
            instance_data,
            regression_parameters,
            benchmark_solution,
        ) = flipping_attack.run(config, instance_data, model)
        improvement = (
            (regression_parameters["mse"] - benchmark_solution["mse"])
            / benchmark_solution["mse"]
            * 100
        )
        results[run] = {"mse": regression_parameters["mse"], "improvement": improvement}
        benchmark_results[run] = {"mse": benchmark_solution["mse"]}

    # Add average MSE to results
    results["average"] = np.mean([results[run]["mse"] for run in range(runs)])
    results["average_improvement"] = np.mean(
        [results[run]["improvement"] for run in range(runs)]
    )
    benchmark_results["average"] = np.mean(
        [benchmark_results[run]["mse"] for run in range(runs)]
    )
    # Create dataframe from results dictionary
    results_dataframe = pd.DataFrame.from_dict(results)
    print(results_dataframe)
    benchmark_results_dataframe = pd.DataFrame.from_dict(benchmark_results)
    # Save results to csv file
    results_dataframe.to_csv(
        f"results/{config['dataset_name']}/{folder_name}/results.csv"
    )
    benchmark_results_dataframe.to_csv(
        f"results/{config['dataset_name']}/{folder_name}/benchmark_results.csv"
    )
