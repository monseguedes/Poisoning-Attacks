import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

import flipping_attack
import instance_data_class
import numerical_attack
import ridge_regression

sns.set_style("whitegrid")

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_mse(config, just_average=True):
    """Plot MSE for each computational experiment and average.
    We want poisoning rates as horizontal axis, and MSE as vertical axis.
    """

    # Create plot to fill later on
    fig, ax = plt.subplots()

    # Add labels
    ax.set_xlabel("Poisoning rate (%)")
    ax.set_ylabel("MSE")
    plt.title("MSE for different poisoning rates")

    # Create directory to store results
    isExist = os.path.exists(f"results/{config['dataset_name']}/plots")
    if not isExist:
        os.makedirs(f"results/{config['dataset_name']}/plots")

    # Add data for each poisoning rate
    for i, poisoning_rate in enumerate(config["poison_rates"]):
        folder_name = f"{config['runs']}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_PR{str(poisoning_rate)}"
        directory = f"results/{config['dataset_name']}/{folder_name}"
        flipping_results = np.load(f"{directory}/flipping_results.npz")
        benchmark_results = np.load(f"{directory}/benchmark_results.npz")
        unposioned_results = np.load(f"{directory}/unpoisoned_results.npz")
        ax.scatter(
            poisoning_rate,
            flipping_results["average_mse"],
            marker="o",
            color="firebrick",
        )
        ax.scatter(
            poisoning_rate,
            benchmark_results["average_mse"],
            marker="o",
            color="steelblue",
        )
        ax.scatter(
            poisoning_rate,
            np.mean(unposioned_results["mse_final"]),
            marker="o",
            color="darkolivegreen",
        )

        if not just_average:
            ax.scatter(
                [poisoning_rate] * len(flipping_results["mse_final"]),
                flipping_results["mse_final"],
                marker="o",
                color="lightcoral",
                alpha=0.3,
                s=5,
            )
            ax.scatter(
                [poisoning_rate] * len(benchmark_results["mse_final"]),
                benchmark_results["mse_final"],
                marker="o",
                color="lightskyblue",
                alpha=0.3,
                s=5,
            )

            # Add legend
            ax.legend(
                [
                    "Average Flipping",
                    "Average Benchmark",
                    "All Flipping",
                    "All Benchmark",
                ]
            )
            file_name = "mse_all.pdf"
        else:
            ax.legend(["Average Flipping", "Average Benchmark", "Average Unpoisoned"])
            file_name = "mse_average.pdf"

    plt.ylim(0, flipping_results["average_mse"] * 1.1)

    # Save plot
    fig.savefig(
        f"results/{config['dataset_name']}/plots/{file_name}",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def make_predictions(
    data_type: str, instance_data: instance_data_class.InstanceData, solution_dict
):
    """
    Take the regression coefficents given by solving the nonpoisoned model
    and use them to make predictions on training dataset.
    """

    if data_type == "test":
        X_cat = instance_data.get_cat_x_test_dataframe(wide=True).to_numpy()
        X_num = instance_data.get_num_x_test_dataframe(wide=True).to_numpy()
        X = np.concatenate((X_num, X_cat), axis=1)

    if data_type == "train":
        X_cat = instance_data.get_cat_x_train_dataframe(wide=True).to_numpy()
        X_num = instance_data.get_num_x_train_dataframe(wide=True).to_numpy()
        X = np.concatenate((X_num, X_cat), axis=1)

    print(solution_dict["weights_cat"])
    numerical_weights = solution_dict["weights_num"]
    categorical_weights = solution_dict["weights_cat"]
    weights = np.concatenate((numerical_weights, categorical_weights))

    # Make predictions
    predictions = np.dot(X, weights)

    return predictions


def plot_actual_vs_predicted(config, plot_config, data_type: str):
    """
    Plot actual vs predicted values for each instance in the training or test dataset.
    """
    folder_name = f"{plot_config['runs']}_BS{plot_config['numerical_attack_mini_batch_size']}_TS{plot_config['training_samples']}_PR{plot_config['poisoning_rate']}"
    directory = f"results/{plot_config['dataset_name']}/{folder_name}"
    flipping_results = np.load(f"{directory}/flipping_results.npz")
    benchmark_results = np.load(f"{directory}/benchmark_results.npz")
    unposioned_results = np.load(f"{directory}/unpoisoned_results.npz")

    instance_data = instance_data_class.InstanceData(config)

    print(list(flipping_results.keys()))
    print(flipping_results["average_mse"])
    print(flipping_results["mse_final"])

    # Make predictions
    flipping_predictions = make_predictions(
        data_type, instance_data, flipping_results
    )
    benchmark_predictions = make_predictions(
        data_type, instance_data, benchmark_results
    )
    unpoisoned_predictions = make_predictions(
        data_type, instance_data, unposioned_results
    )

    # Get actual values
    if data_type == "test":
        y = instance_data.get_y_test_dataframe().to_numpy()
    if data_type == "train":
        y = instance_data.get_y_train_dataframe().to_numpy()

    # Create plot to fill later on
    fig, ax = plt.subplots()

    # Add labels
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    plt.title("Actual vs Predicted values")

    # Add data
    ax.scatter(y, flipping_predictions, marker="o", color="firebrick")
    ax.scatter(y, benchmark_predictions, marker="o", color="steelblue")
    ax.scatter(y, unpoisoned_predictions, marker="o", color="darkolivegreen")

    # Add legend
    ax.legend(["Flipping", "Benchmark", "Unpoisoned"])

    # Save plot
    fig.savefig(
        f"results/{plot_config['dataset_name']}/plots/actual_vs_predicted_{data_type}.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()
