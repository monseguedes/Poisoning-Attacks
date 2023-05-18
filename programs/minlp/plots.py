
import matplotlib.pyplot as plt
import numpy as np
import os

import os

import matplotlib
import matplotlib.pyplot as plt
import instance_data_class
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import flipping_attack

sns.set_style("whitegrid")

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

config = {"runs": 40,
          "numerical_attack_mini_batch_size": 0.5,
          "training_samples": 100,
          "poison_rates": [4,8,12,16,20],
          "dataset_name": "5num5cat",
          "just_average": False,
          "datatype": "train"}

def plot_mse(config):
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
        dictionary = f"results/{config['dataset_name']}/{folder_name}"
        flipping_results = np.load(f"{dictionary}/flipping_results.npz")
        benchmark_results = np.load(f"{dictionary}/benchmark_results.npz")
        ax.scatter(
            poisoning_rate,
            flipping_results["average_mse"],
            marker="o",
            color="firebrick"
        )
        ax.scatter(
            poisoning_rate,
            benchmark_results["average_mse"],
            marker="o",
            color="steelblue"
        )
        if not config["just_average"]:
            ax.scatter(
                [poisoning_rate] * len(flipping_results["mse_final"]),
                flipping_results["mse_final"],
                marker="o",
                color="lightcoral",
                alpha=.3,
                s=5
            )
            ax.scatter(
                [poisoning_rate] * len(benchmark_results["mse_final"]),
                benchmark_results["mse_final"],
                marker="o",
                color="lightskyblue",
                alpha=.3,
                s=5
            )

            # Add legend
            ax.legend(["Average Flipping", "Average Benchmark",  "All Flipping", "All Benchmark"])
            file_name = 'mse_all.pdf'
        else:
            ax.legend(["Average Flipping", "Average Benchmark"])
            file_name = 'mse_average.pdf'
    
    ax.set_ylim(ymin=0)

    # Save plot
    fig.savefig(
        f"results/{config['dataset_name']}/plots/{file_name}",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

def make_predictions(data_type: str, instance_data: instance_data_class.InstanceData, solution_dict):
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

    numerical_weights = solution_dict["numerical_weights"].values
    categorical_weights = solution_dict["categorical_weights"].values
    weights = np.concatenate((numerical_weights, categorical_weights))

    # Make predictions
    predictions = np.dot(X, weights)

    return predictions   

def plot_actual_vs_predicted(data_type: str, instance_data: instance_data_class.InstanceData, solution_dict):
    """
    Plot actual vs predicted values for each instance in the training or test dataset.
    """

    # Make predictions
    predictions = make_predictions(data_type, instance_data, solution_dict)

    # Get actual values
    if data_type == "test":
        y = instance_data.get_y_test_dataframe.to_numpy()
    if data_type == "train":
        y = instance_data.get_y_train_dataframe.to_numpy()

    # Create plot to fill later on
    fig, ax = plt.subplots()

    # Add labels
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    plt.title("Actual vs Predicted values")

    # Add data
    ax.scatter(
        y,
        predictions,
        marker="o",
        color="firebrick"
    )

    # Save plot
    fig.savefig(
        f"results/{config['dataset_name']}/plots/actual_vs_predicted_{data_type}.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


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
    "datatype": "test"
}
print(type(config))
instance_data = instance_data_class.InstanceData(config)
_,_,solution_dict = flipping_attack.run(config, instance_data)
plot_actual_vs_predicted("train", instance_data, solution_dict)





