import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml

import flipping_attack
import instance_data_class
import numerical_attack
import ridge_regression
import regularization_parameter

sns.set_style("whitegrid")

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# SMALL_SIZE = 14
# MEDIUM_SIZE = 16
# BIGGER_SIZE = 20

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# matplotlib.rcParams["mathtext.fontset"] = "Arial"
# matplotlib.rcParams["font.family"] = "Tahoma"


def plot_mse(config, data_type="train", just_average=True):
    """Plot MSE for each computational experiment and average.
    We want poisoning rates as horizontal axis, and MSE as vertical axis.
    """

    # Create plot to fill later on
    fig, ax = plt.subplots()

    # Add labels
    ax.set_xlabel("Poisoning rate (%)")
    ax.set_ylabel("MSE")

    # Create directory to store results
    isExist = os.path.exists(f"results/{config['dataset_name']}/plots")
    if not isExist:
        os.makedirs(f"results/{config['dataset_name']}/plots")

    # Add data for each poisoning rate
    averages = []
    for i, poisoning_rate in enumerate(config["poison_rates"]):
        folder_name = f"{config['runs']}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_PR{str(poisoning_rate)}_lambda{config['regularization']}"
        directory = f"results/{config['dataset_name']}/{folder_name}"
        flipping_results = np.load(f"{directory}/flipping_results.npz")
        benchmark_results = np.load(f"{directory}/benchmark_results.npz")
        unposioned_results = np.load(f"{directory}/unpoisoned_results.npz")

        if data_type == "train":
            flipping_average = np.mean(flipping_results["mse_final"])
            benchmark_average = np.mean(benchmark_results["mse_final"])
            unpoisoned_average = np.mean(unposioned_results["mse_final"])
            plt.title("Average MSE Comparison (Training data)")
        elif data_type == "test":
            instance_data = instance_data_class.InstanceData(config)
            flipping_average = np.mean(
                [
                    get_test_mse(instance_data, flipping_results, run)
                    for run in range(config["runs"])
                ]
            )
            benchmark_average = np.mean(
                [
                    get_test_mse(instance_data, benchmark_results, run)
                    for run in range(config["runs"])
                ]
            )
            unpoisoned_average = np.mean(
                [
                    get_test_mse(instance_data, unposioned_results, run)
                    for run in range(config["runs"])
                ]
            )
            plt.title("Average MSE Comparison (Validation data)")
        averages.extend(
            (flipping_average, benchmark_average, unpoisoned_average)
        )

        ax.scatter(
            poisoning_rate,
            flipping_average,
            marker="o",
            color="firebrick",
        )
        ax.scatter(
            poisoning_rate,
            benchmark_average,
            marker="o",
            color="steelblue",
        )
        ax.scatter(
            poisoning_rate,
            unpoisoned_average,
            marker="o",
            color="darkolivegreen",
        )
        plt.vlines(
            x=poisoning_rate,
            ymin=benchmark_average,
            ymax=flipping_average,
            color="black",
            linestyle="--",
            linewidth=0.5,
        )
        min_value = np.min([benchmark_average, flipping_average])
        max_value = np.max([benchmark_average, flipping_average])
        ax.text(
            poisoning_rate + 0.2,
            min_value + (max_value - min_value) / 2,
            f"{(flipping_average - benchmark_average) / benchmark_average * 100:.0f}%",
            fontsize=8,
        )
        plt.yticks([int(r) for r in config["poison_rates"]])

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
            ax.legend(["Flipping attack", "Şuvak et al.", "Unpoisoned"])
            file_name = f"{data_type}_{config['runs']}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_lambda{config['regularization']}_mse_average.pdf"

    plt.ylim(0, max(averages) * 1.1)
    # plt.xlim(0, max(config["poison_rates"]) * 1.1)

    # Save plot
    fig.savefig(
        f"results/{config['dataset_name']}/plots/{file_name}",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    plt.show()


def make_predictions(
    data_type: str,
    instance_data: instance_data_class.InstanceData,
    numerical_weights: np.ndarray,
    categorical_weights: np.ndarray,
    bias: float,
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

    weights = np.concatenate((numerical_weights, categorical_weights))

    # Make predictions
    predictions = np.dot(X, weights) + bias

    return predictions


def get_test_mse(instance_data, results, run):
    """
    Get the MSE for a given run.
    """
    y = instance_data.get_y_test_dataframe().to_numpy()
    predictions = make_predictions(
        "test",
        instance_data,
        results["weights_num"][run],
        results["weights_cat"][run],
        results["bias"][run],
    )
    mse = mean_squared_error(y, predictions)
    return mse


def plot_actual_vs_predicted(config, plot_config, data_type: str):
    """
    Plot actual vs predicted values for each instance in the training or test dataset.
    """
    folder_name = f"{plot_config['runs']}_BS{plot_config['numerical_attack_mini_batch_size']}_TS{plot_config['training_samples']}_PR{plot_config['poisoning_rate']}_lambda{plot_config['regularization']}"
    directory = f"results/{plot_config['dataset_name']}/{folder_name}"
    flipping_results = np.load(f"{directory}/flipping_results.npz")
    benchmark_results = np.load(f"{directory}/benchmark_results.npz")
    unposioned_results = np.load(f"{directory}/unpoisoned_results.npz")

    instance_data = instance_data_class.InstanceData(config)

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

    flipping_mse = mean_squared_error(y, flipping_predictions)
    benchmark_mse = mean_squared_error(y, benchmark_predictions)
    nonpoisoned_mse = mean_squared_error(y, unpoisoned_predictions)
    print(
        f"Avg Flipping MSE from model is:   {flipping_results['average_mse']}"
    )
    print(f"Flipping MSE from model is:       {flipping_results['mse_final']}")
    print(f"Flipping MSE from prediction is:  {flipping_mse}")
    print(f"Benchmark MSE from prediction is: {benchmark_mse}")
    print(f"Unpoisoned MSE from prediction is:{nonpoisoned_mse}")

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


def new_plot_mse(config, data_type="train", just_average=True):
    """Plot MSE for each computational experiment and average.
    We want poisoning rates as horizontal axis, and MSE as vertical axis.
    """

    # Create plot to fill later on
    fig, ax = plt.subplots()

    # Add labels
    ax.set_xlabel("Poisoning rate (%)")
    ax.set_ylabel("MSE")

    # Create directory to store results
    isExist = os.path.exists(f"results/plots/{config['dataset']}/{config['dataset_name']}")
    if not isExist:
        os.makedirs(f"results/plots/{config['dataset']}/{config['dataset_name']}")

    # Add data for each poisoning rate
    averages = []
    for i, poisoning_rate in enumerate(config["poison_rates"]):
        poisoning_rate = int(poisoning_rate)
        unposioned_results = []
        benchmark_results = []
        flipping_results = []
        for run in range(config["runs"]):
            folder_name = f"R{run}_PR{poisoning_rate}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_{config['regularization']}"
            directory = f"results/ifcf/{config['dataset']}/{config['dataset_name']}"
            results_dict = np.load(f"{directory}/{folder_name}/results.npz")
            if data_type == "train":
                benchmark_results.append(results_dict["benchmark_mse"])
                flipping_results.append(results_dict["flipping_mse"])
                instance_data = instance_data_class.InstanceData(config, seed=run, thesis=True)
                unpoisoned_results = ridge_regression.run_not_poisoned(
                    config, instance_data, data_type="train"
                )
                unposioned_results.append(unpoisoned_results["mse"])
                plt.title("Average MSE Comparison", y=1.03)
            elif data_type == "test":
                benchmark_results.append(results_dict["benchmark_test_mse"])
                flipping_results.append(results_dict["flipping_test_mse"])
                instance_data = instance_data_class.InstanceData(config, seed=run, thesis=True)
                unpoisoned_results = ridge_regression.run_not_poisoned(
                    config, instance_data, data_type="test"
                )
                unposioned_results.append(unpoisoned_results["mse"])
                plt.title("Average MSE Comparison", y=1.03)

            improvements = [
                (flipping_results[i] - benchmark_results[i]) / benchmark_results[i]
                for i in range(len(flipping_results))
            ]

        flipping_average = np.mean(flipping_results)
        benchmark_average = np.mean(benchmark_results)
        unpoisoned_average = np.mean(unposioned_results)
        def geo_mean(iterable):
            a = np.array(iterable)
            return a.prod()**(1.0/len(a))
        geometric_mean = geo_mean(improvements)

        averages.extend(
            (flipping_average, benchmark_average, unpoisoned_average)
        )

        ax.scatter(
            poisoning_rate,
            flipping_average,
            marker="o",
            color="firebrick",
        )
        ax.scatter(
            poisoning_rate,
            benchmark_average,
            marker="o",
            color="steelblue",
        )
        ax.scatter(
            poisoning_rate,
            unpoisoned_average,
            marker="o",
            color="darkolivegreen",
        )
        plt.vlines(
            x=poisoning_rate,
            ymin=benchmark_average,
            ymax=flipping_average,
            color="black",
            linestyle="--",
            linewidth=0.5,
        )
        min_value = np.min([benchmark_average, flipping_average])
        max_value = np.max([benchmark_average, flipping_average])
        ax.text(
            poisoning_rate + 0.2,
            min_value + (max_value - min_value) / 2,
            str(int(geometric_mean * 100)) + "%",
            fontsize=12,
        )

        #f"{geometric_mean:.0f}%"

        if not just_average:
            ax.scatter(
                [poisoning_rate] * len(flipping_results),
                flipping_results,
                marker="o",
                color="lightcoral",
                alpha=0.4,
                s=5,
            )
            ax.scatter(
                [poisoning_rate] * len(benchmark_results),
                benchmark_results,
                marker="o",
                color="lightskyblue",
                alpha=0.4,
                s=5,
            )

            # Add legend
            ax.legend(
                ["IFCF", "Şuvak et al.", "Unpoisoned"]
            )
            file_name = "mse_all.pdf"
        else:
            ax.legend(["IFCF", "Şuvak et al.", "Unpoisoned"])
    
    file_name = f"{data_type}_{config['runs']}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_{config['regularization']}"

    plt.ylim(0, max(averages) * 1.1)
    plt.xlim(2,22)
    plt.xticks([int(r) for r in config["poison_rates"]])

    # Save plot
    fig.savefig(
        f"results/plots/{config['dataset']}/{config['dataset_name']}/{file_name}.pdf",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    fig.savefig(
        f"results/plots/{config['dataset']}/{config['dataset_name']}/{file_name}.png",
        bbox_inches="tight",
        dpi=300,
        transparent=False
    )
    plt.show()


def batch_size_IAS_vs_SAS(config, cross_validation=False):
    """
    Plot with batch size on x-axis and MSE on y-axis.
    We compare all poisoning rates, each rate as a different color.
    IAS and SAS are diffent line styles.
    """
    colors = sns.color_palette("husl", len(config["poison_rates"]))
    batch_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, poisoning_rate in enumerate(config["poison_rates"]):
        config["poison_rate"] = poisoning_rate
        MSE_IAS = []
        MSE_SAS = []
        if cross_validation:
                    instance_data = instance_data_class.InstanceData(
                        config=config, benchmark_data=False, thesis=True, seed=config["run"]
                    )
                    config["regularization"] = regularization_parameter.cross_validation_lambda(
                        instance_data, np.linspace(0.001, 10, 20)
                    )["alpha"]

        for j, batch_size in enumerate(batch_sizes):
            config["numerical_attack_mini_batch_size"] = batch_size
            folder_name = f"{config['run']}_PR{poisoning_rate}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_{config['regularization']}"
            
            # Iterative attack strategy results (IAS)------------------------------
            ias_directory = f"results/IAS/{config['dataset']}/{config['dataset_name']}"
            if not os.path.exists(os.path.join(ias_directory, folder_name)):
                os.makedirs(os.path.join(ias_directory, folder_name))
            # Check if results exist
            if not os.path.exists(f"{ias_directory}/{folder_name}/results.npz"):
                # Run experiments and store
                print("Running IAS experiments for poisoning rate ", poisoning_rate)
                config["numerical_attack_incremental"] = True
                instance_data = instance_data_class.InstanceData(
                    config, benchmark_data=False, seed=config["seed"], thesis=True
                )
                _, _, ias_results = numerical_attack.run(config, instance_data)
                # Save results as npz
                np.savez(f"{ias_directory}/{folder_name}/results.npz", **ias_results)
            else:
                ias_results = np.load(f"{ias_directory}/{folder_name}/results.npz")
            MSE_IAS.append(ias_results["mse"])

            # Shift attack strategy results (SAS)----------------------------------
            sas_directory = f"results/SAS/{config['dataset']}/{config['dataset_name']}"
            if not os.path.exists(os.path.join(sas_directory, folder_name)):
                os.makedirs(os.path.join(sas_directory, folder_name))
            # Check if results exist
            if not os.path.exists(f"{sas_directory}/{folder_name}/results.npz"):
                # Run experiments and store
                print("Running SAS experiments for poisoning rate ", poisoning_rate)
                config["numerical_attack_incremental"] = False
                instance_data = instance_data_class.InstanceData(
                    config, benchmark_data=False, seed=config["seed"], thesis=True
                )
                _, _, sas_results = numerical_attack.run(config, instance_data)
                # Save results as npz
                np.savez(f"{sas_directory}/{folder_name}/results.npz", **sas_results)
            else:
                sas_results = np.load(f"{sas_directory}/{folder_name}/results.npz")
            MSE_SAS.append(sas_results["mse"])
            print("SAS results poisioning rate ", poisoning_rate, " MSE ", sas_results["mse"])

        ax.plot(batch_sizes, MSE_IAS, label="IAS", color=colors[i], linestyle="--")
        ax.plot(batch_sizes, MSE_SAS, label="SAS", color=colors[i])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("MSE")

    # plt.title("Batch size IAS vs SAS  λ = {:.4f}".format(config["regularization"]), fontsize=28)
    plt.title("Batch size IAS vs SAS", fontsize=28)
    legend_lines = [matplotlib.lines.Line2D([0], [0], color="black", linestyle="--", lw=2),
                    matplotlib.lines.Line2D([0], [0], color="black", lw=2)] 
    legend_colors = [matplotlib.lines.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(config["poison_rates"]))]
    ax.add_artist(ax.legend(legend_lines, ["IAS", "SAS"], title="Attack strategy", loc="lower right"))
    ax.add_artist(ax.legend(legend_colors, [f"r={pr}" for pr in config["poison_rates"]], title="Poisoning rate", loc="upper right"))


    # Save plot
    fig.savefig(
        f"results/plots/{config['dataset']}/{config['dataset_name']}/{config['run']}_CV_{cross_validation}_batch_IAS_vs_SAS.pdf",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )

    fig.savefig(
        f"results/plots/{config['dataset']}/{config['dataset_name']}/{config['run']}_CV_{cross_validation}_batch_IAS_vs_SAS.png",
        bbox_inches="tight",
        dpi=300,
        transparent=False
    )


def hyperparameter_IAS_vs_SAS(config):
    """
    Plot with hyperparameter on x-axis and MSE on y-axis.
    We compare all poisoning rates, each rate as a different color.
    IAS and SAS are diffent line styles.
    """
    colors = sns.color_palette("husl", len(config["poison_rates"]))
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, poisoning_rate in enumerate(config["poison_rates"]):
        config["poison_rate"] = poisoning_rate
        MSE_IAS = []
        MSE_SAS = []
        for hyperparameter in [0.001, 0.01, 0.1, 1, 10]:
            config["regularization"] = hyperparameter
            folder_name = f"PR{poisoning_rate}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_{config['regularization']}"
            
            # Iterative attack strategy results (IAS)------------------------------
            ias_directory = f"results/IAS/{config['dataset']}/{config['dataset_name']}"
            if not os.path.exists(os.path.join(ias_directory, folder_name)):
                os.makedirs(os.path.join(ias_directory, folder_name))
            # Check if results exist
            if not os.path.exists(f"{ias_directory}/{folder_name}/results.npz"):
                # Run experiments and store
                print("Running IAS experiments for poisoning rate ", poisoning_rate)
                config["numerical_attack_incremental"] = True
                instance_data = instance_data_class.InstanceData(
                    config, benchmark_data=False, seed=config["seed"], thesis=True
                )
                _, _, ias_results = numerical_attack.run(config, instance_data)
                # Save results as npz
                np.savez(f"{ias_directory}/{folder_name}/results.npz", **ias_results)
            else:
                ias_results = np.load(f"{ias_directory}/{folder_name}/results.npz")
            MSE_IAS.append(ias_results["mse"])
            # print("IAS results poisioning rate ", config["poison_rate"], " MSE ", ias_results["mse"])

            # Shift attack strategy results (SAS)----------------------------------
            sas_directory = f"results/SAS/{config['dataset']}/{config['dataset_name']}"
            if not os.path.exists(os.path.join(sas_directory, folder_name)):
                os.makedirs(os.path.join(sas_directory, folder_name))
            # Check if results exist
            if not os.path.exists(f"{sas_directory}/{folder_name}/results.npz"):
                # Run experiments and store
                print("Running SAS experiments for poisoning rate ", poisoning_rate)
                config["numerical_attack_incremental"] = False
                instance_data = instance_data_class.InstanceData(
                    config, benchmark_data=False, seed=config["seed"], thesis=True
                )
                _, _, sas_results = numerical_attack.run(config, instance_data)
                # Save results as npz
                np.savez(f"{sas_directory}/{folder_name}/results.npz", **sas_results)
            else:
                sas_results = np.load(f"{sas_directory}/{folder_name}/results.npz")
            MSE_SAS.append(sas_results["mse"])
            # print("SAS results poisioning rate ", poisoning_rate, " MSE ", sas_results["mse"])

        ax.plot([0.001, 0.01, 0.1, 1, 10], MSE_IAS, label="IAS", color=colors[i], linestyle="--")
        ax.plot([0.001, 0.01, 0.1, 1, 10], MSE_SAS, label="SAS", color=colors[i])
        ax.set_xscale("log")
        ax.set_xlabel("Regularization hyperparameter", fontsize=20)
        ax.set_ylabel("MSE", fontsize=20)
        plt.title("Hyperparameter IAS vs SAS", fontsize=28)
        
    legend_lines = [matplotlib.lines.Line2D([0], [0], color="black", linestyle="--", lw=2),
                    matplotlib.lines.Line2D([0], [0], color="black", lw=2)] 
    legend_colors = [matplotlib.lines.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(config["poison_rates"]))]
    ax.add_artist(ax.legend(legend_lines, ["IAS", "SAS"], title="Attack strategy", loc="lower right", fontsize=18))
    ax.add_artist(ax.legend(legend_colors, [f"{pr}%" for pr in config["poison_rates"]], title="Poisoning rate", loc="upper right", fontsize=18))

    # Fontsize and ticks
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    # Save plot
    fig.savefig(
        f"results/plots/{config['dataset']}/{config['dataset_name']}/hyperparameter_IAS_vs_SAS.pdf",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )

    fig.savefig(
        f"results/plots/{config['dataset']}/{config['dataset_name']}/hyperparameter_IAS_vs_SAS.png",
        bbox_inches="tight",
        dpi=300,
        transparent=False
    )


if __name__ == "__main__":
    with open("programs/minlp/config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # plot_mse(config, "test")

    config["dataset"] = "house"
    config["regularization"] = 0.1 
    config["runs"] = 20
    config["dataset_name"] = "allnum10cat"
    config["numerical_attack_mini_batch_size"] = 0.1
    new_plot_mse(config, just_average=False, data_type="train")
    # hyperparameter_IAS_vs_SAS(config)
    config["run"] = 2
    # batch_size_IAS_vs_SAS(config, cross_validation=False)
    # batch_size_IAS_vs_SAS(config, cross_validation=True)
