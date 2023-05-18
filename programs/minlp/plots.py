import matplotlib.pyplot as plt
import numpy as np
import os

config = {
    "runs": 40,
    "numerical_attack_mini_batch_size": 0.5,
    "training_samples": 100,
    "poison_rates": [4, 8, 12, 16, 20],
    "dataset_name": "5num5cat",
    "just_average": False,
}


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
            color="firebrick",
        )
        ax.scatter(
            poisoning_rate,
            benchmark_results["average_mse"],
            marker="o",
            color="steelblue",
        )
        if not config["just_average"]:
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
            ax.legend(["Average Flipping", "Average Benchmark"])
            file_name = "mse_average.pdf"

    ax.set_ylim(ymin=0)

    # Save plot
    fig.savefig(
        f"results/{config['dataset_name']}/plots/{file_name}",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


plot_mse(config)
