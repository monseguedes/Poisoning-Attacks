
import matplotlib.pyplot as plt
import numpy as np

config = {"runs": 5,
          "numerical_attack_mini_batch_size": 0.5,
          "training_samples": 100,
          "poison_rates": [4,8,12,16,20]}

def plot_mse(config):
    """Plot MSE for each computational experiment and average.
    We want poisoning rates as horizontal axis, and MSE as vertical axis.
    """

    # Create plot to fill later on
    fig, ax = plt.subplots()

    # Add data for each poisoning rate
    for i, poisoning_rate in enumerate(config["poison_rates"]):
        folder_name = f"{config['runs']}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_PR{str(poisoning_rate)}"
        dictionary = f"results/{'5num5cat'}/{folder_name}"
        flipping_results = np.load(f"{dictionary}/flipping_results.npz")
        benchmark_results = np.load(f"{dictionary}/benchmark_results.npz")
        ax.scatter(
            poisoning_rate,
            flipping_results["average_mse"],
            marker="x",
            color="darkgreen"
        )
        ax.scatter(
            [poisoning_rate] * len(flipping_results["mse_final"]),
            flipping_results["mse_final"],
            marker="o",
            color="lightgreen",
            s=5
        )
        ax.scatter(
            poisoning_rate,
            benchmark_results["average_mse"],
            marker="x",
            color="darkred"
        )
        ax.scatter(
            [poisoning_rate] * len(benchmark_results["mse_final"]),
            benchmark_results["mse_final"],
            marker="o",
            color="red",
            s=5
        )

    # Add legend
    ax.legend(["Average Flipping", "All Flipping", "Benchmark", "All Benchmark"])

    # Add labels
    ax.set_xlabel("Poisoning rate (%)")
    ax.set_ylabel("MSE")
    plt.title("MSE for different poisoning rates")

    plt.show()

    # # Save plot
    # fig.savefig(
    #     f"results/{config['dataset_name']}/plots/mse.pdf",
    #     bbox_inches="tight",
    #     dpi=300,
    # )

plot_mse(config)










