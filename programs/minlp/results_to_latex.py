"""
Results to latex table
"""

import numpy as np
import instance_data_class
import plots

def main_comparison_table(config):
    """
    Table to compare suvak and our method for both datasets
    """

    header = r"""
    \begin{table}[!htbp]
    \centering
    \captionof{table}{\todo{update}}
    % \begin{adjustbox}{width=\textwidth}
    \begin{tabular}{lrrrrrrrrrrrrrrrrr}
    \toprule
    & & & \multicolumn{2}{c}{Suvak et al. \cite{Suvak2021}} && \multicolumn{2}{c}{IFCF} \\
            \cmidrule{4-5}\cmidrule{7-8}
            \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
    Dataset & Type & $r$ (\%) & MSE & Time (s) && MSE & Time (s) & $\Delta$ (\%)   \\
    \midrule
    """

    print(header)
    time = 0

    for dataset_name in ["House Price", "Healthcare"]:
        for data_type in ["Train", "Test"]:
            for i, poisoning_rate in enumerate([4, 8, 12, 16, 20]):                
                #Training data
                # Folder name
                folder_name = f"{config['runs']}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_PR{str(poisoning_rate)}_lambda{config['regularization']}"
                directory = f"results/{config['dataset_name']}/{folder_name}"
                
                # Load results
                t_flipping_results = np.load(f"{directory}/flipping_results.npz")
                t_benchmark_results = np.load(f"{directory}/benchmark_results.npz")
                increments = (
                    (t_flipping_results["mse_final"]
                    - t_benchmark_results["mse_final"])
                    / t_benchmark_results["mse_final"]
                    * 100
                )
                # Calculate averages mse
                flipping_average = np.mean(t_flipping_results["mse_final"])
                benchmark_average = np.mean(t_benchmark_results["mse_final"])
                increments_average = np.mean(increments)
                # Calculate average time
                flipping_time_average = 0
                benchmark_time_average = 0

                if data_type == "Test":
                    # Load instance data
                    config["poison_rate"] = poisoning_rate
                    config["categorical_attack_no_nfeatures"] = 0
                    config["categorical_attack_no_cfeatures"] = 0
                    instance_data = instance_data_class.InstanceData(config)

                    # Get results
                    flipping_results = np.array([
                            plots.get_test_mse(instance_data, t_flipping_results, run)
                            for run in range(config["runs"])
                        ])
                    benchmark_results = np.array([
                            plots.get_test_mse(instance_data, t_benchmark_results, run)
                            for run in range(config["runs"])
                        ])
                    increments = np.array([
                        (flipping_results - benchmark_results) / benchmark_results * 100
                    ])

                    # Calculate averages
                    flipping_average = np.mean(flipping_results)
                    benchmark_average = np.mean(benchmark_results)
                    increments_average = np.mean(increments)

                # Print results
                if poisoning_rate == 4 and data_type == "Train":
                    print(
                        "\\textit{"
                        + dataset_name
                        + "} & "
                        + data_type
                        + " & 4 & {:6.6f} & {:6.2f} && {:6.6f} & {:6.2f} & {:6.2f} \\\\".format(
                            benchmark_average,
                            benchmark_time_average,
                            flipping_average,
                            flipping_time_average,
                            increments_average,
                        )
                    )
                elif poisoning_rate == 4 and data_type == "Test":
                    print(r"\cmidrule{2-9}")
                    print(
                        "& "
                        + data_type
                        + " & 4 & {:6.6f} & {:6.2f} && {:6.6f} & {:6.2f} & {:6.2f} \\\\".format(
                            benchmark_average,
                            benchmark_time_average,
                            flipping_average,
                            flipping_time_average,
                            increments_average,
                        )
                    )
                else:
                    print(
                        "& & {:2d} & {:6.6f} & {:6.2f} && {:6.6f} & {:6.2f} & {:6.2f} \\\\".format(
                            poisoning_rate,
                            benchmark_average,
                            benchmark_time_average,
                            flipping_average,
                            flipping_time_average,
                            increments_average,
                        )
                    )

        print(r"\midrule")

    footer = r"""
    \bottomrule
    \end{tabular}
    % \end{adjustbox}
    \label{tab: unit sphere 4-10}
    \end{table}
    """
    print(footer)




if __name__ == "__main__":
    config = {
        "runs": 50,
        "numerical_attack_mini_batch_size": 0.5,
        "training_samples": 300,
        "regularization": 0.0001,
        "dataset_name": "5num5cat",
        "seed": 123,
    }
    main_comparison_table(config)
