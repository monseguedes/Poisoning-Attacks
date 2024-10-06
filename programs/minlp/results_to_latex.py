"""
Results to latex table
"""

import numpy as np
import instance_data_class
import plots
import os
import numerical_attack
import flipping_attack
import yaml # type: ignore
import regularization_parameter

def geo_mean(iterable):
            a = np.array(iterable)
            return a.prod()**(1.0/len(a))

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
                        dataset_name
                        + " & "
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
                        + " & 4 & {:6.6f} & {} && {:6.6f} & {} & {:6.2f} \\\\".format(
                            benchmark_average,
                            "-",
                            flipping_average,
                            "-",
                            increments_average,
                        )
                    )
                elif poisoning_rate > 4 and data_type == "Test":
                    print(
                        "& & {:2d} & {:6.6f} & {} && {:6.6f} & {} & {:6.2f} \\\\".format(
                            poisoning_rate,
                            benchmark_average,
                            "-",
                            flipping_average,
                            "-",
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

def SAS_vs_IAS_table(config, cross_validation=False):
    """
    \begin{table}[!htbp]
    \centering
    \captionof{table}{Comparison of the iterative attack strategy (IAS) and the shifting attack strategy (SAS). \todo{update}}\label{tab: SAS vs IAS}
    \begin{tabular}{lrrrrr}
    \toprule
    Dataset & $r$(\%) & MSE$_{\textup{IAS}}$ & MSE$_{\textup{SAS}}$ &  $\Delta$\textbf{(\%)}\\
    \midrule
    House Price & 4 & 0.00676 & 0.00679 & 0.4308 \\
    & 8  & 0.00623 & 0.00653 & 4.7404 \\
    & 12  & 0.00618 & 0.00626 & 1.3952 \\
    & 16  & 0.00623 & 0.00653 & 4.7404 \\
    & 20  & 0.00623 & 0.00653 & 4.7404 \\
    \midrule
    Healthcare & 4 & 0.00676 & 0.00679 & 0.4308 \\
    & 8  & 0.00623 & 0.00653 & 4.7404 \\
    & 12  & 0.00618 & 0.00626 & 1.3952 \\
    & 16  & 0.00623 & 0.00653 & 4.7404 \\
    & 20  & 0.00623 & 0.00653 & 4.7404 \\
    \bottomrule
    \end{tabular}
    \end{table}
    """

    header = r"""
    \begin{table}[!htbp]
    \centering
    \captionof{table}{Comparison of the iterative attack strategy (IAS) and the shifting attack strategy (SAS). \todo{update}}\label{tab: SAS vs IAS}
    \begin{tabular}{lrrrrr}
    \toprule
    Dataset & $r$(\%) & MSE$_{\textup{IAS}}$ & MSE$_{\textup{SAS}}$ &  $\Delta$\textbf{(\%)}\\
    \midrule
    """

    print(header)

    for dataset_name in ["house", "pharm"]:
        if dataset_name == "pharm":
            print(r"\midrule")
            config["dataset"] = "pharm"
        for i, poisoning_rate in enumerate([4, 8, 12, 16, 20]):
            config["poison_rate"] = poisoning_rate
            if cross_validation:
                        instance_data = instance_data_class.InstanceData(
                            config=config, benchmark_data=False, thesis=True
                        )
                        config["regularization"] = regularization_parameter.cross_validation_lambda(
                            instance_data, np.linspace(0.001, 10, 20)
                        )["alpha"]
            
            IAS_mse = []
            SAS_mse = []
            increments = []
            for run in range(config["runs"]):
                config["seed"] = run
                # Folder name of this experiment and rate
                folder_name = f"{run}_PR{poisoning_rate}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_{config['regularization']}"
                
                # Iterative attack strategy results (IAS)------------------------------
                ias_directory = f"results/IAS/{dataset_name}/{config['dataset_name']}"
                if not os.path.exists(os.path.join(ias_directory, folder_name)):
                    os.makedirs(os.path.join(ias_directory, folder_name))
                # Check if results exist
                if not os.path.exists(f"{ias_directory}/{folder_name}/results.npz"):
                    # Run experiments and store
                    print("Running IAS experiments for poisoning rate ", poisoning_rate)
                    config["numerical_attack_incremental"] = True
                    instance_data = instance_data_class.InstanceData(
                        config, benchmark_data=False, seed=run, thesis=True
                    )
                    _, _, ias_results = numerical_attack.run(config, instance_data)
                    # Save results as npz
                    np.savez(f"{ias_directory}/{folder_name}/results.npz", **ias_results)
                else:
                    ias_results = np.load(f"{ias_directory}/{folder_name}/results.npz")
                IAS_mse.append(ias_results["mse"])

                # Shift attack strategy results (SAS)----------------------------------
                sas_directory = f"results/SAS/{dataset_name}/{config['dataset_name']}"
                if not os.path.exists(os.path.join(sas_directory, folder_name)):
                    os.makedirs(os.path.join(sas_directory, folder_name))
                # Check if results exist
                if not os.path.exists(f"{sas_directory}/{folder_name}/results.npz"):
                    # Run experiments and store
                    print("Running SAS experiments for poisoning rate ", poisoning_rate)
                    config["numerical_attack_incremental"] = False
                    instance_data = instance_data_class.InstanceData(
                        config, benchmark_data=False, seed=run, thesis=True
                    )
                    _, _, sas_results = numerical_attack.run(config, instance_data)
                    # Save results as npz
                    np.savez(f"{sas_directory}/{folder_name}/results.npz", **sas_results)
                else:
                    sas_results = np.load(f"{sas_directory}/{folder_name}/results.npz")
                SAS_mse.append(sas_results["mse"])
                increments.append((sas_results["mse"] - ias_results["mse"]) / ias_results["mse"] * 100)

            # Calculate averages
            ias_average = np.mean(IAS_mse)
            sas_average = np.mean(SAS_mse)
            geo_increments = geo_mean(increments)

            if poisoning_rate == 4:
                print(dataset_name 
                    + " & {} & {:.4f} & {:.4f} & {:.2f} \\\\".format(
                        poisoning_rate,
                        ias_average,
                        sas_average,
                        geo_increments,
                    )
                )
            
            else:
                print(
                    "& {} & {:.4f} & {:.4f} & {:.2f} \\\\".format(
                        poisoning_rate,
                        ias_average,
                        sas_average,
                        geo_increments,
                        )
                )

    footer = r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """
    print(footer)

def all_SAS_vs_IAS_table(config, cross_validation=False):
    """
    \begin{table}[!htbp]
    \centering
    \captionof{table}{Comparison of the iterative attack strategy (IAS) and the shifting attack strategy (SAS) for all the numerical features and 5 categorical features.}\label{tab: SAS vs IAS}
    \begin{adjustbox}{width=\textwidth}
    \begin{tabular}{lrrrrrrrrrrrrr}
    \toprule
    & & \multicolumn{3}{c}{All numerical 5 categorical} && \multicolumn{3}{c}{All numerical 10 categorical} && \multicolumn{3}{c}{All numerical 10 categorical} \\
    \cmidrule{3-5}\cmidrule{7-9}\cmidrule{11-13}
    \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
    Dataset & $r$(\%) & MSE$_{\textup{IAS}}$ & MSE$_{\textup{SAS}}$ &  $\Delta$\textbf{(\%)} && MSE$_{\textup{IAS}}$ & MSE$_{\textup{SAS}}$ &  $\Delta$\textbf{(\%)} && MSE$_{\textup{IAS}}$ & MSE$_{\textup{SAS}}$ &  $\Delta$\textbf{(\%)}\\
    \midrule
    House Price & 4 & 0.0078 & 0.0123 & 0.59 \\
    & 8 & 0.0153 & 0.0203 & 0.33 \\
    & 12 & 0.0164 & 0.0264 & 0.61 \\
    & 16 & 0.0221 & 0.0326 & 0.48 \\
    & 20 & 0.0269 & 0.0388 & 0.44 \\
    \midrule
    Healthcare & 4 & 0.0057 & 0.0096 & 0.69 \\
    & 8 & 0.0102 & 0.0175 & 0.71 \\
    & 12 & 0.0150 & 0.0237 & 0.58 \\
    & 16 & 0.0207 & 0.0356 & 0.72 \\
    & 20 & 0.0277 & 0.0439 & 0.58 \\
    \bottomrule
    \end{tabular}
    \end{adjustbox}
    \end{table}
    """

    header = r"""
    \begin{table}[!htbp]
    \centering
    \captionof{table}{Comparison of the iterative attack strategy (IAS) and the shifting attack strategy (SAS) for all the numerical features and 5 categorical features.}\label{tab: SAS vs IAS}
    \begin{adjustbox}{width=\textwidth}
    \begin{tabular}{lrrrrrrrrrrrrr}
    \toprule
    & & \multicolumn{3}{c}{All numerical 5 categorical} && \multicolumn{3}{c}{All numerical 10 categorical} && \multicolumn{3}{c}{All numerical all categorical} \\
    \cmidrule{3-5}\cmidrule{7-9}\cmidrule{11-13}
    \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
    Dataset & $r$(\%) & MSE$_{\textup{IAS}}$ & MSE$_{\textup{SAS}}$ &  $\Delta$\textbf{(\%)} && MSE$_{\textup{IAS}}$ & MSE$_{\textup{SAS}}$ &  $\Delta$\textbf{(\%)} && MSE$_{\textup{IAS}}$ & MSE$_{\textup{SAS}}$ &  $\Delta$\textbf{(\%)}\\
    \midrule
    """

    print(header)

    for dataset_name in ["house", "pharm"]:
        if dataset_name == "pharm":
            print(r"\midrule")
        for i, poisoning_rate in enumerate([4, 8, 12, 16, 20]):
            config["poison_rate"] = poisoning_rate
            results = {}
            for dataset in ["allnum5cat", "allnum10cat", "allnumallcat"]:
                if cross_validation:
                        instance_data = instance_data_class.InstanceData(
                            config=config, benchmark_data=False, thesis=True
                        )
                        config["regularization"] = regularization_parameter.cross_validation_lambda(
                            instance_data, np.linspace(0.001, 10, 20)
                        )["alpha"]
                SAS_mse = []
                IAS_mse = []
                increments = []
                for run in range(config["runs"]):
                    # Folder name of this experiment and rate
                    folder_name = f"{run}_PR{poisoning_rate}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_{config['regularization']}"
                    
                    # Iterative attack strategy results (IAS)------------------------------
                    ias_directory = f"results/IAS/{dataset_name}/{dataset}"
                    ias_results = np.load(f"{ias_directory}/{folder_name}/results.npz")
                    IAS_mse.append(ias_results["mse"])

                    # Shift attack strategy results (SAS)----------------------------------
                    sas_directory = f"results/SAS/{dataset_name}/{dataset}"
                    sas_results = np.load(f"{sas_directory}/{folder_name}/results.npz")
                    SAS_mse.append(sas_results["mse"])
                    increments.append((sas_results["mse"] - ias_results["mse"]) / ias_results["mse"] * 100)

                # Calculate averages
                ias_average = np.mean(IAS_mse)
                sas_average = np.mean(SAS_mse)
                geo_increments = geo_mean(increments)

                results[dataset] = [ias_average, sas_average, geo_increments]

            if poisoning_rate == 4:
                print(dataset_name + " & {} & {:.4f} & {:.4f} & {:.2f} && {:.4f} & {:.4f} & {:.2f} && {:.4f} & {:.4f} & {:.2f} \\\\".format(
                        poisoning_rate,
                        results["allnum5cat"][0],
                        results["allnum5cat"][1],
                        results["allnum5cat"][2],
                        results["allnum10cat"][0],
                        results["allnum10cat"][1],
                        results["allnum10cat"][2],
                        results["allnumallcat"][0],
                        results["allnumallcat"][1],
                        results["allnumallcat"][2],
                    )
                )

            else:
                print(
                    "& {} & {:.4f} & {:.4f} & {:.2f} && {:.4f} & {:.4f} & {:.2f} && {:.4f} & {:.4f} & {:.2f} \\\\".format(
                        poisoning_rate,
                        results["allnum5cat"][0],
                        results["allnum5cat"][1],
                        results["allnum5cat"][2],
                        results["allnum10cat"][0],
                        results["allnum10cat"][1],
                        results["allnum10cat"][2],
                        results["allnumallcat"][0],
                        results["allnumallcat"][1],
                        results["allnumallcat"][2],
                    )
                )

    footer = r"""
    \bottomrule
    \end{tabular}
    \end{adjustbox}
    \end{table}
    """

    print(footer)

def IFCF_comparison_table(config, cross_validation=False, IAS=False):
    """
    \begin{table}[!htbp]
    \centering
    \captionof{table}{Average MSE and time of 50 runs of benchmark \cite{Suvak2021} and IFCF for all numerical and 5 categorical features. \todo{add 50 runs for real, right now just one run}}
    \begin{adjustbox}{width=\textwidth}
    \begin{tabular}{lrrrrrrrrrrrrrrrrr}
    \toprule
    & & & \multicolumn{2}{c}{Suvak et al. \cite{Suvak2021}} && \multicolumn{2}{c}{IFCF} \\
    \cmidrule{4-5}\cmidrule{7-8}
    \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
    Dataset & Type & $r$ (\%) & MSE & Time (s) && MSE & Time (s) & $\Delta$ (\%)   \\
        \midrule
    House Price\todo{add time} & Train & 4 & 0.004004 &   0.00 && 0.004800 &   0.00 &  20.67 \\
    & &  8 & 0.004736 &   0.00 && 0.006127 &   0.00 &  30.39 \\
    & & 12 & 0.005172 &   0.00 && 0.007070 &   0.00 &  37.76 \\
    & & 16 & 0.005509 &   0.00 && 0.007858 &   0.00 &  43.87 \\
    & & 20 & 0.005745 &   0.00 && 0.008562 &   0.00 &  50.71 \\
    \cmidrule{2-9}
    & Test & 4 & 0.005617 &   0.00 && 0.005779 &   0.00 &   3.17 \\
    & &  8 & 0.006570 &   0.00 && 0.007259 &   0.00 &  10.79 \\
    & & 12 & 0.007127 &   0.00 && 0.008215 &   0.00 &  15.52 \\
    & & 16 & 0.007536 &   0.00 && 0.009041 &   0.00 &  20.27 \\
    & & 20 & 0.007850 &   0.00 && 0.009767 &   0.00 &  24.71 \\
    \midrule
    Healthcare\todo{run} & Train & 4 & 0.004004 &   0.00 && 0.004800 &   0.00 &  20.67 \\
    & &  8 & 0.004736 &   0.00 && 0.006127 &   0.00 &  30.39 \\
    & & 12 & 0.005172 &   0.00 && 0.007070 &   0.00 &  37.76 \\
    & & 16 & 0.005509 &   0.00 && 0.007858 &   0.00 &  43.87 \\
    & & 20 & 0.005745 &   0.00 && 0.008562 &   0.00 &  50.71 \\
    \cmidrule{2-9}
    & Test & 4 & 0.005617 &   0.00 && 0.005779 &   0.00 &   3.17 \\
    & &  8 & 0.006570 &   0.00 && 0.007259 &   0.00 &  10.79 \\
    & & 12 & 0.007127 &   0.00 && 0.008215 &   0.00 &  15.52 \\
    & & 16 & 0.007536 &   0.00 && 0.009041 &   0.00 &  20.27 \\
    & & 20 & 0.007850 &   0.00 && 0.009767 &   0.00 &  24.71 \\
    \bottomrule
    \end{tabular}
    \end{adjustbox}
    \label{tab: 5num5cat 300samples 50runs train}
    \end{table}
    """

    header = r"""
    \begin{table}[!htbp]
    \centering
    \captionof{table}{Average MSE and time of 50 runs of benchmark \cite{Suvak2021} and IFCF for all numerical and 5 categorical features. \todo{add 50 runs for real, right now just one run}}
    \begin{adjustbox}{width=\textwidth}
    \begin{tabular}{lrrrrrrrrrrrrrrrrr}
    \toprule
    & & & \multicolumn{2}{c}{Suvak et al. \cite{Suvak2021}} && \multicolumn{2}{c}{IFCF} \\
    \cmidrule{4-5}\cmidrule{7-8}
    \rule{0pt}{10pt} % Adding space of 10pt between lines and text below
    \multicolumn{1}{c}{Dataset} & \multicolumn{1}{c}{Type} & \multicolumn{1}{c}{$r$ (\%)} & \multicolumn{1}{c}{MSE} & \multicolumn{1}{c}{Time (s)} && \multicolumn{1}{c}{MSE} & \multicolumn{1}{c}{Time (s)} & \multicolumn{1}{c}{$\Delta$ (\%)}   \\        \midrule
    """

    print(header)

    for dataset in ["house", "pharm"]:
        config["dataset"] = dataset
        if dataset == "pharm":
            print(r"\midrule")
            config["dataset_name"] = "allnum" + config["dataset_name"].split("num")[1]
        for type_data in ["Train", "Test"]:
            for poisoning_rate in [4, 8, 12, 16, 20]:
                config["poison_rate"] = poisoning_rate
                benchmark_mse = []
                benchmark_time = []
                ifcf_mse = []
                ifcf_time = []
                for run in range(config["runs"]):
                    if cross_validation:
                        config["seed"] = run
                        instance_data = instance_data_class.InstanceData(
                            config=config, benchmark_data=False, seed=run, thesis=True
                        )
                        config["regularization"] = regularization_parameter.cross_validation_lambda(
                            instance_data, np.linspace(0.001, 10, 20)
                        )["alpha"]

                    if IAS:
                        config["numerical_attack_incremental"] = True
                        folder_name = f"SAS_{run}_PR{poisoning_rate}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_{config['regularization']}"
                    else:
                        # Folder name of this experiment and rate
                        folder_name = f"R{run}_PR{poisoning_rate}_BS{config['numerical_attack_mini_batch_size']}_TS{config['training_samples']}_{config['regularization']}"

                    # Results----------------------------------
                    directory = f"results/ifcf/{dataset}/{config['dataset_name']}"
                    if not os.path.exists(os.path.join(directory, folder_name)):
                        os.makedirs(os.path.join(directory, folder_name))
                    # Check if results exist
                    if not os.path.exists(f"{directory}/{folder_name}/results.npz"):
                        # Run experiments and store
                        print(f"Running experiments for {dataset} poisoning rate {poisoning_rate} and run {run}")
                        instance_data = instance_data_class.InstanceData(
                            config, benchmark_data=False, seed=run, thesis=True
                        )
                        _, _, _, results_dict = flipping_attack.run(config, instance_data)
                        # Save results as npz
                        np.savez(f"{directory}/{folder_name}/results.npz", **results_dict)
                    else:
                        results_dict = np.load(f"{directory}/{folder_name}/results.npz")

                    # Append results
                    if type_data == "Train":
                        benchmark_mse.append(results_dict["benchmark_mse"])
                        benchmark_time.append(results_dict["benchmark_time"])
                        ifcf_mse.append(results_dict["flipping_mse"])
                        ifcf_time.append(results_dict["flipping_time"])
                    elif type_data == "Test":
                        benchmark_mse.append(results_dict["benchmark_test_mse"])
                        benchmark_time.append(0)
                        ifcf_mse.append(results_dict["flipping_test_mse"])
                        ifcf_time.append(0)
                    else:
                        benchmark_mse.append(results_dict["benchmark_validation_mse"])
                        benchmark_time.append(0)
                        ifcf_mse.append(results_dict["flipping_validation_mse"])
                        ifcf_time.append(0)

                # Calculate averages
                benchmark_mse_average = np.mean(benchmark_mse)
                benchmark_time_average = np.mean(benchmark_time)
                ifcf_mse_average = np.mean(ifcf_mse)
                ifcf_time_average = np.mean(ifcf_time)

                # Print results
                if poisoning_rate == 4 and type_data == "Train":
                    print(
                        dataset
                        + "& "
                        + type_data
                        + " & 4 & {:6.6f} & {:6.2f} && {:6.6f} & {:6.2f} & {:6.2f} \\\\".format(
                            benchmark_mse_average,
                            benchmark_time_average,
                            ifcf_mse_average,
                            ifcf_time_average,
                            (ifcf_mse_average - benchmark_mse_average) / benchmark_mse_average * 100
                        )
                    )
                elif poisoning_rate == 4 and (type_data == "Test" or type_data == "Validation"):
                    print(r"\cmidrule{2-9}")
                    print(
                        "& "
                        + type_data
                        + " & 4 & {:6.6f} & {} && {:6.6f} & {} & {:6.2f} \\\\".format(
                            benchmark_mse_average,
                            "-",
                            ifcf_mse_average,
                            "-",
                            (ifcf_mse_average - benchmark_mse_average) / benchmark_mse_average * 100
                        )
                    )
                elif poisoning_rate > 4 and (type_data == "Test" or type_data == "Validation"):
                    print(
                        "& & {:2d} & {:6.6f} & {} && {:6.6f} & {} & {:6.2f} \\\\".format(
                            poisoning_rate,
                            benchmark_mse_average,
                            "-",
                            ifcf_mse_average,
                            "-",
                            (ifcf_mse_average - benchmark_mse_average) / benchmark_mse_average * 100
                        )
                    )
                else:
                    print(
                        "& & {:2d} & {:6.6f} & {:6.2f} && {:6.6f} & {:6.2f} & {:6.2f} \\\\".format(
                            poisoning_rate,
                            benchmark_mse_average,
                            benchmark_time_average,
                            ifcf_mse_average,
                            ifcf_time_average,
                            (ifcf_mse_average - benchmark_mse_average) / benchmark_mse_average * 100
                        )
                    )

    footer = r"""
    \bottomrule
    \end{tabular}
    \end{adjustbox}
    \label{tab: 5num5cat 300samples 50runs train}
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
    # main_comparison_table(config)

    # load config yaml
    with open("programs/minlp/config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    config["numerical_attack_mini_batch_size"] = 0.1
    config["dataset_name"] = "5num5cat"
    config["dataset"] = "house"
    config["seed"] = 123   
    config["regularization"] = 0.1 

    # SAS_vs_IAS_table(config)
    # all_SAS_vs_IAS_table(config)

    config["runs"] = 1
    config["dataset_name"] = "allnum5cat"
    config["numerical_attack_mini_batch_size"] = 0.1
    # SAS_vs_IAS_table(config, cross_validation=True)
    # all_SAS_vs_IAS_table(config, cross_validation=True)
    IFCF_comparison_table(config, cross_validation=True, IAS=True)
