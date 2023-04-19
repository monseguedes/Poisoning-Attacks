"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Auxiliary module of package 'algorithm' with functions to apply solution approaches
and solve models.

- solving_MINLP
- ridge_regression
- single_attack_strategy
- iterative_attack_strategy

"""

# Self-created modules
from model.model_class import *
import model.instance_class as data
import model.pyomo_instance_class as benchmark_data


# Python Libraries
import copy
from os import path
import pandas as pd
import numpy as np
import os
import gurobipy as gp
import time
import csv
import matplotlib.pyplot as plt
import pyomo.environ as pyo

middle_space = long_space
long_space = 80
short_space = 60


def create_matrix(data):
    # get the unique row and column labels from the dictionary keys
    rows, cols = np.array(list(data.keys())).T

    # get the unique row and column labels from the dictionary keys
    row_labels = np.unique(rows)
    col_labels = np.unique(cols)

    # create a matrix array with the same shape as the resulting DataFrame
    matrix = np.zeros((len(row_labels), len(col_labels)))

    # fill the matrix with the values from the dictionary
    for row, col in data:
        row_idx = np.where(row_labels == row)[0][0]
        col_idx = np.where(col_labels == col)[0][0]
        matrix[row_idx, col_idx] = data[(row, col)]

    return matrix


# Main Functions to solve model
def solve_model(model_name: str, config: dict, checking_bilevel=False):
    """
    Function that solves any of the models
    """
    model_functions = {
        "bilevel": solve_bilevel,
        "ridge": solve_ridge,
        "benchmark": solve_benchmark,
    }

    if model_name not in model_functions:
        raise ValueError(f"invalid value of {model_name=}")

    model_func = model_functions[model_name]
    model, intance, solution = model_func(config, checking_bilevel)

    return model, intance, solution


def solve_bilevel(config: dict, checking_bilevel: bool):
    """
    Function to solve and store bilevel results
    """
    # Solve using bilevel model
    bilevel_model, bilevel_instance, bilevel_solution = solving_MINLP(
        dataset_name=config["dataset_name"],
        no_nfeatures=config["no_nfeatures"],
        no_cfeatures=config["no_cfeatures"],
        poison_rate=config["poison_rate"],
        training_samples=config["training_samples"],
        time_limit=config["time_limit"],
        feasibility=config["feasibility"],
        seed=config["seed"],
        function=config["function"],
    )

    if checking_bilevel:
        # To check that solutions of lower level are optimal
        pridge_model, pridge_instance, pridge_solution = solve_ridge(
            dataset_name=config["dataset_name"],
            training_samples=config["training_samples"],
            seed=config["seed"],
            function=config["function"],
            poisoned=True,
            poison_solutions=bilevel_solution,
            bilevel_instance=bilevel_instance,
        )

    return bilevel_model, bilevel_instance, bilevel_solution


def solve_ridge(config: dict, checking_bilevel: bool):
    """
    Function to solve and store ridge results
    """
    # Normal ridge regression
    ridge_model, ridge_instance, ridge_solution = ridge_regression(
        dataset_name=config["dataset_name"],
        training_samples=config["training_samples"],
        seed=config["seed"],
        function=config["function"],
    )

    return ridge_model, ridge_instance, ridge_solution


def solve_benchmark(config: dict, checking_bilevel=False):
    """
    Function to solve and store benchmark results
    """
    # Solve benchmark
    opt = pyo.SolverFactory("ipopt")

    print("" * 2)
    print("*" * long_space)
    print("ITERATIVE CONTINUOUS NONLINEAR ALGORITHM")
    print("*" * long_space)

    print("Building data class")
    instance_data = benchmark_data.InstanceData(
        dataset_name=config["dataset_name"]
    )
    instance_data.prepare_instance(
        poison_rate=config["poison_rate"],
        training_samples=config["training_samples"],
        no_psubsets=config["no_psubsets"],
        seed=config["seed"],
    )

    (
        benchmark_model,
        benchmark_instance,
        benchmark_solution,
    ) = iterative_attack_strategy(
        opt=opt, instance_data=instance_data, config=config
    )
    print("*" * middle_space)

    return benchmark_model, benchmark_instance, benchmark_solution


def solve_gurobi(config: dict, instance_data):
    # Create model
    print("And start solving them using Gurobi")
    gurobi_model = gp.Model("Poisoning_Attack")
    my_model = PoisonAttackModel(
        gurobi_model, instance_data, function=config["function"]
    )
    m = my_model.model

    print("Solving model...")
    m.params.NonConvex = 2
    m.params.FeasibilityTol = config["feasibility"]
    m.params.TimeLimit = config["time_limit"]

    results = m.optimize()
    print("Model has been solved")

    ### Store results
    index = pd.MultiIndex.from_tuples(
        my_model.x_poison_num.keys(), names=("sample", "feature")
    )  # Create index from the keys (indexes) of the solutions of x_poison_num
    num_poison_solution = pd.Series(
        [element.X for element in my_model.x_poison_num.values()], index=index
    )  # Make a dataframe with solutions and desires index
    index = pd.MultiIndex.from_tuples(
        my_model.x_poison_cat.keys(), names=("sample", "feature", "category")
    )  # Create index from the keys (indexes) of the solutions of x_poison_cat
    cat_poison_solution = pd.Series(
        [element.X for element in my_model.x_poison_cat.values()], index=index
    )  # Make a dataframe with solutions and desires index
    num_poison_solution.name = "x_train_num"
    cat_poison_solution.name = "x_train_cat"

    solutions_dict = {
        "x_poison_num": num_poison_solution.to_dict(),
        "x_poison_cat": cat_poison_solution.to_dict(),
        "weights_num": [element.X for element in my_model.weights_num.values()],
        "weights_cat": [element.X for element in my_model.weights_cat.values()],
        "bias": my_model.bias.X,
        "objective": m.ObjVal,
    }

    return my_model, solutions_dict


def solve_pyomo(opt, config: dict, instance_data):
    new_model = BenchmarkPoisonAttackModel(
        instance_data, function=config["function"]
    )

    # Initialise solutions
    for psample, numfeature in itertools.product(
        new_model.psamples_set, new_model.numfeatures_set
    ):
        new_model.x_poison_num[
            psample, numfeature
        ].value = instance_data.num_x_poison_dataframe.to_dict()[psample, numfeature]

    # Solve model
    results = opt.solve(new_model, load_solutions=True, tee=True)

    ### Store results of the poison subset found during this iteration
    index = pd.MultiIndex.from_tuples(
        new_model.x_poison_num.keys(), names=("sample", "feature")
    )  # Create index from the keys (indexes) of the solutions of x_poison
    poison_solution = pd.Series(
        [variable._value for variable in new_model.x_poison_num.values()],
        index=index,
    )  # Make a dataframe with solutions and desires index
    new_x_train_num = poison_solution
    new_x_train_num.name = "x_train_num"
    ###

    solutions_dict = {
        "x_poison_num": poison_solution.to_dict(),
        "weights_num": {
            index: new_model.weights_num[index].value
            for index in new_model.numfeatures_set
        },
        "weights_cat": {
            (cat_feature, category): new_model.weights_cat[
                (cat_feature, category)
            ].value
            for cat_feature in new_model.catfeatures_set
            for category in new_model.categories_sets[cat_feature]
        },
        "bias": new_model.bias.value,
        "objective": pyo.value(new_model.objective_function),
    }

    return new_model, solutions_dict


# Algorithmic approaches
def solving_MINLP(
    dataset_name: str,
    no_nfeatures: int,
    no_cfeatures: int,
    poison_rate: int,
    training_samples: int,
    seed: int,
    function="MSE",
    feasibility=0.0001,
    time_limit=600,
):
    """
    Algorithm for solving the MINLP bilevel model.
    """

    print("" * 2)
    print("*" * long_space)
    print("MIXED-INTEGER NONLINEAR ALGORITHM")
    print("*" * long_space)

    print("Building data class")
    # Initialise data (ready for building first instance)
    instance_data = data.InstanceData(dataset_name=dataset_name)
    instance_data.prepare_instance(
        poison_rate=poison_rate,
        training_samples=training_samples,
        no_nfeatures=no_nfeatures,
        no_cfeatures=no_cfeatures,
        no_poison_subsets=1,
        seed=seed,
    )
    print("*" * middle_space)
    print("")

    # Create model
    gurobi_model = gp.Model("Poisoning_Attack")
    my_model = PoisonAttackModel(gurobi_model, instance_data, function=function)
    m = my_model.model

    # Callback function for bounds
    def data_cb(model, where):
        if where == gp.GRB.Callback.MIP:
            cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

            # Did objective value or best bound change?
            if model._obj != cur_obj or model._bd != cur_bd:
                model._obj = cur_obj
                model._bd = cur_bd
                model._data.append([time.time() - m._start, cur_obj, cur_bd])

    # Prepare objects for callback
    m._obj = None
    m._bd = None
    m._data = []
    m._start = time.time()

    print("*" * middle_space)
    print("SOLVING MODEL")
    print("*" * middle_space)

    m.params.NonConvex = 2
    m.params.FeasibilityTol = feasibility
    m.params.TimeLimit = time_limit
    results = m.optimize(callback=data_cb)

    print("*" * middle_space)
    print("Model has been solved")
    print("*" * middle_space)
    print("" * 2)

    # m.write('out.lp')
    # m.write('solut.sol')

    # Save bounds in file
    file_name = "_".join(
        [
            "bounds",
            str(my_model.no_numfeatures),
            str(my_model.no_catfeatures),
            str(len(instance_data.x_train_dataframe)),
            str(int(instance_data.poison_rate * 100)),
        ]
    )
    with open("results/bounds/files/" + file_name + ".csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(m._data)

    def plot_bounds(file_name):
        """
        Function to plot upper and lower bounds.
        """
        # Plot bounds
        bounds_df = pd.read_csv(
            "results/bounds/files/" + file_name + ".csv",
            names=["Time", "Primal", "Dual"],
            header=None,
        )
        bounds_df = bounds_df.iloc[5:]
        bounds_df.set_index("Time", drop=True, inplace=True)
        bounds_df[["Primal", "Dual"]].plot(
            style={"Primal": "-k", "Dual": "--k"}, figsize=(8.5, 6.5)
        )
        plt.title("Evolution of Primal and Dual Bounds", fontsize=30)
        plt.ylabel("Bounds", fontsize=20)
        plt.xlabel("Time (s)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig(
            "results/bounds/plots/" + "plot_" + file_name + ".pdf",
            transparent=True,
            bbox_inches="tight",
        )
        # plt.show()
        plt.close()

    # plot_bounds(file_name)

    def initialise_opt_solution(m):
        """
        Initialise model with known optimal solution.
        """
        m.write("out.sol")
        m.read("out.sol")
        m.update()
        results = m.optimize()

        return results

    # results = initialise_opt_solution(m)

    ### Store results
    index = pd.MultiIndex.from_tuples(
        my_model.x_poison_num.keys(), names=("sample", "feature")
    )  # Create index from the keys (indexes) of the solutions of x_poison_num
    num_poison_solution = pd.Series(
        [element.X for element in my_model.x_poison_num.values()], index=index
    )  # Make a dataframe with solutions and desires index

    index = pd.MultiIndex.from_tuples(
        my_model.x_poison_cat.keys(), names=("sample", "feature", "category")
    )  # Create index from the keys (indexes) of the solutions of x_poison_cat
    cat_poison_solution = pd.Series(
        [element.X for element in my_model.x_poison_cat.values()], index=index
    )  # Make a dataframe with solutions and desires index

    num_poison_solution.name = "x_train_num"
    cat_poison_solution.name = "x_train_cat"

    solutions_dict = {
        "x_poison_num": num_poison_solution.to_dict(),
        "x_poison_cat": cat_poison_solution.to_dict(),
        "weights_num": [element.X for element in my_model.weights_num.values()],
        "weights_cat": [element.X for element in my_model.weights_cat.values()],
        "bias": my_model.bias.X,
    }

    print("*" * middle_space)
    print("RESULTS")
    print("*" * middle_space)
    print("Numerical weights are ", solutions_dict["weights_cat"])
    print("Categorical weights are ", solutions_dict["weights_num"])
    print("Bias is ", solutions_dict["bias"])
    print("*" * middle_space)
    print("Objective value is ", m.ObjVal)
    print("*" * middle_space)
    print("*" * long_space)
    print("" * 2)

    return my_model, instance_data, solutions_dict


def ridge_regression(
    dataset_name: str,
    training_samples: int,
    seed: int,
    function="MSE",
    initialized_solution=0,
    poisoned=False,
    poison_solutions=None,
    bilevel_instance=None,
):
    """
    Fits a fit regression model. The goal of this function is to be able to
    compare the performance of poisoned and nonpoisoned models.
    """

    # Initialise data (ready for building first instance)
    instance_data = data.InstanceData(dataset_name=dataset_name)
    instance_data.create_dataframes(training_samples=training_samples, seed=seed)
    instance_data.split_dataframe()
    instance_data.regularization_parameter()

    if (
        poisoned == True
    ):  # Take as data the training samples + poisoning samples instead
        instance_data = bilevel_instance
        instance_data.append_poisoning_attacks(poison_solutions)

    # Create model
    gurobi_model = gp.Model("Regression_Model")
    my_model = RegressionModel(gurobi_model, instance_data, function=function)
    m = my_model.model
    print("Model has been built")

    print("Solving the model...")
    # Solve model
    # m.params.NonConvex = 2
    m.params.FeasibilityTol = 0.0001
    results = m.optimize()
    print("Model has been solved")

    ###
    solutions_dict = {
        "weights": [element.X for element in my_model.weights.values()],
        "bias": my_model.bias.X,
    }

    print("Objective value is ", m.ObjVal)

    return my_model, instance_data, solutions_dict


def single_attack_strategy(
    opt: pyo.SolverFactory,
    dataset_name: str,
    poison_rate: int,
    seed: int,
    initialized_solution=0,
):
    """
    Algorithm for single attack strategy of benchmark paper.

    Input parameters are: dataframe name, poison rate, seed (to control randomness), and initialized solution.
    """

    # Initialise data (ready for building first instance)
    instance_data = benchmark_data.InstanceData(dataset_name=dataset_name, seed=seed)
    instance_data.prepare_instance(poison_rate=poison_rate, N=1)

    # Create abstract model
    model = BenchmarkPoisonAttackModel(instance_data)

    print("Model has been built")

    print("Solving the model...")
    # Solve model
    results = opt.solve(model, load_solutions=True, tee=True)
    # results = solver_manager.solve(model, load_solutions=True, tee=True, opt=opt)
    print("Model has been solved")

    ### Store results of the poison subset found during this iteration
    index = pd.MultiIndex.from_tuples(
        model.x_poison.keys(), names=("sample", "feature")
    )  # Create index from the keys (indexes) of the solutions of x_poison
    poison_solution = pd.Series(
        model.x_poison.values(), index=index
    )  # Make a dataframe with solutions and desires index
    new_x_train = poison_solution
    new_x_train.name = "x_train"
    ###

    solutions_dict = {
        "x_poison": poison_solution.to_dict(),
        "weights": model.weights.value,
        "bias": model.bias.value,
    }

    return model, instance_data, solutions_dict


def iterative_attack_strategy(opt: pyo.SolverFactory, instance_data, config):
    """
    Algorithm for iterative attack strategy.

    It starts by creating the abstract model, and an initial data object for creating the first
    instance. After this, while the iteration count is smaller than the number of subsets (there
    is an iteration per subset), the model instance is created with the intance data object and the
    model is solved for current instance. After that, solutions are stored in a dataframe, and data
    object for instance is updated to that current iteration becomes data. Then, we go back to start
    of while loop and process is repeated for all subsets/iterations.
    """

    print("" * 2)
    print("*" * long_space)
    print("ITERATIVE ATTACK STRATEGY")
    print("*" * long_space)

    # Iteration count
    iteration = 1
    iterations_solutions = []

    model = BenchmarkPoisonAttackModel(instance_data, config["function"])

    # Initialise variables
    for psample, numfeature in itertools.product(
        model.psamples_set, model.numfeatures_set
    ):
        model.x_poison_num[
            psample, numfeature
        ].value = instance_data.num_x_poison_dataframe.to_dict()[psample, numfeature]

    while (
        iteration <= instance_data.no_psubsets
    ):  # There is an iteration for each poison subset
        # Solve model
        results = opt.solve(model, load_solutions=True, tee=True)

        # Store results of the poison subset found during this iteration
        index = pd.MultiIndex.from_tuples(
            model.x_poison_num.keys(), names=("sample", "feature")
        )  # Create index from the keys (indexes) of the solutions of x_poison
        poison_solution = pd.Series(
            [variable._value for variable in model.x_poison_num.values()],
            index=index,
        )  # Make a dataframe with solutions and desires index
        new_x_poison_num = poison_solution
        new_x_poison_num.name = "x_poison_num"
        solutions_dict = {
            "x_poison_num": poison_solution.to_dict(),
            "weights_num": {
                index: model.weights_num[index].value for index in model.numfeatures_set
            },
            "weights_cat": {
                (cat_feature, category): model.weights_cat[
                    (cat_feature, category)
                ].value
                for cat_feature in model.catfeatures_set
                for category in model.categories_sets[cat_feature]
            },
            "bias": model.bias.value,
            "objective": pyo.value(model.objective_function),
        }
        iterations_solutions.append(solutions_dict)

        instance_data.update_data(iteration, new_x_poison_num=new_x_poison_num)
        # Update parameter from data
        for k, v in enumerate(instance_data.flag_array):
            model.flag_array[k + 1].value = v
        for k, v in instance_data.num_x_poison_dataframe.to_dict().items():
            model.x_poison_num_data[k].value = v
        for k, v in (
            instance_data.get_cat_poison_dataframe().to_dict()["x_poison_cat"].items()
        ):
            model.x_poison_cat[k].value = v
        for k, v in instance_data.y_poison_dataframe.to_dict()["y_poison"].items():
            model.y_poison[k].value = v

        next_iteration = 0 if iteration == instance_data.no_psubsets else iteration

        dataframe = instance_data.num_x_poison_dataframe.copy().iloc[
            next_iteration
            * instance_data.no_numfeatures
            * instance_data.no_psamples_per_subset : (next_iteration + 1)
            * instance_data.no_numfeatures
            * instance_data.no_psamples_per_subset
        ]

        dataframe.index = dataframe.index.set_levels(
            dataframe.index.levels[0] - dataframe.index.get_level_values(0)[0] + 1,
            level=0,
        )
        for psample in model.psamples_per_subset_set:
            for numfeature in model.numfeatures_set:
                model.x_poison_num[psample, numfeature].set_value(
                    dataframe.to_dict()[psample, numfeature]
                )

        # TODO Update the master dataframe
        instance_data.num_x_poison_dataframe.to_numpy()[
            (iteration - 1)
            * model.no_numfeatures
            * model.no_psamples_per_subset : iteration
            * model.no_numfeatures
            * model.no_psamples_per_subset
        ] = new_x_poison_num.to_numpy()

        print("Iteration no. {} is finished".format(iteration))
        iteration += 1

    final_solutions = {
        "x_poison_num": instance_data.num_x_poison_dataframe.to_dict(),
        "weights_num": {
            index: model.weights_num[index].value for index in model.numfeatures_set
        },
        "weights_cat": {
            (cat_feature, category): model.weights_cat[(cat_feature, category)].value
            for cat_feature in model.catfeatures_set
            for category in model.categories_sets[cat_feature]
        },
        "bias": model.bias.value,
        "objective": pyo.value(model.objective_function),
    }

    print("" + "\n" + "*" * middle_space + "\n" + "RESULTS" + "\n" + "*" * middle_space)
    # print('Numerical weights are ', final_solutions['weights_cat'])
    # print('Categorical weights are ', final_solutions['weights_num'])
    # print('Bias is ', final_solutions['bias'])
    # print('*' * middle_space)
    print(
        "Objectives are: ",
        [dict["objective"] for dict in iterations_solutions],
    )
    print("Objective value is ", pyo.value(model.objective_function))
    print("*" * middle_space + "\n" + "*" * long_space + "\n" + "" * 2)

    solutions = {
        "iteration no." + str(iteration + 1): solution
        for iteration, solution in enumerate(iterations_solutions)
    }
    objectives = {
        "iteration no." + str(iteration + 1): solution["objective"]
        for iteration, solution in enumerate(iterations_solutions)
    }

    return model, instance_data, final_solutions


def benchmark_plus_optimising_heuristic(config: dict):
    """
    This is the heuristic algorithm we use to get feasible
    solutions. It works as follows. We quickly optimise numerical
    features (locally) using ipopt. Then take these solutions and
    fix them. Then we optimize subsets of features on top
    of this first optimization. Features are chosen using LASSO.

    PROBLEM: Second optimization round is too slow.
    BENEFIT: Guarantees benchmark improvement if second problem is
    solved to optimality. PROBLEM: this is never the case.
    SOLUTION: Chop earlier and iterate over num features.
    """

    print("" * 2 + "\n" + "-" * long_space + "\n" + "-" * long_space)
    print("HEURISTIC ALGORITHM: CONTINOUS + SUBSET MIXED-INTEGER")
    print("-" * long_space + "\n" + "-" * long_space)

    # Optimise numerical features locally
    benchmark_model, benchmark_instance, benchmark_solution = solve_benchmark(
        config
    )
    print("Benchmark has been solved, now adding this solution to data")

    # Get poisoning samples as data
    benchmark_data = benchmark_solution["x_poison_num"]
    matrix = create_matrix(benchmark_data)

    print("We first build a data class for a mixed-integer model")
    instance_data = data.InstanceData(dataset_name=config["dataset_name"])
    instance_data.prepare_instance(
        poison_rate=config["poison_rate"],
        training_samples=config["training_samples"],
        no_nfeatures="all",
        no_cfeatures="all",
        no_poison_subsets=config["no_psubsets"],
        seed=config["seed"],
    )
    print("And then change old columns for solution from continuous nonlinear model")
    instance_data.x_poison_dataframe[instance_data.numerical_columns] = matrix

    print("Now select new features to be optimised using Gurobi")
    instance_data.feature_selection(
        no_nfeatures=config["no_nfeatures"],
        no_cfeatures=config["no_cfeatures"],
    )
    instance_data.poison_samples()

    chosen_features = instance_data.chosen_categorical

    for i in range(0, len(chosen_features), config["heuristic_subset"]):
        instance_data.chosen_categorical = chosen_features[
            i : i + config["heuristic_subset"]
        ]
        print("Subset of features is", instance_data.chosen_categorical)

        my_model, solutions_dict = solve_gurobi(config, instance_data)

        # Update categorical column just solved
        for key, value in solutions_dict["x_poison_cat"].items():
            instance_data.cat_x_poison_dataframe.loc[key] = value
        benchmark_instance.cat_poison_dataframe_data[
            "x_poison_cat"
        ] = instance_data.cat_x_poison_dataframe["x_data_poison_cat"]
        benchmark_instance.complete_cat_poison_dataframe.to_numpy()[:] = (
            instance_data.cat_x_poison_dataframe["x_data_poison_cat"]
            .to_numpy()
            .reshape(benchmark_instance.complete_cat_poison_dataframe.shape)
        )
        benchmark_instance.update_cat_poison_dataframe()
        opt = pyo.SolverFactory("ipopt")
        print(benchmark_instance.flag_array)
        print
        (
            benchmark_model,
            benchmark_instance,
            solution,
        ) = iterative_attack_strategy(opt, benchmark_instance, config)

        # # Get poisoning samples as data
        # benchmark_data = solution['x_poison_num']
        # matrix = create_matrix(benchmark_data)
        # instance_data.x_poison_dataframe[instance_data.numerical_columns] = matrix
        instance_data.poison_samples(only_num=True)

    print("Heuristic objective value is     {:.5f}".format(solutions_dict["objective"]))
    print("New benchmark objective value is {:.5f}".format(solution["objective"]))
    print(
        "Benchmark objective value is     {:.5f}".format(
            benchmark_solution["objective"]
        )
    )
    print(
        "Increment is                     {:.5f}".format(
            (solutions_dict["objective"] - benchmark_solution["objective"])
            / benchmark_solution["objective"]
            * 100
        )
    )

    return my_model, instance_data, solutions_dict


def benchmark_plus_optimising_chopping_heuristic(config: dict):
    """
    This is the heuristic algorithm we use to get feasible
    solutions. It works as follows. We quickly optimise numerical
    features (locally) using ipopt. Then take these solutions and
    fix them. Then we create smaller problems with just those columns
    we want to generate categorical features for. We find feasible
    categorical features this way. Features are chosen using LASSO.

    PROBLEM: Ignoring rest of dataset leads to bad results.
    BENEFIT: Very fast, but benchmark is faster and better.
    """

    # Solve model using benchmark method.
    opt = pyo.SolverFactory("ipopt")
    (
        benchmark_model,
        benchmark_instance,
        benchmark_solution,
    ) = iterative_attack_strategy(
        opt=opt,
        dataset_name=config["dataset_name"],
        poison_rate=config["poison_rate"],
        training_samples=config["training_samples"],
        no_psubsets=config["no_psubsets"],
        seed=config["seed"],
    )
    print("Benchmark has been solved, now let us add this solution to data")

    # Get poisoning samples as data
    benchmark_data = benchmark_solution["x_poison_num"]

    matrix = create_matrix(benchmark_data)

    # Optimise now categorical features separately (MINLP)
    print("We now build the original data...")
    instance_data = data.InstanceData(dataset_name=config["dataset_name"])
    instance_data.prepare_instance(
        poison_rate=config["poison_rate"],
        training_samples=config["training_samples"],
        no_nfeatures="all",
        no_cfeatures="all",
        seed=config["seed"],
    )
    # Change the necessary dataframes to have benchmark as columns in data dataframes (x_poison_dataframe)
    print("And change old columns for benchmark solution")
    instance_data.x_poison_dataframe[instance_data.numerical_columns] = matrix

    print("Now select new categorical features to be optimised using Gurobi")
    instance_data.feature_selection(
        no_nfeatures=config["no_nfeatures"],
        no_cfeatures=config["no_cfeatures"],
    )
    instance_data.poison_samples()

    # Chop dataset and create one per each categorical feature
    for feature in instance_data.chosen_categorical:
        chose_columns = (
            [
                column
                for column in instance_data.categorical_columns
                if column.startswith(str(feature) + ":")
            ]
            + [str(column) for column in instance_data.chosen_numerical]
            + ["target"]
        )
        # Store as old datasets for now to minimise effort (always same name)
        instance_data.whole_dataframe[chose_columns].to_csv(
            "data/subset/data-binary.csv"
        )
        # Optimise just these small datasets for all their categorical features
        small_instance = data.InstanceData(dataset_name="subset")
        small_instance.prepare_instance(
            poison_rate=config["poison_rate"],
            training_samples=config["training_samples"],
            no_nfeatures=config["no_nfeatures"],
            no_cfeatures=1,
            seed=config["seed"],
        )

        # Create model
        small_model, solutions_dict = solve_gurobi(config, small_instance)
        # Substitute these values back into the full dataset
        index = pd.MultiIndex.from_tuples(
            small_model.x_poison_cat.keys(),
            names=("sample", "feature", "category"),
        )  # Create index from the keys (indexes) of the solutions of x_poison_cat
        cat_poison_solution = pd.Series(
            [element.X for element in small_model.x_poison_cat.values()],
            index=index,
        )  # Make a dataframe with solutions and desires index
        cat_poison_solution.name = "x_train_cat"
        solutions_dict = {"x_poison_cat": cat_poison_solution.to_dict()}
        for sample in range(1, instance_data.no_psamples + 1):
            for category in range(1, instance_data.no_categories_dict[feature] + 1):
                print(
                    "Old value: ",
                    instance_data.cat_x_poison_dataframe.loc[
                        (sample, feature, category)
                    ],
                )
                instance_data.cat_x_poison_dataframe.loc[
                    (sample, feature, category)
                ] = int(solutions_dict["x_poison_cat"][(sample, feature, category)])
                print(
                    "New value: ",
                    instance_data.cat_x_poison_dataframe.loc[
                        (sample, feature, category)
                    ],
                )

    # update x_poison_dataframe
    print("Now select new categorical features to be optimised using Gurobi")
    instance_data.chosen_numerical = []
    instance_data.chosen_categorical = []

    # benchmark_instance.cat_poison_dataframe = instance_data.cat_x_poison_dataframe.rename(columns={'x_data_poison_cat': 'x_poison_cat'}) #TODO update master dataframe

    new_model, solutions_dict = solve_pyomo(config, benchmark_instance)

    print("Objective value is ", solutions_dict["objective"])
    print("Benchmark objective value is ", benchmark_solution["objective"])

    return new_model, benchmark_data, solutions_dict


def flipping_heuristic(config: dict, instance, solution):
    """
     This is the heuristic algorithm we use to get feasible
    solutions. It works as follows. We quickly optimise numerical
    features (locally) using ipopt. Then take these solutions and
    fix them. Then we perturb categorical features as follow. For
    each sample, if the response variable is smaller than 0.5, make
    the column with the largest weight for all features be 1, and all
    other be 0. If target is geq 0.5, make the category with the
    smallest weight equal to 1, and everything else equal to 0.

    PROBLEM:
    """

    print("" * 2)
    print("-" * long_space)
    print("-" * long_space)
    print("HEURISTIC ALGORITHM: CONTINOUS + FLIPPING CATEGORICAL")
    print("-" * long_space)
    print("-" * long_space)

    # # Solve numerical features locally
    # _, instance, solution = solve_benchmark(config)
    # print('Benchmark has been solved, now let us add this solution to data')

    benchmark_objective = solution["objective"]

    # Flip all categorical features.
    for i in range(0, instance.no_psamples, config["heuristic_subset"]):
        # Save the original data and try flipping. If the flipping does not improve,
        # we restore the data.
        original_instance = copy.deepcopy(instance)
        original_solution = copy.deepcopy(solution)

        chosen_samples = list(range(1, instance.no_psamples + 1))[
            i : i + config["heuristic_subset"]
        ]
        print("Subset of samples is", chosen_samples)

        for psample in chosen_samples:
            # Make (just num) prediction
            cat_weights = solution["weights_cat"]
            num_weights = solution["weights_num"]
            num_features = {
                k: v for k, v in solution["x_poison_num"].items() if k[0] == psample
            }
            num_y = (
                np.array(list(num_weights.values()))
                @ np.array(list(num_features.values()))
                + solution["bias"]
            )
            target_y = instance.y_poison_dataframe.loc[psample, "y_poison"]
            difference = num_y - target_y

            # We consider two case: Make prediction as large as possible and make prediction
            # as small as possible. We then take the best one.

            # categories_up/down[feature] is the category to push prediction up/down.
            cat_features = list(instance.categories_dict.keys())
            categories_up = dict()
            categories_down = dict()
            for feature in cat_features:
                # Filter the keys based on given values for first two elements
                filtered_keys = [k for k in cat_weights.keys() if k[0] == feature]
                categories_up[feature] = max(filtered_keys, key=cat_weights.get)[1]
                categories_down[feature] = min(filtered_keys, key=cat_weights.get)[1]

            # Let's compute the prediction of each case.
            pred_up = num_y + sum(
                cat_weights[(feature, categories_up[feature])]
                for feature in cat_features
            )
            pred_down = num_y + sum(
                cat_weights[(feature, categories_down[feature])]
                for feature in cat_features
            )

            if np.abs(pred_up - target_y) < np.abs(pred_down - target_y):
                # Pushing down is more effective.
                categories_chosen = categories_down
            else:
                # Pushing up is more effective.
                categories_chosen = categories_up

            # Update the dataframes.
            instance.cat_poison_dataframe_data.loc[psample, :, :] = 0
            for feature in cat_features:
                instance.cat_poison_dataframe_data.loc[
                    psample, feature, categories_chosen[feature]
                ] = 1
                instance.complete_cat_poison_dataframe.to_numpy()[:] = (
                    instance.cat_poison_dataframe_data["x_poison_cat"]
                    .to_numpy()
                    .reshape(instance.complete_cat_poison_dataframe.shape)
                )
                instance.update_cat_poison_dataframe()

        opt = pyo.SolverFactory("ipopt")
        _, instance, solution = iterative_attack_strategy(
            opt, instance, config
        )

        if original_solution["objective"] > solution["objective"]:
            # The flipping actually made the poisoning attack worse.
            # We will restore the original data and try flipping the next sample.
            solution = original_solution
            instance = original_instance

    print("Objective value is           ", solution["objective"])
    print("Benchmark objective value is ", benchmark_objective)

    return instance, solution


def IP_heuristic(config: dict):
    pass
