"""Run iterative attack which which poison training data row by row"""

import copy
import timeit

import numpy as np
import pandas as pd

import pyomo_model
import ridge_regression
import testing

long_space = 80
short_space = 60
middle_space = long_space


def run(config, instance_data, model=None):
    """Run iterative attack which which poison training data row by row

    This is a hueristic to optimize numerical features row by row using IPOPT.
    The given data is not modified but a copy will be returned.

    Parameters
    ----------
    config : dict
    instance_data : InstanceData

    Returns
    -------
    model : pyomo.block
    modified_data : InstanceData
    solution : dict[str, pd.DataFrame]
    """
    config = copy.deepcopy(config)
    instance_data = instance_data.copy()

    # Solve benchmark

    print("" * 2)
    print("*" * long_space)
    print("NUMERICAL ATTACK STRATEGY")
    print("*" * long_space)

    if not config.get("solver_name"):
        config["solver_name"] = "ipopt"

    if model is None:
        model = pyomo_model.PyomoModel(instance_data, config)
    else:
        model.update_parameters(instance_data)

    n_epochs = config["numerical_attack_n_epochs"]
    mini_batch_size = config["numerical_attack_mini_batch_size"]

    incremental = config["numerical_attack_incremental"]

    if incremental and (n_epochs > 1):
        raise ValueError(f"n_epochs should be 1 when incremental but got {n_epochs}")

    no_poison_samples = instance_data.no_poison_samples

    if isinstance(
        mini_batch_size, int
    ):  # TODO what happens for the case with just one batch?
        mini_batch_absolute_size = mini_batch_size
    elif mini_batch_size == "all":
        mini_batch_absolute_size = no_poison_samples
    else:
        # mini batch size is specified as a fraction
        mini_batch_absolute_size = int(no_poison_samples * mini_batch_size)
    mini_batch_absolute_size = max(mini_batch_absolute_size, 1)
    breaks = np.arange(0, no_poison_samples, mini_batch_absolute_size)
    breaks = np.r_[breaks, no_poison_samples]
    n_mini_batches = len(breaks) - 1

    solution_list = []

    F = model.POISON_DATA_FIXED
    O = model.POISON_DATA_OPTIMIZED
    R = model.POISON_DATA_REMOVED

    regression_parameters = ridge_regression.run(config, instance_data)
    best_mse = regression_parameters["mse"]
    best_solution = regression_parameters
    best_instance_data = instance_data

    # Project numerical features
    if config["binary"]:
        round_except_last = (
            lambda x: round(x, 0)
            if x.name != best_instance_data.poison_dataframe.columns[-1]
            else x
        )
        best_instance_data.poison_dataframe = best_instance_data.poison_dataframe.apply(
            round_except_last
        )
        round_except_last = (
            lambda x: round(x, 0)
            if x.name != instance_data.poison_dataframe.columns[-1]
            else x
        )
        instance_data.poison_dataframe = instance_data.poison_dataframe.apply(
            round_except_last
        )

    for epoch in range(n_epochs):
        for mini_batch_index in range(n_mini_batches):
            # Modify num_feature_flag to specify which features are to be
            # optimized, fixed and removed.
            # model.unfix: 0
            # model.fix: 1
            # model.remove: 2
            num_feature_flag = np.full(instance_data.no_poison_samples, -1)
            start, stop = breaks[mini_batch_index], breaks[mini_batch_index + 1]
            num_feature_flag[:start] = F
            num_feature_flag[start:stop] = O
            if incremental:
                num_feature_flag[stop:] = R
            else:
                num_feature_flag[stop:] = F
            model.set_poison_data_status(
                instance_data, num_feature_flag[:, None], model.POISON_DATA_FIXED
            )

            model.solve()
            solution = model.get_solution()
            if solution["mse"] > best_mse:
                buff = instance_data.copy()
                model.update_data(instance_data)
                best_mse = solution["mse"]
                best_solution = solution
                best_instance_data = instance_data.copy()
            else:
                instance_data = best_instance_data.copy()

            solution_list.append(solution)
            if (epoch * n_mini_batches + mini_batch_index) % 20 == 0:
                print(f"{'epoch':>5s}  {'batch':>5s}  {'mse':>9s}  {'best_mse':>9s}")
            print(
                f"{epoch:5d}  {mini_batch_index:5d}  {solution['mse']:9.6f}  {best_mse:9.6f}"
            )

    # This will break when solution_list is empty, but maybe it's unlikely
    keys = solution_list[0].keys()
    out = {key: np.stack([x[key] for x in solution_list]) for key in keys}

    print("mse in each iteration:")
    print(out["mse"])
    print("improvement from the start (%):")
    print(((out["mse"] - out["mse"][0]) / out["mse"][0] * 100).round(2))

    return model, best_instance_data, best_solution


if __name__ == "__main__":
    import doctest

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
