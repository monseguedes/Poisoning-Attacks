# -*- coding: utf-8 -*-

"""Run iterative attack which poison categorical data"""

import copy

import numpy as np
import pyomo_model
import ridge_regression

long_space = 80
short_space = 60
middle_space = long_space


def run(config, instance_data, model=None, features=None):
    """Run categorical attack which poison categorical data

    This is a hueristic to poison categorical features using gurobi.
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

    print("\n" + "*" * long_space)
    print("CATEGORICAL ATTACK STRATEGY")
    print("*" * long_space)

    if not config.get("solver_name"):
        config["solver_name"] = "gurobi"
    np.testing.assert_equal(config["solver_name"], "gurobi")

    if model is None:
        model = pyomo_model.PyomoModel(instance_data, config)
    else:
        model.update_parameters(instance_data)

    if features is None:
        features = instance_data.categorical_feature_names
    elif not isinstance(features, list):
        features = [features]

    n_epochs = config["categorical_attack_n_epochs"]

    no_poison_samples = instance_data.no_poison_samples

    objective_list = []  # TODO where used?

    counter = 0

    solution_list = []

    F = model.POISON_DATA_FIXED
    O = model.POISON_DATA_OPTIMIZED
    R = model.POISON_DATA_REMOVED

    regression_parameters = ridge_regression.run(config, instance_data)
    best_mse = regression_parameters["mse"]
    best_solution = regression_parameters

    # We want to solve for a subset of numerical features given by config['categorical_attack_no_nfeatures'],
    # and a subset of categorical ones, given by config['categorical_attack_no_cfeatures']. But for categorical
    # ones we want to iterate over a subset of batches.

    for epoch in range(n_epochs):
        # for poison_sample_index in range(no_poison_samples):
            for categorical_feature_name in features:  # TODO change this to first
                num_feature_flag = F
                shape = (instance_data.no_poison_samples, instance_data.no_catfeatures)
                cat_feature_flag = np.full(shape, F)
                # TODO fix this to be flexible
                for poison_sample_index in range(no_poison_samples):
                    cat_feature_flag[poison_sample_index, categorical_feature_name] = O
                model.set_poison_data_status(
                    instance_data, num_feature_flag, cat_feature_flag
                )
                model.solve()
                solution = model.get_solution()
                if solution["mse"] > best_mse:
                    model.update_data(instance_data)
                    best_mse = solution["mse"]
                    best_solution = solution
                solution_list.append(solution)
            if (counter) % 20 == 0:
                print(f"{'epoch':>5s}  {'row':>5s}  {'mse':>9s}")
            # print(f"{epoch:5d}  {poison_sample_index:5d}  {solution['mse']:9.6f}")
            print(f"{epoch:5d}  {solution['mse']:9.6f}")
            counter += 1

    # This will break when objective_list is empty, but maybe it's unlikely
    # This will break when solution_list is empty, but maybe it's unlikely
    keys = solution_list[0].keys()
    out = {key: np.stack([x[key] for x in solution_list]) for key in keys}

    print("mse in each iteration:")
    print(out["mse"])
    print("improvement from the start (%):")
    print(((out["mse"] - out["mse"][0]) / out["mse"][0] * 100).round(2))

    return model, instance_data, best_solution


if __name__ == "__main__":
    import doctest

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
