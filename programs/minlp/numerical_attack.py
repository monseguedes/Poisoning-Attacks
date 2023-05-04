# -*- coding: utf-8 -*-

"""Run iterative attack which which poison training data row by row"""

import copy
import pandas as pd

import numpy as np
import pyomo_model
import ridge_regression

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

    if isinstance(mini_batch_size, int): # TODO what happens for the case with just one batch?
        mini_batch_absolute_size = mini_batch_size
    elif mini_batch_size == 'all':
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

    for epoch in range(n_epochs):
        print(f'Starting epoch {epoch}')
        for mini_batch_index in range(n_mini_batches):
            print(f'Starting minibatch {mini_batch_index}')
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
            run_test(config, best_instance_data, best_solution)
            print('Solving model')
            model.solve()
            print('Testing after solve')
            run_test(config, best_instance_data, best_solution)
            solution = model.get_solution()
            print('Testing after solve')
            run_test(config, best_instance_data, best_solution)
            if solution["mse"] > best_mse:
                buff = instance_data.copy()
                model.update_data(instance_data)
                print('Testing solution')
                run_test(config, instance_data, solution)
                best_mse = solution["mse"]
                best_solution = solution
                best_instance_data = instance_data.copy()
                print('Best solution updated')
                run_test(config, best_instance_data, best_solution)
            
            else:
                instance_data = best_instance_data.copy()

            solution_list.append(solution)
            if (epoch * n_mini_batches + mini_batch_index) % 20 == 0:
                print(f"{'epoch':>5s}  {'batch':>5s}  {'mse':>9s}  {'best_mse':>9s}")
            print(f"{epoch:5d}  {mini_batch_index:5d}  {solution['mse']:9.6f}  {best_mse:9.6f}")
            run_test(config, best_instance_data, best_solution)

    # This will break when solution_list is empty, but maybe it's unlikely
    keys = solution_list[0].keys()
    out = {key: np.stack([x[key] for x in solution_list]) for key in keys}

    print("mse in each iteration:")
    print(out["mse"])
    print("improvement from the start (%):")
    print(((out["mse"] - out["mse"][0]) / out["mse"][0] * 100).round(2))

    return model, best_instance_data, best_solution

# Run the utitlity to check the results with scikitlearn.
def run_test(config, instance_data, solution):
    scikit_learn_regression_parameters = ridge_regression.run(config, instance_data)

    def assert_solutions_are_close(sol1, sol2):
        def flatten(x):
            try:
                x = x.to_numpy()
            except AttributeError:
                pass
            try:
                return x.ravel()
            except AttributeError:
                return x
            
        failed = []
        for key in ["weights_num", "weights_cat", "bias", "mse"]:
            a = flatten(sol1[key])
            b = flatten(sol2[key])
            if not np.allclose(a, b, rtol=1e-4, atol=1e-4):
                failed.append(key)
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=key)

        if failed:
            raise AssertionError(f'Failed on value {",".join(failed)}')
    
    assert_solutions_are_close(solution, scikit_learn_regression_parameters)

def print_diff(instance_data_a, instance_data_b):
    print(
        instance_data_a.get_num_x_train_dataframe()[
            instance_data_a.get_num_x_train_dataframe()
            != instance_data_b.get_num_x_train_dataframe()
        ]
    )
    print(
        instance_data_a.get_cat_x_train_dataframe()[
            instance_data_a.get_cat_x_train_dataframe()
            != instance_data_b.get_cat_x_train_dataframe()
        ]
    )
    print(
        instance_data_a.get_num_x_poison_dataframe()[
            instance_data_a.get_num_x_poison_dataframe()
            != instance_data_b.get_num_x_poison_dataframe()
        ]
    )
    print(
        instance_data_a.get_cat_x_poison_dataframe()[
            instance_data_a.get_cat_x_poison_dataframe()
            != instance_data_b.get_cat_x_poison_dataframe()
        ]
    )

if __name__ == "__main__":
    import doctest

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
