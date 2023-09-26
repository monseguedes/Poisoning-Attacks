"""Run binary attack """

import copy
import timeit

import numpy as np
import pandas as pd

import numerical_attack
import pyomo_model
import ridge_regression
import testing

long_space = 80
short_space = 60
middle_space = long_space


def run(config, instance_data, warmstart_data=None, model=None):
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
    print("*" * long_space)
    print("BINARY ATTACK STRATEGY")
    print("*" * long_space)
    print("*" * long_space)

    if not config.get("solver_name"):
        config["solver_name"] = "gurobi"

    if model is None:
        model = pyomo_model.PyomoModel(instance_data, config)
    else:
        model.update_parameters(instance_data)

    no_poison_samples = instance_data.no_poison_samples

    F = model.POISON_DATA_FIXED
    O = model.POISON_DATA_OPTIMIZED
    R = model.POISON_DATA_REMOVED

    # Solve benchmark
    config["iterative_attack_incremental"] = True
    _, benchmark_data, benchmark_solution = numerical_attack.run(
        config, instance_data, model
    )

    # Poison everything
    config["binary"] = True
    config["solver_name"] = "gurobi"
    config["solver_output"] = True
    round_except_last = (
        lambda x: round(x, 0)
        if x.name != instance_data.poison_dataframe.columns[-1]
        else x
    )
    instance_data.poison_dataframe = instance_data.poison_dataframe.apply(
        round_except_last
    )
    model = pyomo_model.PyomoModel(instance_data, config)
    num_feature_flag = O
    shape = (instance_data.no_poison_samples, instance_data.no_catfeatures)
    cat_feature_flag = np.full(shape, O)
    model.set_poison_data_status(
        instance_data, num_feature_flag, cat_feature_flag
    )
    if warmstart_data is not None:
        pass
        # model.warmstart(warmstart_data)
    start = timeit.timeit()
    model.solve()
    end = timeit.timeit()
    solution = model.get_solution()
    model.update_data(instance_data)

    solution["mse_final"] = solution["mse"]
    solution["computational_time_final"] = end - start

    print("RESULTS")
    print(f'Benchmark mse:       {benchmark_solution["mse"]:7.6f}')
    print(f'Flipping method mse: {solution["mse"]:7.6f}')
    print(
        f'Improvement:         {(solution["mse"] - benchmark_solution["mse"]) / benchmark_solution["mse"] * 100:7.6f}'
    )

    return model, instance_data, solution


if __name__ == "__main__":
    import doctest
    from programs.minlp.unused_scripts.main import config
    import instance_data_class

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

    instance_data = instance_data_class.InstanceData(config)

    numerical_model = None

    numerical_model, instance_data, regression_parameters = run(
        config, instance_data, numerical_model
    )

# vimquickrun: python % && ./vimquickrun.sh
