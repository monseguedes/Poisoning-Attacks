"""Run attack of aeverything"""

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


def run(config, instance_data, model=None, solver=None):
    """Run solver to optimize all features

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
    print("FULL ATTACK SOLVER")
    print("*" * long_space)
    print("*" * long_space)

    if not solver == None:
        config["solver_name"] = "gurobi"
    else:
        config["solver_name"] = solver

    if model is None:
        model = pyomo_model.PyomoModel(instance_data, config)
    else:
        model.update_parameters(instance_data)

    no_poison_samples = instance_data.no_poison_samples

    F = model.POISON_DATA_FIXED
    O = model.POISON_DATA_OPTIMIZED
    R = model.POISON_DATA_REMOVED

    # Poison everything
    config["solver_output"] = True
    model = pyomo_model.PyomoModel(instance_data, config)
    num_feature_flag = O
    shape = (instance_data.no_poison_samples, instance_data.no_catfeatures)
    cat_feature_flag = np.full(shape, O)
    model.set_poison_data_status(
        instance_data, num_feature_flag, cat_feature_flag
    )
    start = timeit.timeit()
    model.solve()
    end = timeit.timeit()
    solution = model.get_solution()
    model.update_data(instance_data)

    return model, instance_data, solution


if __name__ == "__main__":
    import doctest

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
