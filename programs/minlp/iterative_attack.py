# -*- coding: utf-8 -*-

"""Iterative heuristic attack which poisons both numerical and categorical data"""

import copy

import categorical_attack
import numerical_attack
import numpy as np
import pyomo_model

long_space = 80
short_space = 60
middle_space = long_space


def run(config, instance_data, model=None):
    """Run categorical attack which poison categorical data

    This is a hueristic to poison features using a combunation of
    locally optimising numerical features using ipots and 'optimising'
    categorical ones using gurobi.
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
    print("ITERATIVE ATTACK HEURISTIC")
    print("*" * long_space)

    # TODO we use both solvers. but ipopts first.
    if not config.get("solver_name"):
        config["solver_name"] = "ipopt"
    np.testing.assert_equal(config["solver_name"], "ipopt")

    if model is None:
        model = pyomo_model.PyomoModel(instance_data, config)
    else:
        model.update_parameters(instance_data)

    n_epochs = config["iterative_attack_n_epochs"]

    # Solve benchmark
    config["iterative_attack_incremental"] = True
    _, _, benchmark_solution = numerical_attack.run(config, instance_data)

    config["iterative_attack_incremental"] = False
    for epoch in range(n_epochs):
        for feature in instance_data.chosen_categorical_feature_names:
            config["solver_name"] = "ipopt"
            numerical_model = None
            numerical_model, instance_data, solution = numerical_attack.run(
                config, instance_data, numerical_model
            )

            # Run categorical attack TODO improve categorical attack
            config["solver_name"] = "gurobi"
            categorical_model = None
            categorical_model, instance_data, solution = categorical_attack.run(
                config, instance_data, categorical_model, features=feature
            )

    # TODO printing of solutions
    print("RESULTS")
    print(f'Benchmark mse:  {benchmark_solution["mse"]:7.4f}')
    print(f'Our method mse: {solution["mse"]:7.4f}')
    print(
        f'Improvement:    {(solution["mse"] - benchmark_solution["mse"]) / benchmark_solution["mse"] * 100:7.4f}'
    )

    # TODO what do we do with model
    return model, instance_data, solution


if __name__ == "__main__":
    import doctest

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
