# -*- coding: utf-8 -*-

"""Iterative heuristic attack which poisons both numerical and categorical data"""

import copy

import numpy as np

import categorical_attack
import numerical_attack
import pyomo_model

long_space = 80
short_space = 60
middle_space = long_space


def run(config, instance_data, model=None):
    """Run iterative attack which poisons both numerica and categorical data

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

    n_epochs = config["iterative_attack_n_epochs"]
    numerical_attack_n_epochs = config["numerical_attack_n_epochs"]

    # Solve benchmark
    config["numerical_attack_incremental"] = True
    config["numerical_attack_n_epochs"] = 1
    _, _, benchmark_solution = numerical_attack.run(config, instance_data)
    config["numerical_attack_incremental"] = False
    config["numerical_attack_n_epochs"] = numerical_attack_n_epochs
    numerical_model = None
    categorical_model = None
    for epoch in range(n_epochs):
        if config["categorical_attack_no_cfeatures"] == 0:
            config["solver_name"] = "ipopt"
            numerical_model, instance_data, solution = numerical_attack.run(
                config, instance_data, numerical_model
            )
        for feature in instance_data.chosen_categorical_feature_names:
            config["solver_name"] = "ipopt"
            numerical_model, instance_data, solution = numerical_attack.run(
                config, instance_data, numerical_model
            )

            # Run categorical attack TODO improve categorical attack
            config["solver_name"] = "gurobi"
            categorical_model, instance_data, solution = categorical_attack.run(
                config, instance_data, categorical_model, features=feature
            )

    # TODO printing of solutions
    print("RESULTS")
    print(f'Benchmark mse:  {benchmark_solution["mse"]:7.5f}')
    print(f'Our method mse: {solution["mse"]:7.5f}')
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
