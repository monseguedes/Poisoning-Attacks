# -*- coding: utf-8 -*-

"""Run iterative attack which which poison training data row by row"""

import numpy as np
import pyomo.environ as pyo
import instance_data_class
import pyomo_model

# TODO Refactor and simplify function calls around model building.
# TODO Improve efficiency by avoid calling unnecesary instance_data.get_x.

long_space = 80
short_space = 60
middle_space = long_space


# TODO Modify this function to take instance_data and pyomo model as arguments.
def run(config):
    """Run iterative attack which which poison training data row by row"""


    # Solve benchmark
    opt = pyo.SolverFactory("ipopt")

    print("" * 2)
    print("*" * long_space)
    print("ITERATIVE CONTINUOUS NONLINEAR ALGORITHM")
    print("*" * long_space)

    print("Building data class")
    instance_data = instance_data_class.InstanceData(config)

    (
        benchmark_model,
        benchmark_instance,
        benchmark_solution,
    ) = iterative_attack_strategy(opt=opt, instance_data=instance_data, config=config)
    print("*" * middle_space)

    return benchmark_model, benchmark_instance, benchmark_solution


def iterative_attack_strategy(opt: pyo.SolverFactory, instance_data, config):
    """
    Algorithm for iterative attack strategy.

    It starts by creating the abstract model, and an initial data object for
    creating the first instance. After this, while the iteration count is
    smaller than the number of subsets (there is an iteration per subset), the
    model instance is created with the intance data object and the model is
    solved for current instance. After that, solutions are stored in a
    dataframe, and data object for instance is updated to that current
    iteration becomes data. Then, we go back to start of while loop and process
    is repeated for all subsets/iterations.
    """

    print("" * 2)
    print("*" * long_space)
    print("ITERATIVE ATTACK STRATEGY")
    print("*" * long_space)

    model = pyomo_model.IterativeAttackModel(instance_data, config["function"])

    n_epochs = config["iterative_attack_n_epochs"]
    mini_batch_size = config["iterative_attack_mini_batch_size"]

    incremental = config["iterative_attack_incremental"]

    if incremental:
        if n_epochs > 1:
            raise ValueError(
                f"n_epochs should be 1 when incremental but got {n_epochs}"
            )

    no_poison_samples = instance_data.no_poison_samples

    if mini_batch_size > 1:
        mini_batch_absolute_size = mini_batch_size
    else:
        # mini batch size is specified as a fraction
        mini_batch_absolute_size = max(int(no_poison_samples * mini_batch_size), 1)
    breaks = np.arange(0, no_poison_samples, mini_batch_absolute_size)
    breaks = np.r_[breaks, no_poison_samples]
    n_mini_batches = len(breaks) - 1

    solution_list = []

    for epoch in range(n_epochs):
        for mini_batch_index in range(n_mini_batches):
            # Modify flag to specify which one to remove.
            # model.unfix: 0
            # model.fix: 1
            # model.remove: 2
            flag = np.full(instance_data.no_poison_samples, -1)
            flag[: breaks[mini_batch_index]] = model.POISON_DATA_FIXED
            flag[
                breaks[mini_batch_index] : breaks[mini_batch_index + 1]
            ] = model.POISON_DATA_OPTIMIZED
            if incremental:
                flag[breaks[mini_batch_index + 1] :] = model.POISON_DATA_REMOVED
            else:
                flag[breaks[mini_batch_index + 1] :] = model.POISON_DATA_FIXED
            model.fix_rows_in_poison_dataframe(instance_data, flag)
            opt.solve(model, load_solutions=True, tee=False)
            solution = model.get_solution()
            instance_data.update_numerical_features(solution["optimized_x_poison_num"])
            solution_list.append(solution)
            if (epoch * n_mini_batches + mini_batch_index) % 20 == 0:
                print(f"{'epoch':>5s}  " f"{'batch':>5s}  " f"{'mse':>9s}")
            print(f"{epoch:5d}  " f"{mini_batch_index:5d}  " f"{solution['mse']:9.6f}")

    # This will break when solution_list is empty, but maybe it's unlikely
    keys = solution_list[0].keys()
    out = {key: np.stack([x[key] for x in solution_list]) for key in keys}

    print("mse in each iteration:")
    print(out["mse"])
    print("improvement from the start (%):")
    print(((out["mse"] - out["mse"][0]) / out["mse"][0] * 100).round(2))

    return model, instance_data, solution


if __name__ == "__main__":
    import doctest

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
