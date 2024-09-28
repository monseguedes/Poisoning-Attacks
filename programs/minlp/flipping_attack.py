# -*- coding: utf-8 -*-

"""Flipping heuristic attack which poisons both numerical and categorical data"""

import copy
import time

import numpy as np
import pandas as pd
import yaml
# import timeit
import time

import numerical_attack
import pyomo_model
import ridge_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import instance_data_class

long_space = 80
short_space = 60
middle_space = long_space


def run(config, instance_data, model=None):
    """Run flipping attack which poisons both numerica and categorical data

    This is a hueristic to poison features using a combination of
    locally optimising numerical features using ipots and flipping
    categorical features to push mse in a specific direction.
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
    print("FLIPPING ATTACK HEURISTIC")
    print("*" * long_space)

    if not config.get("solver_name"):
        raise NameError("solver_name not set in config")

    n_epochs = config["flipping_attack_n_epochs"]
    no_poison_samples = instance_data.no_poison_samples

    # Run ridge regression on the unpoisoned data---------------------------
    regression_parameters = ridge_regression.run_not_poisoned(
        config, instance_data, data_type="train"
    )
    regression_parameters_test = ridge_regression.run_not_poisoned(
        config, instance_data, data_type="test"
    )
    regression_parameters_validation = ridge_regression.run_not_poisoned(
        config, instance_data, data_type="validation"
    )

    # Solve benchmark-------------------------------------------------------
    benchmark_start = time.time()
    config["iterative_attack_incremental"] = True
    config["bounding"] = False
    _, benchmark_data, benchmark_solution = numerical_attack.run(
        config, instance_data
    )
    config["iterative_attack_incremental"] = False
    config["bounding"] = False
    benchmark_end = time.time()

    # Start flipping attack-------------------------------------------------
    numerical_model = model  # Reset model to None
    start = time.time()
    for epoch in range(n_epochs):
        (
            numerical_model,
            numerical_attack_instance_data,
            solution,
        ) = numerical_attack.run(config, instance_data, numerical_model)
        if (epoch == 0) or (best_sol["mse"] <= solution["mse"]):
            # Store the best solution found so far.
            best_sol = solution
            # And the instance data to achieve this best solution.
            best_instance_data = numerical_attack_instance_data
            instance_data = numerical_attack_instance_data
            best_model = numerical_model
        else:
            instance_data = best_instance_data.copy()

        mse_iteration_array = []
        time_iteration_array = []
        for poison_sample_index in range(no_poison_samples):
            # Make (just num) prediction
            cat_weights = best_sol["weights_cat"].to_dict()
            num_weights = best_sol["weights_num"].to_dict()

            num_features = {
                k: v
                for k, v in best_sol["x_poison_num"].items()
                if k[0] == poison_sample_index
            }
            num_y = (
                np.array(list(num_weights.values()))
                @ np.array(list(num_features.values()))
                + best_sol["bias"]
            )
            target_y = instance_data.get_y_poison_dataframe().iloc[
                poison_sample_index
            ]
            difference = num_y - target_y

            # We consider two case: Make prediction as large as possible and make prediction
            # as small as possible. We then take the best one.

            # categories_up/down[feature] is the category to push prediction up/down.
            # cat_features = instance_data.categorical_feature_category_tuples
            cat_features = set(
                [cat_feature[0] for cat_feature in cat_weights.keys()]
            )
            categories_up = dict()
            categories_down = dict()
            for feature in cat_features:
                # Filter the keys based on given values for first two elements
                filtered_keys = [
                    k for k in cat_weights.keys() if k[0] == feature
                ]
                categories_up[feature] = max(
                    filtered_keys, key=cat_weights.get
                )[1]
                categories_down[feature] = min(
                    filtered_keys, key=cat_weights.get
                )[1]

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

            # Update the dataframe.
            for feat, cat in categories_chosen.items():
                instance_data.cat_poison[poison_sample_index, feat] = cat

            # Run the regression and see if the purturbation was effective or not.
            sol = ridge_regression.run(config, instance_data)

            if poison_sample_index % 20 == 0:
                print(f"{'it':>3s}  {'mse':>9s}  {'best':>9s}")
            print(
                f"{poison_sample_index:3d}  {sol['mse']:9.6f}  {best_sol['mse']:9.6f}"
            )

            # Check if the updated data was better than the current best.
            if best_sol["mse"] > sol["mse"]:
                # The current data is actually worse than the current best.
                # Revert the change.
                instance_data = best_instance_data.copy()
            else:
                # We found a better one than the current best.
                best_sol = sol
                best_instance_data = instance_data.copy()

            mse_iteration_array.append(best_sol["mse"])

        # config["numerical_attack_mini_batch_size"] = 0.5
        (
            numerical_model,
            numerical_attack_instance_data,
            solution,
        ) = numerical_attack.run(config, instance_data, numerical_model)
        if best_sol["mse"] <= solution["mse"]:
            # Store the best solution found so far.
            best_sol = solution
            # And the instance data to achieve this best solution.
            best_instance_data = numerical_attack_instance_data
            instance_data = numerical_attack_instance_data
        else:
            instance_data = best_instance_data.copy()

        # # Project numerical features
        # round_except_last = lambda x: round(x, 0) if x.name != best_instance_data.poison_dataframe.columns[-1] else x
        # best_instance_data.poison_dataframe = best_instance_data.poison_dataframe.apply(round_except_last)
        # best_sol = ridge_regression.run(config, instance_data)

    end = time.time()

    # Get test and validation errors.
    validation_benchmark_error = mean_squared_error(
        instance_data.get_y_validation_dataframe().to_numpy(),
        make_predictions(
            "validation",
            benchmark_data,
            benchmark_solution["weights_num"],
            benchmark_solution["weights_cat"],
            benchmark_solution["bias"],
        ),
    )
    test_benchmark_error = mean_squared_error(
        instance_data.get_y_test_dataframe().to_numpy(),
        make_predictions(
            "test",
            benchmark_data,
            benchmark_solution["weights_num"],
            benchmark_solution["weights_cat"],
            benchmark_solution["bias"],
        ),
    )
    validation_flipping_error = mean_squared_error(
        instance_data.get_y_validation_dataframe().to_numpy(),
        make_predictions(
            "validation",
            best_instance_data,
            best_sol["weights_num"],
            best_sol["weights_cat"],
            best_sol["bias"],
        ),
    )
    test_flipping_error = mean_squared_error(
        instance_data.get_y_test_dataframe().to_numpy(),
        make_predictions(
            "test",
            best_instance_data,
            best_sol["weights_num"],
            best_sol["weights_cat"],
            best_sol["bias"],
        ),
    )

    print("RESULTS")
    print("*" * short_space)
    print(
        f'Unpoisoned mse validation:      {regression_parameters_validation["mse"]:7.6f}'
    )
    print(
        f'Unpoisoned mse test:            {regression_parameters_test["mse"]:7.6f}'
    )
    print("*" * short_space)
    print(f'Benchmark mse:                   {benchmark_solution["mse"]:7.6f}')
    print(
        f"Benchmark mse validation:        {validation_benchmark_error:7.6f}"
    )
    print(f"Benchmark mse test:              {test_benchmark_error:7.6f}")
    print("*" * short_space)
    print(f'Flipping method mse:             {best_sol["mse"]:7.6f}')
    print(f"Flipping method mse validation:  {validation_flipping_error:7.6f}")
    print(f"Flipping method mse test:        {test_flipping_error:7.6f}")
    print("*" * short_space)
    print(
        f'Improvement:                     {(best_sol["mse"] - benchmark_solution["mse"]) / benchmark_solution["mse"] * 100:7.6f}'
    )
    print(
        f"Improvement validation:          {(validation_flipping_error - validation_benchmark_error) / validation_benchmark_error * 100:7.6f}"
    )
    print(
        f"Improvement test:                {(test_flipping_error - test_benchmark_error) / test_benchmark_error * 100:7.6f}"
    )
    print("*" * short_space)
    print(
        f"Benchmark computation time:      {benchmark_end - benchmark_start:7.6f}"
    )
    print(f"Flipping computation time:     {end - start:7.6f}")

    # Save results as dictionary
    results_dict = {
        "unpoisoned_validation_mse": regression_parameters_validation["mse"],
        "unpoisoned_test_mse": regression_parameters_test["mse"],
        "benchmark_mse": benchmark_solution["mse"],
        "benchmark_validation_mse": validation_benchmark_error,
        "benchmark_test_mse": test_benchmark_error,
        "benchmark_parameters": benchmark_solution,
        "flipping_mse": best_sol["mse"],
        "flipping_validation_mse": validation_flipping_error,
        "flipping_test_mse": test_flipping_error,
        "flipping_parameters": best_sol,
        "benchmark_time": (benchmark_end - benchmark_start),
        "flipping_time": (end - start),
    }

    # Save results as dict using numpy
    np.save(
        f"programs/minlp/results/{config['seed']}_{config['poison_rate']}_{config['numerical_attack_mini_batch_size']}_bilevel_results.npy",
        results_dict,
    )

    # TODO: should I add new items to best_sol or should I return new dictionary?
    # best_sol["mse_per_iteration"] = mse_iteration_array
    # best_sol["mse_final"] = best_sol["mse"]
    # best_sol["computational_time_final"] = end - start
    # best_sol["benchmark_mse_final"] = benchmark_solution["mse"]
    # best_sol["benchmark_computational_time"] = benchmark_end - benchmark_start

    return best_model, best_instance_data, best_sol


def make_predictions(
    data_type: str,
    instance_data: instance_data_class.InstanceData,
    numerical_weights: np.ndarray,
    categorical_weights: np.ndarray,
    bias: float,
):
    """
    Take the regression coefficents given by solving the nonpoisoned model
    and use them to make predictions on training dataset.
    """

    if data_type == "test":
        X_cat = instance_data.get_cat_x_test_dataframe(wide=True).to_numpy()
        X_num = instance_data.get_num_x_test_dataframe(wide=True).to_numpy()
        X = np.concatenate((X_num, X_cat), axis=1)

    if data_type == "train":
        X_cat = instance_data.get_cat_x_train_dataframe(wide=True).to_numpy()
        X_num = instance_data.get_num_x_train_dataframe(wide=True).to_numpy()
        X = np.concatenate((X_num, X_cat), axis=1)

    if data_type == "validation":
        X_cat = instance_data.get_cat_x_validation_dataframe(
            wide=True
        ).to_numpy()
        X_num = instance_data.get_num_x_validation_dataframe(
            wide=True
        ).to_numpy()
        X = np.concatenate((X_num, X_cat), axis=1)

    weights = np.concatenate((numerical_weights, categorical_weights))

    # Make predictions
    predictions = np.dot(X, weights) + bias

    return predictions


def save_dataframes(instance, solution, config, it):
    # Save poisoning samples after numerical attack
    instance.poison_dataframe.to_csv(
        "programs/minlp/attacks/{}/poison_dataframe{}.csv".format(
            config["dataset_name"], it
        )
    )
    solution["weights_num"].to_csv(
        "programs/minlp/attacks/{}/numerical_weights{}.csv".format(
            config["dataset_name"], it
        )
    )
    solution["weights_cat"].to_csv(
        "programs/minlp/attacks/{}/categorical_weights{}.csv".format(
            config["dataset_name"], it
        )
    )


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


# Run the utitlity to check the results with scikitlearn.
def run_test(config, instance_data, solution):
    scikit_learn_regression_parameters = ridge_regression.run(
        config, instance_data
    )

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
            if not np.allclose(a, b, rtol=1e-4):
                failed.append(key)
            # np.testing.assert_allclose(a, b, rtol=1e-4, err_msg=key)

        if failed:
            raise AssertionError(f'Failed on value {",".join(failed)}')

    assert_solutions_are_close(solution, scikit_learn_regression_parameters)


def flip_row():
    # TODO fill
    raise NotImplementedError


if __name__ == "__main__":
    import doctest
    import instance_data_class

    with open("programs/minlp/config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

    seed = 2

    instance_data = instance_data_class.InstanceData(
        config, benchmark_data=True, seed=seed
    )
    numerical_model = None

    _, instance_data, regression_parameters = run(
        config, instance_data, numerical_model
    )

    # Print gradient results
    # Load npy file
    dictionary = np.load(
        f"programs/benchmark/manip-ml-master/poisoning/results/{seed}_60_gradient_results.npy",
        allow_pickle=True,
    )
    # Print table with gradient results
    print("Gradient results")
    print("*" * middle_space)
    print(
        f'Unpoisoned mse validation:       {dictionary.item()["unpoisoned_validation_mse"]:7.6f}'
    )
    print(
        f'Unpoisoned mse test:             {dictionary.item()["unpoisoned_test_mse"]:7.6f}'
    )
    print("*" * middle_space)
    print(
        f"Gradient mse validation:         {dictionary.item()['poisoned_validation_mse']:7.6f}"
    )
    print(
        f"Gradient mse test:               {dictionary.item()['poisoned_test_mse']:7.6f}"
    )
    print("*" * middle_space)
    print(
        f"Computation time:                {dictionary.item()['compute_time']:7.6f}"
    )

    # # Run the utitlity to check the results with scikitlearn
    # scikit_learn_regression_parameters = ridge_regression.run(config, instance_data)

    # testing.assert_solutions_are_close(
    #     regression_parameters, scikit_learn_regression_parameters
    # )
    # print("test passed")

    # vimquickrun: python % && ./vimquickrun.sh
