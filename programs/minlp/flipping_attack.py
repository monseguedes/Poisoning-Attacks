# -*- coding: utf-8 -*-

"""Flipping heuristic attack which poisons both numerical and categorical data"""

import copy

import categorical_attack
import numerical_attack
import numpy as np
import pyomo_model
import ridge_regression

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

    # TODO we use both solvers. but ipopts first.
    if not config.get("solver_name"):
        config["solver_name"] = "ipopt"
    np.testing.assert_equal(config["solver_name"], "ipopt")

    if model is None:
        model = pyomo_model.PyomoModel(instance_data, config)
    else:
        model.update_parameters(instance_data)

    n_epochs = config["flipping_attack_n_epochs"]

    no_poison_samples = instance_data.no_poison_samples

    # Solve benchmark
    config["iterative_attack_incremental"] = True
    _, _, benchmark_solution = numerical_attack.run(config, instance_data)
    config["iterative_attack_incremental"] = False

    for epoch in range(n_epochs):
        config["solver_name"] = "ipopt"
        numerical_model = None
        numerical_model, instance_data, solution = numerical_attack.run(
            config, instance_data, numerical_model
        )

        # Store the best solution found so far.
        best_sol = solution
        # And the instance data to achieve this best solution.
        best_instance_data = instance_data.copy()

        for poison_sample_index in range(no_poison_samples):
            # Make (just num) prediction
            cat_weights = best_sol["weights_cat"].to_dict()
            print(cat_weights)
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
            target_y = instance_data.get_y_poison_dataframe().iloc[poison_sample_index]
            difference = num_y - target_y

            # We consider two case: Make prediction as large as possible and make prediction
            # as small as possible. We then take the best one.

            # categories_up/down[feature] is the category to push prediction up/down.
            cat_features = instance_data.categorical_feature_category_tuples
            categories_up = dict()
            categories_down = dict()
            for feature in [cat_feature[0] for cat_feature in cat_features]:
                # Filter the keys based on given values for first two elements
                filtered_keys = [k for k in cat_weights.keys() if k[0] == feature]
                categories_up[feature] = max(filtered_keys, key=cat_weights.get)[1]
                categories_down[feature] = min(filtered_keys, key=cat_weights.get)[1]

            # Let's compute the prediction of each case.
            # TODO Why is this just the sum of weights?
            # TODO Why do we get key error when key exists? type?
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
            # TODO add printing
            # Check if the updated data was better than the current best.
            if best_sol["mse"] > sol["mse"]:
                # The current data is actually worse than the current best.
                # Revert the change.
                instance_data = best_instance_data.copy()
            else:
                # We found a better one than the current best.
                best_sol = sol  # TODO make sure we use new weights (which are already computed)
                best_instance_data = instance_data.copy()

    # TODO printing of solutions
    print("RESULTS")
    print(f'Benchmark mse:       {benchmark_solution["mse"]:7.4f}')
    print(f'Flipping method mse: {solution["mse"]:7.4f}')
    print(
        f'Improvement:         {(solution["mse"] - benchmark_solution["mse"]) / benchmark_solution["mse"] * 100:7.4f}'
    )

    # TODO what do we do with model
    return model, instance_data, solution


def flip_row():
    # TODO fill
    raise NotImplementedError


if __name__ == "__main__":
    import doctest

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
