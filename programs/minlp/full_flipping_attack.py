
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
    print("FULL FLIPPING ATTACK HEURISTIC")
    print("*" * long_space)

    n_epochs = config["flipping_attack_n_epochs"]

    no_poison_samples = instance_data.no_poison_samples

    # Solve benchmark
    config["iterative_attack_incremental"] = True
    _, benchmark_data, benchmark_solution = numerical_attack.run(config, instance_data)
    config["iterative_attack_incremental"] = False
    numerical_model = None

    for epoch in range(n_epochs):
        sol = ridge_regression.run(config, instance_data)
        # Save poisoning samples after numerical attack
        if (epoch == 0):
            # Store the best solution found so far.
            best_sol = sol
            best_instance_data = instance_data

        for poison_sample_index in range(no_poison_samples):
            print(poison_sample_index)
            # Make (just num) prediction
            cat_weights = best_sol["weights_cat"].to_dict()
            num_weights = best_sol["weights_num"].to_dict()

            target_y = instance_data.get_y_poison_dataframe().iloc[poison_sample_index]

            # We consider two case: Make prediction as large as possible and make prediction
            # as small as possible. We then take the best one.
            # categories_up/down[feature] is the category to push prediction up/down.

            # categories_up/down[feature] is the category to push prediction up/down.
            cat_features = set([cat_feature[0] for cat_feature in cat_weights.keys()])
            categories_up = dict()
            categories_down = dict()
            for feature in cat_features:
                # Filter the keys based on given values for first two elements
                filtered_keys = [k for k in cat_weights.keys() if k[0] == feature]
                categories_up[feature] = max(filtered_keys, key=cat_weights.get)[1]
                categories_down[feature] = min(filtered_keys, key=cat_weights.get)[1]

            # Let's compute the prediction of each case.
            pred_up_numerical = sum(
                num_weights[feature]
                for feature in cat_features if num_weights[feature] > 0
            )
            pred_up_categorical = sum(
                cat_weights[(feature, categories_up[feature])]
                for feature in cat_features
            )
            pred_up = pred_up_numerical + pred_up_categorical

            pred_down_numerical = sum(
                num_weights[feature]
                for feature in cat_features if num_weights[feature] < 0
            )
            pred_down_categorical = sum(
                cat_weights[(feature, categories_down[feature])]
                for feature in cat_features
            )
            pred_down = pred_down_numerical + pred_down_categorical

            negative_num_weights = [key for key, value in num_weights.items() if value < 0]
            positive_num_weights = [key for key, value in num_weights.items() if value > 0]

            print(instance_data.poison_dataframe)
            for f in num_weights.keys():
                instance_data.num_poison[poison_sample_index, f] = 0
            print(instance_data.poison_dataframe)
            if np.abs(pred_up - target_y) < np.abs(pred_down - target_y):
                # Pushing down is more effective.
                categories_chosen = categories_down
                for k in negative_num_weights:
                    instance_data.num_poison[poison_sample_index, k] = 1
            else:
                # Pushing up is more effective.
                categories_chosen = categories_up
                for k in positive_num_weights:
                    instance_data.num_poison[poison_sample_index, k] = 1
            
            print(instance_data.poison_dataframe)

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


    print("RESULTS")
    print(f'Benchmark mse:       {benchmark_solution["mse"]:7.6f}')
    print(f'Flipping method mse: {best_sol["mse"]:7.6f}')
    print(
        f'Improvement:         {(best_sol["mse"] - benchmark_solution["mse"]) / benchmark_solution["mse"] * 100:7.6f}'
    )

    return model, best_instance_data, best_sol


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

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
