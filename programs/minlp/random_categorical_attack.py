# -*- coding: utf-8 -*-

"""Randomly perturb categorical features"""

import numpy as np
import ridge_regression


def run(config, instance_data):
    """Randomly perturb categorical features"""
    rng = np.random.RandomState(0)

    # Store the best solution found so far.
    best_sol = ridge_regression.run(config, instance_data)
    # And the instance data to achieve this best solution.
    best_instance_data = instance_data.copy()

    # A list of all the pairs of (categorical_feature, categories).
    feat_cat_list = instance_data.categorical_feature_category_tuples

    for iteration in range(100):
        sample = rng.randint(instance_data.no_poison_samples)
        # Purturbe 20 categorical features.
        for j in range(20):
            # Randomly choose a pair of categorical feature and category
            # and set the data to this sampled value.
            feat, cat = feat_cat_list[rng.choice(len(feat_cat_list))]
            instance_data.cat_poison[sample, feat] = cat
        # Run the regression and see if the purturbation was effective or not.
        sol = ridge_regression.run(config, instance_data)
        if iteration % 20 == 0:
            print(f"{'it':>3s}  {'mes':>9s}  {'best':>9s}")
        print(f"{iteration:3d}  {sol['mse']:9.6f}  {best_sol['mse']:9.6f}")
        # Check if the updated data was better than the current best.
        if best_sol['mse'] > sol['mse']:
            # The current data is actually worse than the current best.
            # Revert the change.
            instance_data = best_instance_data.copy()
        else:
            # We found a better one than the current best.
            best_sol = sol
            best_instance_data = instance_data

    return None, best_instance_data, best_sol
