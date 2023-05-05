# -*- coding: utf-8 -*-

"""Testing routines"""

import numpy as np

import ridge_regression


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

    msg = []

    for key in ["weights_num", "weights_cat", "bias", "mse"]:
        a = flatten(sol1[key])
        b = flatten(sol2[key])

        try:
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=key)
        except AssertionError as e:
            # print(e)
            msg.append(str(e))

    if msg:
        raise AssertionError(("\n" + "- " * 30).join(msg))


def validate_solution(config, instance_data, solution):
    scikit_learn_solution = ridge_regression.run(config, instance_data)
    assert_solutions_are_close(solution, scikit_learn_solution)


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
