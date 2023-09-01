# -*- coding: utf-8 -*-

"""Ridge regression without poisoning"""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def run(config, instance_data, wide=False):
    """Run ridge regression without poisoning"""
    num_dataframe = pd.concat(
        [
            instance_data.get_num_x_train_dataframe(wide=True),
            instance_data.get_num_x_poison_dataframe(wide=True),
        ]
    )
    cat_dataframe = pd.concat(
        [
            instance_data.get_cat_x_train_dataframe(wide=True),
            instance_data.get_cat_x_poison_dataframe(wide=True),
        ]
    )
    X_df = pd.concat([num_dataframe, cat_dataframe], axis=1)
    X = X_df.to_numpy()
    y_df = pd.concat(
        [
            instance_data.get_y_train_dataframe(),
            instance_data.get_y_poison_dataframe(),
        ]
    )
    y = y_df.to_numpy()
    model = Ridge(
        alpha=len(X) * config["regularization"],
        fit_intercept=True,
        solver="svd",
    )
    model.fit(X, y)

    weights_num = model.coef_[: instance_data.no_numfeatures]
    weights_cat = model.coef_[instance_data.no_numfeatures :]
    bias = model.intercept_

    if not wide:
        # To make long format dataframes.
        _weights_num = pd.Series()
        for k, v in zip(instance_data.numerical_feature_names, weights_num):
            _weights_num[k] = v
        index = pd.MultiIndex(
            levels=[[], []], codes=[[], []], names=["feature", "category"]
        )
        _weights_cat = pd.Series(index=index)
        iter = zip(
            instance_data.categorical_feature_category_tuples, weights_cat
        )
        for k, v in iter:
            _weights_cat.loc[k] = v
    else:
        # To make wide fromat dataframes.
        _weights_num = pd.DataFrame()
        for k, v in zip(instance_data.numerical_feature_names, weights_num):
            _weights_num[k] = [v]
        _weights_cat = pd.DataFrame()
        iter = zip(
            instance_data.categorical_feature_category_tuples, weights_cat
        )
        for k, v in iter:
            column = f"{k[0]}:{k[1]}"
            _weights_cat[column] = [v]

    y_pred = model.predict(X[: instance_data.no_train_samples])
    mse = mean_squared_error(y[: instance_data.no_train_samples], y_pred)

    print(f"MSE: {mse}")

    return {
        "weights_num": _weights_num,
        "weights_cat": _weights_cat,
        "bias": bias,
        "mse": mse,
        "x_poison_num": instance_data.get_num_x_poison_dataframe(),
        "x_poison_cat": instance_data.get_cat_x_poison_dataframe(),
    }


def run_not_poisoned(config, instance_data, data_type="train", wide=False):
    """Run ridge regression without poisoning"""
    num_dataframe = instance_data.get_num_x_train_dataframe(wide=True)
    cat_dataframe = instance_data.get_cat_x_train_dataframe(wide=True)
    y_df = instance_data.get_y_train_dataframe()
    X_df = pd.concat([num_dataframe, cat_dataframe], axis=1)
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    model = Ridge(
        alpha=len(X) * config["regularization"],
        fit_intercept=True,
        solver="svd",
    )
    model.fit(X, y)

    weights_num = model.coef_[: instance_data.no_numfeatures]
    weights_cat = model.coef_[instance_data.no_numfeatures :]
    bias = model.intercept_

    if not wide:
        # To make long format dataframes.
        _weights_num = pd.Series()
        for k, v in zip(instance_data.numerical_feature_names, weights_num):
            _weights_num[k] = v
        index = pd.MultiIndex(
            levels=[[], []], codes=[[], []], names=["feature", "category"]
        )
        _weights_cat = pd.Series(index=index)
        iter = zip(
            instance_data.categorical_feature_category_tuples, weights_cat
        )
        for k, v in iter:
            _weights_cat.loc[k] = v
    else:
        # To make wide fromat dataframes.
        _weights_num = pd.DataFrame()
        for k, v in zip(instance_data.numerical_feature_names, weights_num):
            _weights_num[k] = [v]
        _weights_cat = pd.DataFrame()
        iter = zip(
            instance_data.categorical_feature_category_tuples, weights_cat
        )
        for k, v in iter:
            column = f"{k[0]}:{k[1]}"
            _weights_cat[column] = [v]

    if data_type == "train":
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
    elif data_type == "test":
        num_dataframe = instance_data.get_num_x_test_dataframe(wide=True)
        cat_dataframe = instance_data.get_cat_x_test_dataframe(wide=True)
        y_df = instance_data.get_y_test_dataframe()
        X_df = pd.concat([num_dataframe, cat_dataframe], axis=1)
        X = X_df.to_numpy()
        y = y_df.to_numpy()
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
    

    _weights_num.to_csv(
        "programs/minlp/attacks/{}/initial_numerical_weights.csv".format(
            config["dataset_name"]
        )
    )
    _weights_cat.to_csv(
        "programs/minlp/attacks/{}/initial_categorical_weights.csv".format(
            config["dataset_name"]
        )
    )

    print(f"MSE: {mse}")

    return {
        "weights_num": _weights_num,
        "weights_cat": _weights_cat,
        "bias": bias,
        "mse": mse,
    }

if __name__ == "__main__":
    from main import config
    import instance_data_class

    instance_data = instance_data_class.InstanceData(config)
    numerical_model = None

    regression_parameters = run_not_poisoned(
        config, instance_data, data_type="test"
    )


