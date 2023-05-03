# -*- coding: utf-8 -*-

"""Gradient descent method with pytorch"""

import copy
import dataclasses
import textwrap

import flipping_attack
import instance_data_class
import numpy as np
import pandas as pd
import ridge_regression
import torch

long_space = 80
short_space = 60
middle_space = long_space


class GradModel:
    """Gradient descent method

    This is a naive implementation of poisoning attack with gradient
    decent.
    """

    POISON_DATA_FIXED = 0
    POISON_DATA_OPTIMIZED = 1
    POISON_DATA_REMOVED = 2

    def __init__(
        self,
        instance_data,
        config,
        **kwds,
    ):
        self.config = copy.deepcopy(config)
        self.set_poison_data_status(
            instance_data, self.POISON_DATA_OPTIMIZED, self.POISON_DATA_FIXED
        )

    def update_parameters(self, instance_data, build=False):
        """
        Build or update parameters in pyomo.
        """
        pass

    def set_poison_data_status(self, instance_data, num_feature_flag, cat_feature_flag):
        """Set status of the variables corresponding to the features in poisoned data

        This sets status of the variables corresponding to the features in poisoned data.
        One can 1) fix the variables to the ones in instance_data,
        2) let the optimizer to optimize them or 3) remove them from the model (this
        is useful when we want to add poisoned data incrementally).
        `num_feature_flag` must be an array broadcastable to
        (no_poison_samples, no_numfeatures). `cat_feature_flag` must be broadcastable
        to (no_poison_samples, no_catfeatures). The elements of these arrays must be
        either self.POISON_DATA_FIXED, self.POISON_DATA_OPTIMIZED or
        self.POISON_DATA_REMOVED. If self.POISON_DATA_REMOVED is used, the corresponding
        entire row is removed. Otherwise, each variable status is set individually.

        Parameters
        ----------
        instance_data : InstanceData
        num_feature_flag : array of int, broadcastable to (no_poison_samples, no_numfeatures)
        cat_feature_flag : array of int, broadcastable to (no_poison_samples, no_catfeatures)
        """
        self.num_feature_flag = np.broadcast_to(
            num_feature_flag,
            (instance_data.no_poison_samples, instance_data.no_numfeatures),
        )
        self.cat_feature_flag = np.broadcast_to(
            cat_feature_flag,
            (instance_data.no_poison_samples, instance_data.no_catfeatures),
        )
        self.poison_data_is_removed = np.any(
            self.num_feature_flag == self.POISON_DATA_REMOVED, axis=1
        )
        self.poison_data_is_removed |= np.any(
            self.cat_feature_flag == self.POISON_DATA_REMOVED, axis=1
        )
        np.testing.assert_equal(
            self.poison_data_is_removed.shape,
            (instance_data.no_poison_samples,),
        )
        if np.any(self.poison_data_is_removed):
            raise NotImplementedError("removing poisoning data is not yet supported")
        self.instance_data = instance_data

    def continuous_relaxation(self, mode=True):
        """Create a continuous relaxation of the poisoning attack

        This replaces binary variables with continuous variables.

        Parameters
        ----------
        mode : bool, default True
        """
        raise NotImplementedError

    def solve(self):
        """Run gradient descent method with pytorch"""
        instance_data = self.instance_data

        X_num_train = instance_data.get_num_x_train_dataframe(wide=True).to_numpy()
        X_cat_train_one_hot = instance_data.get_cat_x_train_dataframe(
            wide=True
        ).to_numpy()
        y_train = instance_data.get_y_train_dataframe().to_numpy()
        X_num_poison = instance_data.get_num_x_poison_dataframe(wide=True).to_numpy()
        X_cat_poison_one_hot = instance_data.get_cat_x_poison_dataframe(
            wide=True
        ).to_numpy()
        y_poison = instance_data.get_y_poison_dataframe().to_numpy()

        regularisation_parameter = instance_data.regularization
        X_num_train = torch.tensor(X_num_train)
        X_cat_train_one_hot = torch.tensor(X_cat_train_one_hot)
        y_train = torch.tensor(y_train)
        X_num_poison = torch.tensor(X_num_poison, requires_grad=True)
        X_cat_poison_one_hot = torch.tensor(
            X_cat_poison_one_hot.astype(float), requires_grad=True
        )
        y_poison = torch.tensor(y_poison)

        lr = self.config["gradient_solver_lr"]
        momentum = self.config["gradient_solver_momentum"]
        optimizer = torch.optim.SGD(
            [X_num_poison, X_cat_poison_one_hot],
            lr=lr,
            momentum=momentum,
            maximize=True,
        )

        X_train = torch.cat([X_num_train, X_cat_train_one_hot], axis=1)

        num_feature_is_optimized = torch.Tensor(
            self.num_feature_flag == self.POISON_DATA_OPTIMIZED
        )
        cat_feature_is_optimized = torch.Tensor(
            self.cat_feature_flag == self.POISON_DATA_OPTIMIZED
        )

        loss_scaler = 1.0

        log_every = self.config["gradient_solver_log_every"]
        iteration_limit = self.config["gradient_solver_iteration_limit"]

        best_solutions = Solution()
        best_loss = -np.inf
        best_loss_updated = False
        best_loss_no_update_count = 0
        tol = self.config["gradient_solver_loss_record_update_tol"]
        patience = 4

        include_bias = True

        no_samples = instance_data.no_train_samples + instance_data.no_poison_samples
        if include_bias:
            regularisation_parameter_matrix = (
                no_samples * regularisation_parameter * torch.eye(X_train.shape[1] + 1)
            )
            regularisation_parameter_matrix[-1, -1] = 0
        else:
            regularisation_parameter_matrix = (
                no_samples * regularisation_parameter * torch.eye(X_train.shape[1])
            )

        for iteration in range(iteration_limit):
            _X_num_poison = (
                num_feature_is_optimized * X_num_poison
                + (1 - num_feature_is_optimized) * X_num_poison.detach()
            )
            _X_cat_poison_one_hot = (
                cat_feature_is_optimized * X_cat_poison_one_hot
                + (1 - cat_feature_is_optimized) * X_cat_poison_one_hot.detach()
            )
            X_poison = torch.cat([_X_num_poison, _X_cat_poison_one_hot], axis=1)
            X = torch.cat([X_train, X_poison], axis=0)
            y = torch.cat([y_train, y_poison])
            # Add a column of ones to include bias.
            if include_bias:
                X = torch.cat([X, torch.ones(X.shape[0], 1)], axis=1)
            theta = torch.linalg.solve(
                X.T @ X + regularisation_parameter_matrix, X.T @ y
            )
            if include_bias:
                bias = theta[-1]
                theta = theta[:-1]
            else:
                bias = 0
            y_pred = X_train @ theta + bias
            np.testing.assert_equal(y_train.shape, y_pred.shape)
            loss = torch.nn.functional.mse_loss(y_train, y_pred)

            if iteration == 0:
                best_loss_updated = True
            else:
                best_loss_updated = best_loss + np.abs(best_loss) * tol < loss.item()

            if best_loss_updated:
                best_loss = loss.item()
                best_loss_no_update_count = 0
                best_solutions.update(
                    X_num_poison.detach().numpy().copy(),
                    X_cat_poison_one_hot.detach().numpy().copy(),
                    y_poison.detach().numpy().copy(),
                    theta.detach().numpy().copy(),
                    bias.item(),
                    loss.item(),
                )
            else:
                best_loss_no_update_count += 1

            if iteration == 0:
                best_loss_updated_flag = " "
            elif best_loss_updated:
                best_loss_updated_flag = "*"
            elif best_loss_no_update_count > patience:
                best_loss_updated_flag = " "
            else:
                best_loss_updated_flag = " "

            should_log = (iteration % log_every == 0) or (
                best_loss_no_update_count > patience
            )

            if should_log:
                if iteration % (20 * log_every) == 0:
                    print(f"{'iter':>4s}  {'mse':>9s}")
                print(f"{iteration:4d}  {loss.item():9.6f}{best_loss_updated_flag:1s}")

            if best_loss_no_update_count > patience:
                break

            if iteration < iteration_limit - 1:
                optimizer.zero_grad()
                (loss_scaler * loss).backward()
                optimizer.step()

            if iteration == 0:
                loss_scaler = (
                    1
                    / max(
                        torch.norm(X_num_poison.grad, 2),
                        torch.norm(X_cat_poison_one_hot.grad, 2),
                    ).item()
                )
                if not np.isfinite(loss_scaler):
                    loss_scaler = 1.0

            with torch.no_grad():
                X_num_poison.clamp_(0, 1)
                X_cat_poison_one_hot.clamp_(0, 1)

        self.solutions = best_solutions

        return self.solutions

    def get_mse(self):
        """Get the MSE on the training data after poisoning

        Returns
        -------
        mse : float
        """
        return self.solutions.mse

    def get_solution(self, wide=False):
        """Retrieve solutions

        This returns all the solution. More precisely, this combines the data
        returned by get_poison_data and get_regression_model_parameters.

        Parameters
        ----------
        wide : bool, default False
            Control the format of the output.

        Returns
        -------
        solution : dict
        """
        return {
            **self.get_regression_model_parameters(wide=wide),
            **self.get_poison_data(wide=wide),
        }

    def get_poison_data(self, wide=False, only_optimized=False):
        """Retrieve solutions

        This returns a solution as a dict with the following items.

        - x_poison_num: pd.Series or pd.DataFrame
            Numerical features of poisoned data. If `only_optimized` is
            True, this excludes fixed nor moreved ones. Otherwise, this
            contains fixed ones.
        - x_poison_cat: pd.Series or pd.DataFrame
            Categorical features of poisoned data. If `only_optimized` is
            True, this excludes fixed nor moreved ones. Otherwise, this
            contains fixed ones.

        Parameters
        ----------
        wide : bool, default False
            Control the format of the output.
        only_optimized : bool, default False

        Returns
        -------
        solution : dict
        """
        if not wide:
            # TODO Simplify the construction of dataframes and series.
            # TODO Exrract logic to build solutions and reuse from ridge regression.
            # To make long format dataframes.
            index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["sample", "feature"]
            )
            _x_poison_num = pd.Series(index=index)
            index = self.instance_data.get_num_x_poison_dataframe(wide=wide).index
            flag = self.num_feature_flag.ravel()
            solution = self.solutions.X_num_poison.ravel()
            for i, k in enumerate(index):
                if (not only_optimized) or (flag[i] == self.POISON_DATA_OPTIMIZED):
                    _x_poison_num.loc[k] = solution[i]
            index = pd.MultiIndex(
                levels=[[], [], []],
                codes=[[], [], []],
                names=["sample", "feature", "category"],
            )
            _x_poison_cat = pd.Series(index=index)
            index = self.instance_data.get_cat_x_poison_dataframe(wide=wide).index
            flag = self.cat_feature_flag.ravel()
            solution = self.solutions.X_cat_poison_one_hot.ravel()
            np.testing.assert_equal(flag.shape, index.shape)
            np.testing.assert_equal(solution.shape, index.shape)
            for i, k in enumerate(index):
                if (not only_optimized) or (flag[i] == self.POISON_DATA_OPTIMIZED):
                    _x_poison_cat.loc[k] = solution[i]
        else:
            raise NotImplementedError
            # To make wide fromat dataframes.
            _x_poison_num = pd.DataFrame()
            for k, v in self.x_poison_num.items():
                if (not only_optimized) or (not v.is_fixed()):
                    _x_poison_num.loc[k] = v.value
            _x_poison_cat = pd.DataFrame()
            for k, v in self.x_poison_cat.items():
                # TODO
                if (not only_optimized) or (not v.is_fixed()):
                    _x_poison_cat.loc[k] = v.value
        out = {
            "x_poison_num": _x_poison_num,
            "x_poison_cat": _x_poison_cat,
        }
        return out

    def get_regression_model_parameters(self, wide=False):
        """Retrieve solutions

        This returns a solution as a dict with the following items.

        - weights_num: pd.Series or pd.DataFrame
        - weights_cat: pd.Series or pd.DataFrame
        - bias: float
        - objective: float
        - mse : float
            This is the same as objective if `config['function'] == 'MSE'`.
            If `config['function'] == 'SLS'`, this is the same as
            objective / no_train_samples

        Parameters
        ----------
        wide : bool, default False
            Control the format of the output.

        Returns
        -------
        solution : dict
        """
        if not wide:
            # TODO Simplify the construction of dataframes and series.
            # TODO Exrract logic to build solutions and reuse from ridge regression.
            # To make long format dataframes.
            _weights_num = pd.Series()

            index = self.instance_data.get_num_x_poison_dataframe(wide=True).columns
            solution = self.solutions.weights[: len(index)]
            np.testing.assert_equal(index.shape, solution.shape)

            for k, v in zip(index, solution):
                _weights_num[k] = v
            index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["feature", "category"]
            )
            _weights_cat = pd.Series(index=index)
            index = self.instance_data.categorical_feature_category_tuples
            solution = self.solutions.weights[-len(index) :]
            np.testing.assert_equal((len(index),), solution.shape)
            for k, v in zip(index, solution):
                _weights_cat.loc[k] = v
        else:
            raise NotImplementedError
            # To make wide format dataframes.
            _weights_num = pd.DataFrame()
            for k, v in self.weights_num.items():
                _weights_num[k] = [v.value]
            _weights_cat = pd.DataFrame()
            for k, v in self.weights_cat.items():
                column = f"{k[0]}:{k[1]}"
                _weights_cat[column] = [v.value]
        out = {
            "weights_num": _weights_num,
            "weights_cat": _weights_cat,
            "bias": self.solutions.bias,
            "objective": self.solutions.mse,
            "mse": self.solutions.mse,
        }
        return out

    def update_data(self, *args, **kwargs):
        """Update instance_data using current solution"""
        return self._update_data(*args, **kwargs, inplace=True)

    def updated_data(self, *args, **kwargs):
        """Create a new data_instance updated with current solution"""
        return self._update_data(*args, **kwargs, inplace=False)

    def _update_data(
        self, instance_data, numerical=True, categorical=True, inplace=False
    ):
        if inplace:
            out = instance_data
        else:
            out = instance_data.copy()
        solution = self.get_poison_data(only_optimized=True)
        if numerical:
            out.update_numerical_features(solution["x_poison_num"])
        if categorical:
            out.update_categorical_features(solution["x_poison_cat"])
        return out


@dataclasses.dataclass
class Solution:
    X_num_poison: np.ndarray = None
    X_cat_poison_one_hot: np.ndarray = None
    y_poison: np.ndarray = None
    weights: np.ndarray = None
    bias: float = np.nan
    mse: float = np.nan

    def update(
        self,
        X_num_poison: np.ndarray = None,
        X_cat_poison_one_hot: np.ndarray = None,
        y_poison: np.ndarray = None,
        weights: np.ndarray = None,
        bias: float = np.nan,
        mse: float = np.nan,
    ):
        self.X_num_poison = X_num_poison
        self.X_cat_poison_one_hot = X_cat_poison_one_hot
        self.y_poison = y_poison
        self.weights = weights
        self.bias = bias
        self.mse = mse


def run(config, instance_data):
    """Optimize numerical feature row by row using the gradient method"""
    print("\n" + "*" * long_space)
    print("Gradient Solver: update the numerical features")
    print("*" * long_space)

    solver = GradModel(instance_data, config)

    num_feature_flag = solver.POISON_DATA_OPTIMIZED
    cat_feature_flag = solver.POISON_DATA_FIXED

    for row in range(instance_data.no_poison_samples):
        print(f">>> Optimizing the numerical features in row {row}")
        num_feature_flag = np.full(
            (instance_data.no_poison_samples, 1), solver.POISON_DATA_FIXED
        )
        num_feature_flag[row] = solver.POISON_DATA_OPTIMIZED

        solver.set_poison_data_status(instance_data, num_feature_flag, cat_feature_flag)
        solver.solve()
        solver.update_data(instance_data)

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

        for key in ["bias", "weights_num", "weights_cat", "bias", "mse"]:
            a = flatten(sol1[key])
            b = flatten(sol2[key])
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, err_msg=key)

    assert_solutions_are_close(
        solver.get_regression_model_parameters(),
        scikit_learn_regression_parameters,
    )


def to_one_hot(X, n_categories):
    """Create a one hot encoding of categorical features

    Examples
    --------
    >>> X = [[1, 1], [2, 0], [2, 1]]
    >>> print(to_one_hot(X, n_categories=[3, 2]))
    [[0. 1. 0. 0. 1.]
     [0. 0. 1. 1. 0.]
     [0. 0. 1. 0. 1.]]
    >>> print(to_one_hot(X, n_categories=[4, 2]))
    [[0. 1. 0. 0. 0. 1.]
     [0. 0. 1. 0. 1. 0.]
     [0. 0. 1. 0. 0. 1.]]

    Parameters
    ----------
    X : (n_samples, n_categorical_features) array
    n_categories : (n_categorical_features,) array
        `n_categories[i]` is  the number of categories categorical feature
        `i` has.

    Returns
    -------
    X_one_hot : (n_samples, sum(n_categories)) array
    """
    X = np.asarray(X)
    n_samples, n_cat_features = X.shape
    X_one_hot = np.zeros((n_samples, np.sum(n_categories)))
    cat_one_hot_start = np.r_[0, np.cumsum(n_categories[:-1])]
    for sample in range(n_samples):
        for cat_feature in range(n_cat_features):
            X_one_hot[
                sample, cat_one_hot_start[cat_feature] + X[sample, cat_feature]
            ] = 1
    return X_one_hot


def flipping_without_numerical(config, instance_data, model=None):
    """Copied from flipping_attack but numerical attack is removed"""
    config = copy.deepcopy(config)
    instance_data = instance_data.copy()

    print("\n" + "*" * long_space)
    print("Flipping Attack: Update the categorical features")
    print("*" * long_space)

    # TODO we use both solvers. but ipopts first.
    if not config.get("solver_name"):
        config["solver_name"] = "ipopt"
    np.testing.assert_equal(config["solver_name"], "ipopt")

    n_epochs = config["flipping_attack_n_epochs"]

    no_poison_samples = instance_data.no_poison_samples

    # Solve benchmark
    config["iterative_attack_incremental"] = False
    numerical_model = None

    best_sol = ridge_regression.run(config, instance_data)
    best_instance_data = instance_data.copy()

    for epoch in range(n_epochs):
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
            target_y = instance_data.get_y_poison_dataframe().iloc[poison_sample_index]
            difference = num_y - target_y

            # We consider two case: Make prediction as large as possible and make prediction
            # as small as possible. We then take the best one.

            # categories_up/down[feature] is the category to push prediction up/down.
            # cat_features = instance_data.categorical_feature_category_tuples
            cat_features = set([cat_feature[0] for cat_feature in cat_weights.keys()])
            categories_up = dict()
            categories_down = dict()
            for feature in cat_features:
                # Filter the keys based on given values for first two elements
                filtered_keys = [k for k in cat_weights.keys() if k[0] == feature]
                categories_up[feature] = max(filtered_keys, key=cat_weights.get)[1]
                categories_down[feature] = min(filtered_keys, key=cat_weights.get)[1]

            # Let's compute the prediction of each case.
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
                best_sol = sol  # TODO make sure we use new weights (which are already computed)
                best_instance_data = instance_data.copy()

    # TODO printing of solutions

    # TODO what do we do with model
    return model, best_instance_data, best_sol


def main():
    """Run gradient descent method with pytorch"""
    config = {
        # Dataset
        "dataset_name": "5num5cat",
        "training_samples": 40,
        "poison_rate": 20,
        "seed": 3,
        # Problem
        "function": "MSE",
        "regularization": 0.6612244897959183,
        "solver_name": "ipopt",
        # Solvers
        "solver_output": False,
        "feasibility": 0.00001,
        "time_limit": 20,
        # Numerical attack
        "numerical_attack_n_epochs": 1,
        "numerical_attack_mini_batch_size": 0.2,
        "numerical_attack_incremental": False,
        # Categorical attack
        "categorical_attack_n_epochs": 1,
        "categorical_attack_mini_batch_size": 0.1,
        "categorical_attack_no_nfeatures": 100,
        "categorical_attack_no_cfeatures": 100,
        # Iterative attack
        "iterative_attack_n_epochs": 10,
        # Flipping attack
        "flipping_attack_n_epochs": 1,
        # Solutions
        "datatype": "test",
        # Gradient solver
        "gradient_solver_iteration_limit": 10,
        "gradient_solver_log_every": 1,
        "gradient_solver_lr": 0.1,
        "gradient_solver_momentum": 0.0,
        "gradient_solver_loss_record_update_tol": 1e-3,
    }
    instance_data = instance_data_class.InstanceData(config)

    print(
        textwrap.dedent(
            """
            ====================================================
            || Gradient solver & flipping heuristic           ||
            ====================================================

            In this example, we optimize the numerical fatures
            and the categorical features in the poisoning dataset
            alternatively, using the gradient method and
            the iterative flipping, respectively.

            The gradient method is a local method: it starts from
            the current values of the poisoning data and tries to
            improve the objective. Unlike Ipopt, this method will
            not terminate at a point with a worse objective value.

            To compute the gradient we use PyTorch.
            """
        ).strip()
    )

    for outer_iteration in range(2):
        run(config, instance_data)
        _, instance_data, _ = flipping_without_numerical(config, instance_data)


if __name__ == "__main__":
    import doctest

    n_failures, _ = doctest.testmod()
    if n_failures > 0:
        raise ValueError(f"{n_failures} tests failed")

    main()

# vimquickrun: python %
