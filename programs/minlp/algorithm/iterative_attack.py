# -*- coding: utf-8 -*-

"""Run iterative attack which which poison training data row by row"""

import numpy as np
import pyomo.environ as pyo
import pyomo.kernel as pmo
import pandas as pd
import pprint

# TODO Refactor and simplify function calls around model building.
# TODO Improve efficiency by avoid calling unnecesary instance_data.get_x.


long_space = 80
short_space = 60
middle_space = long_space


# TODO Modify this function to take instance_data and pyomo model as arguments.
def run(config):
    """Run iterative attack which which poison training data row by row"""
    from model import pyomo_instance_class

    # Solve benchmark
    opt = pyo.SolverFactory("ipopt")

    print("" * 2)
    print("*" * long_space)
    print("ITERATIVE CONTINUOUS NONLINEAR ALGORITHM")
    print("*" * long_space)

    print("Building data class")
    instance_data = pyomo_instance_class.InstanceData(config)

    (
        benchmark_model,
        benchmark_instance,
        benchmark_solution,
    ) = iterative_attack_strategy(
        opt=opt, instance_data=instance_data, config=config
    )
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

    model = IterativeAttackModel(instance_data, config["function"])

    n_epochs = config["iterative_attack_n_epochs"]
    mini_batch_size = config["iterative_attack_mini_batch_size"]

    incremental = config["iterative_attack_incremental"]

    if incremental:
        if n_epochs > 1:
            raise ValueError(f"n_epochs should be 1 when incremental but got {n_epochs}")

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
            flag[:breaks[mini_batch_index]] = model.POISON_DATA_FIXED
            flag[breaks[mini_batch_index] : breaks[mini_batch_index + 1]] = model.POISON_DATA_OPTIMIZED
            if incremental:
                flag[breaks[mini_batch_index + 1]:] = model.POISON_DATA_REMOVED
            else:
                flag[breaks[mini_batch_index + 1]:] = model.POISON_DATA_FIXED
            model.fix_rows_in_poison_dataframe(instance_data, flag)
            opt.solve(model, load_solutions=True, tee=False)
            solution = model.get_solution()
            instance_data.update_numerical_features(solution["optimized_x_poison_num"])
            solution_list.append(solution)
            if (epoch * n_mini_batches + mini_batch_index) % 20 == 0:
                print(f"{'epoch':>5s}  " f"{'batch':>5s}  " f"{'objective':>9s}")
            print(
                f"{epoch:5d}  "
                f"{mini_batch_index:5d}  "
                f"{solution['objective']:9.6f}"
            )

    # This will break when solution_list is empty, but maybe it's unlikely
    keys = solution_list[0].keys()
    out = {key: np.stack([x[key] for x in solution_list]) for key in keys}

    print("objective in each iteration:")
    print(out["objective"])
    print("improvement from the start (%):")
    print(
        ((out["objective"] - out["objective"][0]) / out["objective"][0] * 100).round(2)
    )

    return model, instance_data, solution


class IterativeAttackModel(pmo.block):
    """Pyomo model to run iterative attack

    This is a naive implementation of iterative attack.
    By default this optimizes all the numerical features in the poison data.
    One can fix some rows in the poison data and only optimize the remaining
    by calling `fix_rows_in_poison_dataframe`.

    ```
    model = IterativeAttackModel(instance_data, function="MSE")
    fixed = np.ones(instance_data.no_poison_samples)
    fixed[0, 1] = 0  # Only optimize the first two rows in the poison data.
    model.fix_rows_in_poison_dataframe(instance_data, fixed)
    model.get_solution()['x_poison_num']
    ```
    """

    POISON_DATA_FIXED = 0
    POISON_DATA_OPTIMIZED = 1
    POISON_DATA_REMOVED = 2

    def __init__(
        self,
        instance_data,
        function,
        **kwds,
    ):
        # Gives access to methods in a superclass from the subclass that
        # inherits from it
        super().__init__(**kwds)
        # Initialize the whole abstract model whenever PoisonAttackModel is created:
        self.function = function
        print("" * 2)
        print("*" * long_space)
        print("CONTINUOUS NONLINEAR MODEL")
        print("*" * long_space)
        self.update_parameters(instance_data, build=True)
        print("*" * short_space)
        self.build_variables(instance_data)
        print("*" * short_space)
        self.build_constraints(instance_data)
        print("*" * long_space)
        self.build_objective(instance_data)
        print("*" * long_space)

    def update_parameters(self, instance_data, build=False):
        """
        Build or update parameters in pyomo.
        """
        if build:
            self.x_train_num = {}
            self.x_train_cat = {}
            self.y_train = {}
            self.x_poison_cat = {}
            self.y_poison = {}

            # Status code (POISON_DATA_FIXED/OPTIMIZED/REMOVED)
            self.poison_data_status = {
                k: pmo.parameter() for k in range(instance_data.no_poison_samples)
            }
            # 1 if the corresponding row is removed and 0 otherwise.
            self.poison_data_is_removed = {
                k: pmo.parameter() for k in range(instance_data.no_poison_samples)
            }
            self.no_poison_samples_in_model = pmo.parameter(instance_data.no_poison_samples)

        for k, v in instance_data.get_num_x_train_dataframe().items():
            self.x_train_num.setdefault(k, pmo.parameter())
            self.x_train_num[k].value = v
        for k, v in instance_data.get_cat_x_train_dataframe().items():
            self.x_train_cat.setdefault(k, pmo.parameter())
            self.x_train_cat[k].value = v
        for k, v in instance_data.get_y_train_dataframe().items():
            self.y_train.setdefault(k, pmo.parameter())
            self.y_train[k] = v
        for k, v in instance_data.get_cat_x_poison_dataframe().items():
            self.x_poison_cat.setdefault(k, pmo.parameter())
            self.x_poison_cat[k].value = v
        for k, v in instance_data.get_y_poison_dataframe().items():
            self.y_poison.setdefault(k, pmo.parameter())
            self.y_poison[k] = v

    def build_variables(self, instance_data):
        """
        PYOMO
        Decision variables of single level model: features of poisoned samples,
        weights of regression model, and bias of regression model.
        """

        print("Creating variables")

        # Numerical feature vector of poisoned samples
        self.x_poison_num = pmo.variable_dict()
        for psample in range(instance_data.no_poison_samples):
            for numfeature in instance_data.numerical_feature_names:
                self.x_poison_num[psample, numfeature] = pmo.variable(
                    domain=pmo.PercentFraction
                )

        # TODO Fix bounds.
        # upper_bound = bnd.find_bounds(instance_data, self)
        upper_bound = 10
        lower_bound = -upper_bound
        print(f"Upper bound is: {upper_bound:.2f}")
        print(f"Lower bound is: {lower_bound:.2f}")

        self.weights_num = pmo.variable_dict()
        for numfeature in instance_data.numerical_feature_names:
            self.weights_num[numfeature] = pmo.variable(
                domain=pmo.Reals, lb=lower_bound, ub=upper_bound, value=0
            )

        self.weights_cat = pmo.variable_dict()
        for cat_feature in instance_data.categorical_feature_names:
            categories = instance_data.categories_in_categorical_feature[cat_feature]
            for category in categories:
                self.weights_cat[cat_feature, category] = pmo.variable(
                    domain=pmo.Reals, lb=lower_bound, ub=upper_bound, value=0
                )

        self.bias = pmo.variable(domain=pmo.Reals, lb=lower_bound, ub=upper_bound)

    def build_constraints(self, instance_data):
        """
        PYOMO
        Constraints of the single-level reformulation: first order optimality
        conditions for lower-level variables: weights and bias of regression
        model
        """

        print("Building num weights contraints")
        self.cons_first_order_optimality_conditions_num_weights = (
            pmo.constraint_dict()
        )  # There is one constraint per feature
        for numfeature in instance_data.numerical_feature_names:
            self.cons_first_order_optimality_conditions_num_weights[
                numfeature
            ] = pmo.constraint(
                body=loss_function_derivative_num_weights(
                    instance_data, self, numfeature, self.function
                ),
                rhs=0,
            )
        print("Building cat weights contraints")
        self.cons_first_order_optimality_conditions_cat_weights = pmo.constraint_dict()
        for cat_feature in instance_data.categorical_feature_names:
            for category in instance_data.categories_in_categorical_feature[
                cat_feature
            ]:
                self.cons_first_order_optimality_conditions_cat_weights[
                    cat_feature, category
                ] = pmo.constraint(
                    body=loss_function_derivative_cat_weights(
                        instance_data, self, cat_feature, category, self.function
                    ),
                    rhs=0,
                )

        print("Building bias constraints")
        self.cons_first_order_optimality_conditions_bias = pmo.constraint(
            body=loss_function_derivative_bias(instance_data, self, self.function),
            rhs=0,
        )

    def build_objective(self, instance_data):
        """
        PYOMO
        Objective function of single-level reformulation, same as leader's
        objective for bi-level model.
        """

        self.objective_function = pmo.objective(
            expr=mean_squared_error(instance_data, self, self.function),
            sense=pyo.maximize,
        )

        print("Objective has been built")

    def fix_rows_in_poison_dataframe(self, instance_data, flag):
        """Fix specified rows in poisoned data

        Parameters
        ----------
        instance_data
        flag : (instance_data.no_poison_samples,) array of int
            If flag[i] is 1, the corresponding poisoned data is fixed.
            Otherwise, the poisoned data is optimized.
        """
        for k, v in enumerate(flag):
            self.poison_data_status[k].value = v
            self.poison_data_is_removed[k].value = int(v == self.POISON_DATA_REMOVED)
        self.no_poison_samples_in_model.value = np.sum(flag != self.POISON_DATA_REMOVED)

        iter = instance_data.get_num_x_poison_dataframe().to_dict().items()
        for k, v in iter:
            if flag[k[0]] == self.POISON_DATA_OPTIMIZED:
                self.x_poison_num[k].unfix()
            else:
                self.x_poison_num[k].fix(v)

    def get_solution(self, wide=False):
        """Retrieve solutions

        This returns a solution as a dict with the following items.

        - x_poison_num: pd.Series or pd.DataFrame
            Numerical features of poisoned data including fixed ones
        - optimized_x_poison_num: pd.DataFrame
            Numerical features of poisoned data excluding fixed ones
        - weights_num: pd.Series or pd.DataFrame
        - weights_cat: pd.Series or pd.DataFrame
        - bias: float
        - objective: float

        Parameters
        ----------
        wide : bool, default False
            Control the format of the output.

        Returns
        -------
        solution : dict
        """
        if not wide:
            # To make long format dataframes.
            index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["sample", "feature"]
            )
            _x_poison_num = pd.Series(index=index)
            _optimized_x_poison_num = pd.Series(index=index)
            for k, v in self.x_poison_num.items():
                _x_poison_num.loc[k] = v.value
                if not v.is_fixed():
                    _optimized_x_poison_num.loc[k] = v.value
            _weights_num = pd.Series()
            for k, v in self.weights_num.items():
                _weights_num[k] = v.value
            index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["feature", "category"]
            )
            _weights_cat = pd.Series(index=index)
            for k, v in self.weights_cat.items():
                _weights_cat.loc[k] = v.value
        else:
            # To make wide fromat dataframes.
            _x_poison_num = pd.DataFrame()
            _optimized_x_poison_num = pd.DataFrame()
            for k, v in self.x_poison_num.items():
                _x_poison_num.loc[k] = v.value
                if not v.is_fixed():
                    _optimized_x_poison_num.loc[k] = v.value
            _weights_num = pd.DataFrame()
            for k, v in self.weights_num.items():
                _weights_num[k] = [v.value]
            _weights_cat = pd.DataFrame()
            for k, v in self.weights_cat.items():
                column = f"{k[0]}:{k[1]}"
                _weights_cat[column] = [v.value]
        return {
            "x_poison_num": _x_poison_num,
            "optimized_x_poison_num": _optimized_x_poison_num,
            "weights_num": _weights_num,
            "weights_cat": _weights_cat,
            "bias": self.bias.value,
            "objective": pyo.value(self.objective_function),
        }


def linear_regression_function(instance_data, model, no_sample):
    """
    Given the sample, the set of features, the features, and the regression
    parameters weights and bias, this function finds the predicted value for
    a sample.
    LRF (prediction) = weight * sample + bias
    """

    # Predict values using linear regression
    numerical_part = sum(
        model.x_train_num[no_sample, j] * model.weights_num[j]
        for j in instance_data.numerical_feature_names
    )
    categorical_part = sum(
        sum(
            model.weights_cat[j, z] * model.x_train_cat[no_sample, j, z]
            for z in instance_data.categories_in_categorical_feature[j]
        )
        for j in instance_data.categorical_feature_names
    )
    y_hat = numerical_part + categorical_part + model.bias
    return y_hat

def mean_squared_error(instance_data, model, function: str):
    """
    Gets mean squared error, which is the mean of sum of the square of the
    difference between predicted values (regression) and target values for all
    samples. MSE = 1 / n * summation( (predicted - target)^2 )
    """

    # Get sum of squared error of regression prediction and target
    sum_square_errors = sum(
        (linear_regression_function(instance_data, model, i) - model.y_train[i]) ** 2
        for i in range(instance_data.no_train_samples)
    )

    # Get mean of squared errors
    if function == "MSE":
        mse = 1 / instance_data.no_train_samples * sum_square_errors
    elif function == "SLS":
        mse = sum_square_errors

    return mse

def loss_function_derivative_num_weights(instance_data, model, j, function):
    """
    Finds the derivetive of the loss function (follower's objective) with
    respect to the weights of the linear regression model, and sets it to 0
    (first order optimality condition).
    """

    train_samples_component = sum(
        (linear_regression_function(instance_data, model, i) - model.y_train[i])
        * model.x_train_num[i, j]
        for i in range(instance_data.no_train_samples)
    )  # Component involving the sum of training samples errors

    poison_samples_component = sum(
        (
            sum(
                model.x_poison_num[q, j] * model.weights_num[j]
                for j in instance_data.numerical_feature_names
            )
            + sum(
                sum(
                    model.weights_cat[j, z] * model.x_poison_cat[q, j, z]
                    for z in instance_data.categories_in_categorical_feature[j]
                )
                for j in instance_data.categorical_feature_names
            )
            + model.bias
            - model.y_poison[q]
        )
        * model.x_poison_num[q, j]
        * (1 - model.poison_data_is_removed[q])
        for q in range(instance_data.no_poison_samples)
    )

    regularization_component = (
        2 * instance_data.regularization * model.weights_num[j]
    )  # Component involving the regularization

    if function == "MSE":
        final = (2 / (instance_data.no_train_samples + model.no_poison_samples_in_model)) * (
            train_samples_component + poison_samples_component
        ) + regularization_component

    if function == "SLS":
        final = (
            2 * (train_samples_component + poison_samples_component)
            + regularization_component
        )

    return final


def loss_function_derivative_cat_weights(instance_data, model, j, w, function):
    """
    Finds the derivative of the loss function (follower's objective) with
    respect to the weights of the linear regression model, and sets it to 0
    (first order optimality condition).
    """

    train_samples_component = sum(
        (linear_regression_function(instance_data, model, i) - model.y_train[i])
        * model.x_train_cat[i, j, w]
        for i in range(instance_data.no_train_samples)
    )  # Component involving the sum of training samples errors
    poison_samples_component = sum(
        (
            sum(
                model.x_poison_num[q, j] * model.weights_num[j]
                for j in instance_data.numerical_feature_names
            )
            + sum(
                sum(
                    model.weights_cat[j, z] * model.x_poison_cat[q, j, z]
                    for z in instance_data.categories_in_categorical_feature[j]
                )
                for j in instance_data.categorical_feature_names
            )
            + model.bias
            - model.y_poison[q]
        )
        * model.x_poison_cat[q, j, w]
        * (1 - model.poison_data_is_removed[q])
        for q in range(instance_data.no_poison_samples)
    )  # Component involving the sum of poison samples errors

    regularization_component = (
        2 * instance_data.regularization * model.weights_cat[j, w]
    )  # Component involving the regularization

    if function == "MSE":
        final = (2 / (instance_data.no_train_samples + model.no_poison_samples_in_model)) * (
            train_samples_component + poison_samples_component
        ) + regularization_component
    elif function == "SLS":
        final = (
            2 * (train_samples_component + poison_samples_component)
            + regularization_component
        )

    return final


def loss_function_derivative_bias(instance_data, model, function):
    """
    Finds the derivetive of the loss function (follower's objective) with
    respect to the bias of the linear regression model, and sets it to 0 (first
    order optimality condition).
    """

    train_samples_component = sum(
        (linear_regression_function(instance_data, model, i) - model.y_train[i])
        for i in range(instance_data.no_train_samples)
    )
    poison_samples_component = sum(
        (
            sum(
                model.x_poison_num[q, j] * model.weights_num[j]
                for j in instance_data.numerical_feature_names
            )
            + sum(
                sum(
                    model.weights_cat[j, z] * model.x_poison_cat[q, j, z]
                    for z in instance_data.categories_in_categorical_feature[j]
                )
                for j in instance_data.categorical_feature_names
            )
            + model.bias
            - model.y_poison[q]
        )
        * (1 - model.poison_data_is_removed[q])
        for q in range(instance_data.no_poison_samples)
    )

    if function == "MSE":
        final = (2 / (instance_data.no_train_samples + model.no_poison_samples_in_model)) * (
            train_samples_component + poison_samples_component
        )
    elif function == "SLS":
        final = 2 * (train_samples_component + poison_samples_component)

    return final


if __name__ == "__main__":
    import doctest

    n_fails, _ = doctest.testmod()
    if n_fails > 0:
        raise SystemExit(1)

# vimquickrun: python % && ./vimquickrun.sh
