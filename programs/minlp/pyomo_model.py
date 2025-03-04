# -*- coding: utf-8 -*-

"""Run iterative attack which which poison training data row by row"""

import copy

import numpy as np
import pandas as pd
import pyomo
import pyomo.environ as pyo
import pyomo.kernel as pmo
import os
import bounding_procedure

# import os
# os.environ['NEOS_SERVER'] = 'neos-server.org:3333'

os.environ["NEOS_EMAIL"] = "s1553571@ed.ac.uk"

# TODO Refactor and simplify function calls around model building.
# TODO Improve efficiency by avoid calling unnecesary instance_data.get_x.

long_space = 70
short_space = 50
middle_space = long_space


class PyomoModel(pmo.block):
    """Pyomo model to formulate poisoning attack

    This is a naive implementation of poisoning attack.
    One can fix some variables, such as categorical features, in the poison data and
    only optimize the remaining by calling `set_poison_data_status`.
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
        # Gives access to methods in a superclass from the subclass that
        # inherits from it
        super().__init__(**kwds)
        # Initialize the whole abstract model whenever PoisonAttackModel is created:
        self.function = config["function"]
        self.solver_name = config["solver_name"]
        self.tee = config["solver_output"]
        self.binary = config["binary"]
        if config["bounding"] == False:
            self.upper_bound = 10000
            self.lb_intercept = -10000
            self.ub_intercept = 10000
        elif config["bounding"] == True:
            bounds = bounding_procedure.find_bounds(
                instance_data.train_dataframe.iloc[:, :-1],
                instance_data.train_dataframe.iloc[:, -1],
                instance_data.no_poison_samples,
                config,
            )
            self.upper_bound = bounds[0]
            self.lb_intercept = bounds[2]
            self.ub_intercept = bounds[1]

        if self.solver_name == "ipopt":
            self.opt = pyo.SolverFactory("ipopt")
        elif self.solver_name == "knitro":
            self.opt = pyo.SolverFactory("knitro")
            # executable='/usr/local/knitro/12.3/knitroampl/knitroampl')
        elif self.solver_name == "baron":
            self.opt = pyo.SolverFactory("baron")
        elif self.solver_name == "neos":
            self.opt = pyo.SolverManagerFactory("neos")
        elif self.binary:
            self.opt = pyo.SolverFactory("gurobi", solver_io="python")
            self.opt.options["NonConvex"] = 2
            self.bilinear_term_cache = dict()
            self.bilinear_term_variable_list = pmo.variable_list()
            self.bilinear_term_constraint_list = pmo.constraint_list()
            self.opt.options["timelimit"] = config["time_limit"]
        else:
            self.opt = pyo.SolverFactory("gurobi", solver_io="python")
            self.opt.options["NonConvex"] = 2
            self.bilinear_term_cache = dict()
            self.bilinear_term_variable_list = pmo.variable_list()
            self.bilinear_term_constraint_list = pmo.constraint_list()
            self.opt.options["timelimit"] = config["time_limit"]
        # print("" * 2)
        # print("-" * long_space)
        # print("Building model...")
        # print("-" * long_space)
        self.update_parameters(instance_data, build=True)
        # print("*" * short_space)
        self.build_variables(instance_data)
        # print("*" * short_space)
        self.build_constraints(instance_data)
        # print("*" * short_space)
        self.build_objective(instance_data)
        # print("*" * short_space)

    def copy(self):
        """Return a deepcopy of self"""
        return copy.deepcopy(self)

    def prod(self, a, b):
        """Return the product of two expressions"""
        if self.solver_name == "gurobi" and not self.binary:
            return self._prod_gurobi(a, b)
        elif self.binary:
            return self._prod_linearise(a, b)
        # if self.solver_name in ["ipopt", "neos"]:
        #     return self._prod_ipopt(a, b)
        else:
            return self._prod_ipopt(a, b)
        # else:
        #     raise ValueError(f"unknown solver name {self.solver_name}")

    def triple_prod(self, a, b, c):
        # if self.solver_name in ["ipopt", "neos"]:
        #     return self._triple_prod_ipopt(a, b, c)
        if self.solver_name == "gurobi" and not self.binary:
            raise NotImplementedError
        elif self.binary:
            return self._triple_prod_linearise(a, b, c)
        else:
            return self._triple_prod_ipopt(a, b, c)
        # else:
        #     raise ValueError(f"unknown solver name {self.solver_name}")

    def _prod_ipopt(self, a, b):
        return a * b

    def _triple_prod_ipopt(self, a, b, c):
        return a * b * c

    def _prod_gurobi(self, a, b):
        # MODIFIED
        # >>>>>
        # u, v = (a, b) if id(a) < id(b) else (b, a)
        # -----
        u, v = sorted((a, b), key=id)
        # <<<<<
        key = (id(u), id(v))
        if key in self.bilinear_term_cache:
            return self.bilinear_term_cache[key]
        x = pmo.variable()
        self.bilinear_term_variable_list.append(x)
        self.bilinear_term_constraint_list.append(pmo.constraint(x == u * v))
        self.bilinear_term_cache[key] = x
        return x

    def _triple_prod_linearise(self, a, b, c):
        # MODIFIED
        # >>>>>
        # array = [a, b, c]
        # ids = [id(a), id(b), id(c)]
        # sorted = np.argsort(ids)
        # u, v, w = (array[sorted[0]], array[sorted[1]], array[sorted[2]])
        # -----
        u, v, w = sorted((a, b, c), key=id)
        # <<<<<

        key = (id(u), id(v), id(w))
        if key in self.bilinear_term_cache:
            return self.bilinear_term_cache[key]

        if a.is_binary() + b.is_binary() + c.is_binary() != 2:
            raise ValueError("two variables must be binary")

        # Two binaries, one continuous.
        if u.is_binary() and w.is_binary():
            binary_1 = u
            binary_2 = w
            continuous = v
        elif v.is_binary() and w.is_binary():
            binary_1 = v
            binary_2 = w
            continuous = u
        else:
            binary_1 = u
            binary_2 = v
            continuous = w
        lb = continuous.lb
        ub = continuous.ub

        z = pmo.variable(lb=min(0, lb), ub=max(0, ub))
        self.bilinear_term_variable_list.append(z)
        self.bilinear_term_constraint_list.append(
            pmo.constraint(z <= ub * binary_1)
        )
        self.bilinear_term_constraint_list.append(
            pmo.constraint(lb * binary_1 <= z)
        )
        self.bilinear_term_constraint_list.append(
            pmo.constraint(z <= ub * binary_2)
        )
        self.bilinear_term_constraint_list.append(
            pmo.constraint(lb * binary_2 <= z)
        )
        self.bilinear_term_constraint_list.append(
            pmo.constraint(continuous - z <= ub * (2 - binary_1 - binary_2))
        )
        self.bilinear_term_constraint_list.append(
            pmo.constraint(lb * (2 - binary_1 - binary_2) <= continuous - z)
        )
        self.bilinear_term_cache[key] = z
        return z

    def _prod_linearise(self, a, b):
        u, v = (a, b) if id(a) < id(b) else (b, a)
        key = (id(u), id(v))
        if key in self.bilinear_term_cache:
            return self.bilinear_term_cache[key]
        if (not u.is_binary()) and (not v.is_binary()):
            raise ValueError("at least one variables must be binary")
        if id(u) == id(v):
            # The same variable is given. x^2 = x, so return itself.
            return u
        if u.is_binary() and v.is_binary():
            # Both are binary variables.
            z = pmo.variable(domain=pmo.Binary)
            self.bilinear_term_variable_list.append(z)
            self.bilinear_term_constraint_list.append(pmo.constraint(z <= u))
            self.bilinear_term_constraint_list.append(pmo.constraint(z <= v))
            self.bilinear_term_constraint_list.append(
                pmo.constraint(z >= u + v - 1)
            )
            self.bilinear_term_cache[key] = z
            return z
        # One binary, one continuous.
        binary = u if u.is_binary() else v
        continuous = v if u.is_binary() else u
        lb = continuous.lb
        ub = continuous.ub

        z = pmo.variable(lb=min(0, lb), ub=max(0, ub))
        self.bilinear_term_variable_list.append(z)
        self.bilinear_term_constraint_list.append(
            pmo.constraint(z <= ub * binary)
        )
        self.bilinear_term_constraint_list.append(
            pmo.constraint(lb * binary <= z)
        )
        self.bilinear_term_constraint_list.append(
            pmo.constraint(continuous - z <= ub * (1 - binary))
        )
        self.bilinear_term_constraint_list.append(
            pmo.constraint(lb * (1 - binary) <= continuous - z)
        )
        self.bilinear_term_cache[key] = z
        return z

    def update_parameters(self, instance_data, build=False):
        """
        Build or update parameters in pyomo.
        """
        if build:
            self.x_train_num = {}
            self.x_train_cat = {}
            self.y_train = {}
            self.y_poison = {}

            # 1 if the corresponding row is removed and 0 otherwise.
            self.poison_data_is_removed = {
                k: pmo.parameter()
                for k in range(instance_data.no_poison_samples)
            }
            self.no_poison_samples_in_model = pmo.parameter(
                instance_data.no_poison_samples
            )

        for k, v in instance_data.get_num_x_train_dataframe().items():
            self.x_train_num.setdefault(k, pmo.parameter())
            self.x_train_num[k].value = v
        for k, v in instance_data.get_cat_x_train_dataframe().items():
            self.x_train_cat.setdefault(k, pmo.parameter())
            self.x_train_cat[k].value = v
        for k, v in instance_data.get_y_train_dataframe().items():
            self.y_train.setdefault(k, pmo.parameter())
            self.y_train[k] = v
        for k, v in instance_data.get_y_poison_dataframe().items():
            self.y_poison.setdefault(k, pmo.parameter())
            self.y_poison[k] = v

    def build_variables(self, instance_data, print_output=False):
        """
        PYOMO
        Decision variables of single level model: features of poisoned samples,
        weights of regression model, and bias of regression model.
        """
        if print_output:
            print("Creating variables")

        # Numerical feature vector of poisoned samples
        if self.binary:
            domain = pmo.Binary
        else:
            domain = pmo.PercentFraction

        self.x_poison_num = pmo.variable_dict()
        for psample in range(instance_data.no_poison_samples):
            for numfeature in instance_data.numerical_feature_names:
                self.x_poison_num[psample, numfeature] = pmo.variable(
                    domain=domain
                )

        # Categorical feature vector of poisoned samples
        self.x_poison_cat = pmo.variable_dict()
        for psample in range(instance_data.no_poison_samples):
            for (
                catfeature
            ) in instance_data.categorical_feature_category_tuples:
                self.x_poison_cat[(psample,) + catfeature] = pmo.variable(
                    domain=pmo.Binary
                )

        upper_bound = self.upper_bound
        lower_bound = -upper_bound
        ub_intercept = self.ub_intercept
        lb_intercept = self.lb_intercept

        if print_output:
            print(f"Weights upper bound is: {upper_bound:.2f}")
            print(f"Weights lower bound is: {lower_bound:.2f}")
            print(f"Intercept upper bound is: {ub_intercept:.2f}")
            print(f"Intercept lower bound is: {lb_intercept:.2f}")

        self.weights_num = pmo.variable_dict()
        for numfeature in instance_data.numerical_feature_names:
            self.weights_num[numfeature] = pmo.variable(
                domain=pmo.Reals, lb=lower_bound, ub=upper_bound, value=0
            )

        self.weights_cat = pmo.variable_dict()
        for cat_feature in instance_data.categorical_feature_names:
            categories = instance_data.categories_in_categorical_feature[
                cat_feature
            ]
            for category in categories:
                self.weights_cat[cat_feature, category] = pmo.variable(
                    domain=pmo.Reals, lb=lower_bound, ub=upper_bound, value=0
                )

        self.bias = pmo.variable(
            domain=pmo.Reals, lb=lb_intercept, ub=ub_intercept
        )

    def build_constraints(self, instance_data, print_output=False):
        """
        PYOMO
        Constraints of the single-level reformulation: first order optimality
        conditions for lower-level variables: weights and bias of regression
        model
        """
        if print_output:
            print("Building SOS contraints")
        self.sos_constraints = pmo.constraint_dict()
        for psample in range(instance_data.no_poison_samples):
            if print_output:
                print(f"Building SOS contraints for sample {psample}")
            for cat_feature in instance_data.categorical_feature_names:
                if print_output:
                    print(f"Building SOS contraints for feature {cat_feature}")
                constraint = pmo.constraint(
                    sum(
                        self.x_poison_cat[psample, cat_feature, category]
                        for category in instance_data.categories_in_categorical_feature[
                            cat_feature
                        ]
                    )
                    == 1
                )
                self.sos_constraints[psample, cat_feature] = constraint

        if print_output:
            print("Building num weights contraints")
        # There is one constraint per feature
        self.cons_first_order_optimality_conditions_num_weights = (
            pmo.constraint_dict()
        )
        for numfeature in instance_data.numerical_feature_names:
            if print_output:
                print(f"Building num weights contraints for feature {numfeature}")
            constraint = pmo.constraint(
                body=loss_function_derivative_num_weights(
                    instance_data, self, numfeature, self.function
                ),
                rhs=0,
            )
            self.cons_first_order_optimality_conditions_num_weights[
                numfeature
            ] = constraint
        if print_output:
            print("Building cat weights contraints")
        self.cons_first_order_optimality_conditions_cat_weights = (
            pmo.constraint_dict()
        )
        for cat_feature in instance_data.categorical_feature_names:
            for category in instance_data.categories_in_categorical_feature[
                cat_feature
            ]:
                if print_output:
                    print(
                        f"Building cat weights contraints for feature {cat_feature} and category {category}"
                    )
                constraint = pmo.constraint(
                    body=loss_function_derivative_cat_weights(
                        instance_data,
                        self,
                        cat_feature,
                        category,
                        self.function,
                    ),
                    rhs=0,
                )
                self.cons_first_order_optimality_conditions_cat_weights[
                    cat_feature, category
                ] = constraint

        if print_output:
            print("Building bias constraints")
        self.cons_first_order_optimality_conditions_bias = pmo.constraint(
            body=loss_function_derivative_bias(
                instance_data, self, self.function
            ),
            rhs=0,
        )

    def build_objective(self, instance_data):
        """
        PYOMO
        Objective function of single-level reformulation, same as leader's
        objective for bi-level model.
        """

        self.objective_function = pmo.objective(
            expr=train_error(instance_data, self, self.function),
            sense=pyo.maximize,
        )

        # print("Objective has been built")

    def set_poison_data_status(
        self, instance_data, num_feature_flag, cat_feature_flag
    ):
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
        num_feature_flag = np.broadcast_to(
            num_feature_flag,
            (instance_data.no_poison_samples, instance_data.no_numfeatures),
        )
        cat_feature_flag = np.broadcast_to(
            cat_feature_flag,
            (instance_data.no_poison_samples, instance_data.no_catfeatures),
        )
        _poison_data_is_removed = np.any(
            num_feature_flag == self.POISON_DATA_REMOVED, axis=1
        )
        _poison_data_is_removed |= np.any(
            cat_feature_flag == self.POISON_DATA_REMOVED, axis=1
        )
        np.testing.assert_equal(
            _poison_data_is_removed.shape, (instance_data.no_poison_samples,)
        )
        for k, v in enumerate(_poison_data_is_removed):
            self.poison_data_is_removed[k].value = float(v)

        for k, v in instance_data.get_num_x_poison_dataframe().items():
            if num_feature_flag[k[:2]] == self.POISON_DATA_OPTIMIZED:
                self.x_poison_num[k].unfix()
                if self.binary:
                    self.x_poison_num[k].domain = pyo.Binary
            else:
                self.x_poison_num[k].fix(v)
                self.x_poison_num[k].domain = pyo.PercentFraction
                if self.binary:
                    self.x_poison_num[k].domain = pyo.Binary

        for k, v in instance_data.get_cat_x_poison_dataframe().items():
            if cat_feature_flag[k[:2]] == self.POISON_DATA_OPTIMIZED:
                self.x_poison_cat[k].unfix()
                self.sos_constraints[k[0], k[1]].activate()
            else:
                self.x_poison_cat[k].fix(v)
                self.sos_constraints[k[0], k[1]].deactivate()

    def continuous_relaxation(self, mode=True):
        """Create a continuous relaxation of the poisoning attack

        This replaces binary variables with continuous variables.

        Parameters
        ----------
        mode : bool, default True
        """
        if mode:
            for value in self.x_poison_cat.values():
                value.domain = pmo.PercentFraction
        else:
            for value in self.x_poison_cat.values():
                value.domain = pmo.Binary

    def solve(self, neos=False):
        if neos == True:
            self.results = self.opt.solve(
                self, load_solutions=True, tee=self.tee, solver="knitro"
            )
        else:
            self.results = self.opt.solve(
                self, load_solutions=True, tee=self.tee
            )

    def assert_optimal(self):
        results = self.results
        if (results.solver.status == pyomo.opt.SolverStatus.ok) and (
            results.solver.termination_condition
            == pyomo.opt.TerminationCondition.optimal
        ):
            return
        raise ValueError(
            f"status={results.solver.status}   termination_condition={results.solver.termination_condition}"
        )

    def get_mse(self):
        """Get the MSE on the training data after poisoning

        Returns
        -------
        mse : float
        """
        objective = pyo.value(self.objective_function)
        if self.function == "SLS":
            mse = objective / len(self.y_train)
        else:
            mse = objective
        return mse

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
            for k, v in self.x_poison_num.items():
                if (not only_optimized) or (not v.is_fixed()):
                    _x_poison_num.loc[k] = v.value
            index = pd.MultiIndex(
                levels=[[], [], []],
                codes=[[], [], []],
                names=["sample", "feature", "category"],
            )
            _x_poison_cat = pd.Series(index=index)
            for k, v in self.x_poison_cat.items():
                if (not only_optimized) or (not v.is_fixed()):
                    _x_poison_cat.loc[k] = v.value
        else:
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
            index = pd.MultiIndex(
                levels=[[], []], codes=[[], []], names=["sample", "feature"]
            )
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
            # To make wide format dataframes.
            _weights_num = pd.DataFrame()
            for k, v in self.weights_num.items():
                _weights_num[k] = [v.value]
            _weights_cat = pd.DataFrame()
            for k, v in self.weights_cat.items():
                column = f"{k[0]}:{k[1]}"
                _weights_cat[column] = [v.value]
        objective = pyo.value(self.objective_function)
        if self.function == "SLS":
            mse = objective / len(self.y_train)
        else:
            mse = objective
        out = {
            "weights_num": _weights_num,
            "weights_cat": _weights_cat,
            "bias": self.bias.value,
            "objective": objective,
            "mse": mse,
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

    def warmstart(self, warmstart_data):
        # TODO add documentation
        pass


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


def train_error(instance_data, model, function: str):
    """
    Gets mean squared error, which is the mean of sum of the square of the
    difference between predicted values (regression) and target values for all
    samples. MSE = 1 / n * summation( (predicted - target)^2 )
    """

    # Get sum of squared error of regression prediction and target
    sum_square_errors = sum(
        (
            linear_regression_function(instance_data, model, i)
            - model.y_train[i]
        )
        ** 2
        for i in range(instance_data.no_train_samples)
    )

    # Get mean of squared errors
    if function == "MSE":
        return 1 / instance_data.no_train_samples * sum_square_errors
    elif function == "SLS":
        return sum_square_errors


def loss_function_derivative_num_weights(instance_data, model, m, function):
    """
    Finds the derivetive of the loss function (follower's objective) with
    respect to the weights of the linear regression model, and sets it to 0
    (first order optimality condition).
    """

    train_samples_component = sum(
        (
            linear_regression_function(instance_data, model, i)
            - model.y_train[i]
        )
        * model.x_train_num[i, m]
        for i in range(instance_data.no_train_samples)
    )  # Component involving the sum of training samples errors

    poison_samples_component = sum(
        (
            sum(
                model.triple_prod(
                    model.weights_num[j],
                    model.x_poison_num[q, j],
                    model.x_poison_num[q, m],
                )
                for j in instance_data.numerical_feature_names
            )
            + sum(
                sum(
                    model.triple_prod(
                        model.weights_cat[j, z],
                        model.x_poison_cat[q, j, z],
                        model.x_poison_num[q, m],
                    )
                    for z in instance_data.categories_in_categorical_feature[j]
                )
                for j in instance_data.categorical_feature_names
            )
            + model.prod(model.bias, model.x_poison_num[q, m])
            - model.y_poison[q] * model.x_poison_num[q, m]
        )
        # * model.x_poison_num[q, j]
        * (1 - model.poison_data_is_removed[q])
        for q in range(instance_data.no_poison_samples)
    )

    regularization_component = (
        2 * instance_data.regularization * model.weights_num[m]
    )  # Component involving the regularization

    n_train_and_poison_samples = (
        instance_data.no_train_samples + model.no_poison_samples_in_model
    )

    if function == "MSE":
        final = (
            2
            / n_train_and_poison_samples
            * (train_samples_component + poison_samples_component)
            + regularization_component
        )

    elif function == "SLS":
        final = (
            2 * (train_samples_component + poison_samples_component)
            + n_train_and_poison_samples * regularization_component
        )

    return final


def loss_function_derivative_cat_weights(instance_data, model, m, w, function):
    """
    Finds the derivative of the loss function (follower's objective) with
    respect to the weights of the linear regression model, and sets it to 0
    (first order optimality condition).
    """

    train_samples_component = sum(
        (
            linear_regression_function(instance_data, model, i)
            - model.y_train[i]
        )
        * model.x_train_cat[i, m, w]
        for i in range(instance_data.no_train_samples)
    )  # Component involving the sum of training samples errors
    poison_samples_component = sum(
        (
            sum(
                model.triple_prod(
                    model.weights_num[j],
                    model.x_poison_num[q, j],
                    model.x_poison_cat[q, m, w],
                )
                for j in instance_data.numerical_feature_names
            )
            + sum(
                sum(
                    model.triple_prod(
                        model.weights_cat[j, z],
                        model.x_poison_cat[q, j, z],
                        model.x_poison_cat[q, m, w],
                    )
                    for z in instance_data.categories_in_categorical_feature[j]
                )
                for j in instance_data.categorical_feature_names
            )
            + model.prod(model.bias, model.x_poison_cat[q, m, w])
            - model.y_poison[q] * model.x_poison_cat[q, m, w]
        )
        # * model.x_poison_cat[q, j, w]
        * (1 - model.poison_data_is_removed[q])
        for q in range(instance_data.no_poison_samples)
    )  # Component involving the sum of poison samples errors

    regularization_component = (
        2 * instance_data.regularization * model.weights_cat[m, w]
    )  # Component involving the regularization

    n_train_and_poison_samples = (
        instance_data.no_train_samples + model.no_poison_samples_in_model
    )

    if function == "MSE":
        final = (2 / n_train_and_poison_samples) * (
            train_samples_component + poison_samples_component
        ) + regularization_component
    elif function == "SLS":
        final = (
            2 * (train_samples_component + poison_samples_component)
            + n_train_and_poison_samples * regularization_component
        )

    return final


def loss_function_derivative_bias(instance_data, model, function):
    """
    Finds the derivetive of the loss function (follower's objective) with
    respect to the bias of the linear regression model, and sets it to 0 (first
    order optimality condition).
    """

    train_samples_component = sum(
        (
            linear_regression_function(instance_data, model, i)
            - model.y_train[i]
        )
        for i in range(instance_data.no_train_samples)
    )

    poison_samples_component = sum(
        (
            sum(
                model.prod(model.weights_num[j], model.x_poison_num[q, j])
                for j in instance_data.numerical_feature_names
            )
            + sum(
                sum(
                    model.prod(
                        model.weights_cat[j, z], model.x_poison_cat[q, j, z]
                    )
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

    n_train_and_poison_samples = (
        instance_data.no_train_samples + model.no_poison_samples_in_model
    )

    if function == "MSE":
        final = (2 / n_train_and_poison_samples) * (
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
