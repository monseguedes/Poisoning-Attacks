"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Main script for the paper of poisoning attacks of categorical variables.
"""

import sys

sys.path.append("./programs/minlp/model")

# Self-created libraries
import model.model_class as model
from algorithm.solution_approaches import *
from solutions_handler.regression_comparison import *
from algorithm import iterative_attack
from algorithm import ridge_regression

import numpy as np

config = {
    "dataset_name": "5num5cat",
    "no_nfeatures": 0,
    "no_cfeatures": 5,
    "poison_rate": 12,
    "training_samples": 30,
    "seed": 3,
    "function": "SLS",
    "no_psubsets": 3,
    "heuristic_subset": 1,
    "datatype": "test",
    "regularization": 0.6612244897959183,
    "feasibility": 0.00001,
    "time_limit": 20,
    "iterative_attack_n_epochs": 2,
    "iterative_attack_mini_batch_size": 0.1,
    "iterative_attack_incremental": False,
}

from model import pyomo_instance_class

instance_data = pyomo_instance_class.InstanceData(config)

np.testing.assert_equal(
    instance_data.get_num_x_train_dataframe(wide=False).shape, (150,)
)
np.testing.assert_equal(
    instance_data.get_num_x_train_dataframe(wide=True).shape, (30, 5)
)
np.testing.assert_equal(
    instance_data.get_cat_x_train_dataframe(wide=False).shape, (720,)
)
np.testing.assert_equal(
    instance_data.get_cat_x_train_dataframe(wide=True).shape, (30, 24)
)
np.testing.assert_equal(
    instance_data.get_num_x_poison_dataframe(wide=False).shape, (20,)
)
np.testing.assert_equal(
    instance_data.get_num_x_poison_dataframe(wide=True).shape, (4, 5)
)
np.testing.assert_equal(
    instance_data.get_cat_x_poison_dataframe(wide=False).shape, (96,)
)
np.testing.assert_equal(
    instance_data.get_cat_x_poison_dataframe(wide=True).shape, (4, 24)
)

# # Solve models
# bilevel_model, bilevel_instance, bilevel_solution = solve_model('bilevel', config)
# ridge_model, ridge_instance, ridge_solution = solve_model('ridge', config)
benchmark_model, benchmark_instance, benchmark_solution = iterative_attack.run(config)

# Run the utitlity to check the results with scikitlearn.
# Maybe the function take config, instance_data, and a solution
# returned from IterativeAttackModel.get_solution, run ridge regression and
# compare the coefficients.
ridge_regression_solution = ridge_regression.run(config, benchmark_instance)


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

    for key in ["weights_num", "weights_cat", "bias"]:
        a = flatten(sol1[key])
        b = flatten(sol2[key])
        np.testing.assert_allclose(a, b, rtol=1e-6)


assert_solutions_are_close(benchmark_solution, ridge_regression_solution)
print("test passed")

raise SystemExit
(
    bilevel_model,
    bilevel_instance,
    bilevel_solution,
) = benchmark_plus_optimising_heuristic(config)
print(bilevel_solution)

# benchmark_plus_optimising_subset_heuristic(config)
# objectives = []
# _, instance, solution = solve_benchmark(config)
# # objectives.append(solution['objective'])
# instance, solution = flipping_heuristic(config, instance, solution)
# objectives.append(solution['objective'])
# instance, solution = flipping_heuristic(config, instance, solution)
# objectives.append(solution['objective'])

# objectives = np.array(objectives)
# print('objectives')
# print(objectives)
# print('improvement (%)')
# print((objectives[1:] - objectives[0]) / objectives[0] * 100)

# Compare models
# comparison = ComparisonModel(config)
# comparison.compare_everything(bilevel_instance=bilevel_instance, bilevel_model=bilevel_model,
#                               ridge_instance=ridge_instance,ridge_model=ridge_model,
#                               benchmark_instance=benchmark_instance, benchmark_model=benchmark_model)


from sklearn.linear_model import Ridge

df = bilevel_instance.num_x_poison_dataframe
df2 = bilevel_instance.poison_dataframe
y_p = bilevel_instance.y_poison_dataframe
print(y_p)
df4 = pd.DataFrame(dict(x=bilevel_solution["x_poison_num"].values())).T
print(df4)
df2[str(1)] = df.loc[1, 1]
df2[str(2)] = df.loc[1, 2]
df2.columns = range(len(df2.columns))
df4 = bilevel_instance.train_dataframe
df4.columns = range(len(df4.columns))
df3 = pd.concat([df4, df2], axis=0)

X = df3.to_numpy()[:, :-1]
y = df3.to_numpy()[:, -1]

model = Ridge(alpha=bilevel_instance.regularization, fit_intercept=1)
model.fit(X, y)
print(model.coef_)
print(model.intercept_)


def mydevfunc(model, poisoned, function, s):
    """
    Finds the derivetive of the loss function (follower's objective) with respect to
    the numerical weights of the linear regression model (to get first order optimality
    condition).
    """
    if poisoned:
        multiplier = model.x_poison_num
    else:
        multiplier = model.x_data_poison_num

    # Component involving the sum of training samples errors
    train_samples_component = sum(
        sum(
            model.weights_num[r].X * model.x_train_num[i, r]
            for r in model.numfeatures_set
        )
        * model.x_train_num[i, s]
        + sum(
            sum(
                model.weights_cat[j, z].X * model.x_train_cat[i, j, z]
                for z in range(1, model.no_categories[j] + 1)
            )
            for j in model.catfeatures_set
        )
        * model.x_train_num[i, s]
        + model.bias.X * model.x_train_num[i, s]
        - model.y_train[i] * model.x_train_num[i, s]
        for i in model.samples_set
    )

    # Component involving the sum of poison samples errors
    poison_samples_component = sum(
        sum(model.tnn_ln_times_numsamples[k, r, s].X for r in model.numfeatures_set)
        + sum(
            sum(
                model.tcn_lc_times_numsamples[k, j, z, s].X
                for z in range(1, model.no_categories[j] + 1)
            )
            for j in model.catfeatures_set
        )
        + model.bias.X * multiplier[k, s].X
        - model.y_poison[k] * multiplier[k, s].X
        for k in model.psamples_set
    )

    # Component involving regularization
    regularization_component = 2 * model.regularization * model.weights_num[s].X

    if function == "MSE":
        derivative_num_weights = (2 / (model.no_samples + model.no_psamples)) * (
            train_samples_component + poison_samples_component
        ) + regularization_component

    elif function == "SLS":
        derivative_num_weights = (
            2 * (train_samples_component + poison_samples_component)
            + regularization_component
        )

    print(derivative_num_weights)


# mydevfunc(bilevel_model, True, "MSE", 1)
# mydevfunc(bilevel_model, True, "MSE", 2)
