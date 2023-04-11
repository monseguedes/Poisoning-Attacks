"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Main script for the paper of poisoning attacks of categorical variables. 
"""

# Self-created libraries
import model.model_class as model
from algorithm.solution_approaches import *
from solutions_handler.regression_comparison import * 

model_parameters = {'dataset_name': '2num2cat',
                    'no_nfeatures': 0,
                    'no_cfeatures': 0, 
                    'poison_rate': 0,
                    'training_samples': 5,
                    'seed': 2,
                    'function': 'SLS',
                    'no_psubsets': 0,
                    'datatype': 'test',
                    'feasibility': 0.0001,
                    'time_limit': 100}

# # Solve models
bilevel_model, bilevel_instance, bilevel_solution = solve_model('bilevel', model_parameters)
# ridge_model, ridge_instance, ridge_solution = solve_model('ridge', model_parameters)
benchmark_model, benchmark_instance, benchmark_solution = solve_model('benchmark', model_parameters)
# benchmark_plus_optimising_heuristic(model_parameters)
# benchmark_plus_optimising_subset_heuristic(model_parameters)
# flipping_heuristic(model_parameters)

# Compare models
# comparison = ComparisonModel(model_parameters)
# comparison.compare_everything(bilevel_instance=bilevel_instance, bilevel_model=bilevel_model,
#                               ridge_instance=ridge_instance,ridge_model=ridge_model,
#                               benchmark_instance=benchmark_instance, benchmark_model=benchmark_model)

# from sklearn.linear_model import Ridge

# df = benchmark_instance.num_x_poison_dataframe
# df2 = benchmark_instance.poison_dataframe
# df2[str(1)] = df.loc[1,1]
# df2[str(2)] = df.loc[1,2]
# df2.columns = range(len(df2.columns))
# df4 = benchmark_instance.train_dataframe
# df4.columns = range(len(df4.columns))
# df3 = pd.concat([df4,df2], axis=0)

# X = df3.to_numpy()[:, :-1]
# y = df3.to_numpy()[:, -1]

# model = Ridge(alpha=benchmark_instance.regularization,fit_intercept=1)
# model.fit(X, y)
# print(model.coef_)
# print(model.intercept_)