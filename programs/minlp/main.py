"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Main script for the paper of poisoning attacks of categorical variables. 
"""

# Self-created libraries
import model.model_class as model
from algorithm.solution_approaches import *
from solutions_handler.regression_comparison import * 

import numpy as np

model_parameters = {'dataset_name': '10num10cat',
                    'no_nfeatures': 0,
                    'no_cfeatures': 5, 
                    'poison_rate': 20,
                    'training_samples': 100,
                    'seed': 3,
                    'function': 'MSE',
                    'no_psubsets': 1,
                    'heuristic_subset': 1,
                    'datatype': 'test',
                    'feasibility': 0.00001,
                    'time_limit': 100}

# # Solve models
# bilevel_model, bilevel_instance, bilevel_solution = solve_model('bilevel', model_parameters)
# ridge_model, ridge_instance, ridge_solution = solve_model('ridge', model_parameters)
# benchmark_model, benchmark_instance, benchmark_solution = solve_model('benchmark', model_parameters)
benchmark_plus_optimising_heuristic(model_parameters)
raise SystemExit
# benchmark_plus_optimising_subset_heuristic(model_parameters)
objectives = []
_, instance, solution = solve_benchmark(model_parameters)
# objectives.append(solution['objective'])
instance, solution = flipping_heuristic(model_parameters, instance, solution)
# objectives.append(solution['objective'])
# instance, solution = flipping_heuristic(model_parameters, instance, solution)
# objectives.append(solution['objective'])

# objectives = np.array(objectives)
# print('objectives')
# print(objectives)
# print('improvement (%)')
# print((objectives[1:] - objectives[0]) / objectives[0] * 100)

# Compare models
# comparison = ComparisonModel(model_parameters)
# comparison.compare_everything(bilevel_instance=bilevel_instance, bilevel_model=bilevel_model,
#                               ridge_instance=ridge_instance,ridge_model=ridge_model,
#                               benchmark_instance=benchmark_instance, benchmark_model=benchmark_model)
raise SystemExit

from sklearn.linear_model import Ridge

df = bilevel_instance.num_x_poison_dataframe
df2 = bilevel_instance.poison_dataframe
y_p = bilevel_instance.y_poison_dataframe
print(y_p)
df4 = pd.DataFrame(dict(x=bilevel_solution["x_poison_num"].values())).T
print(df4)
df2[str(1)] = df.loc[1,1]
df2[str(2)] = df.loc[1,2]
df2.columns = range(len(df2.columns))
df4 = bilevel_instance.train_dataframe
df4.columns = range(len(df4.columns))
df3 = pd.concat([df4,df2], axis=0)

X = df3.to_numpy()[:, :-1]
y = df3.to_numpy()[:, -1]

model = Ridge(alpha=bilevel_instance.regularization,fit_intercept=1)
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

    #Component involving the sum of training samples errors
    train_samples_component = sum(
                                          sum(model.weights_num[r].X * model.x_train_num[i, r] 
                                          for r in model.numfeatures_set)
                                          * model.x_train_num[i,s] \
                                          + sum(sum(model.weights_cat[j, z].X * model.x_train_cat[i, j, z] 
                                                      for z in range(1, model.no_categories[j] + 1)) 
                                            for j in model.catfeatures_set) \
                                          * model.x_train_num[i,s] \
                                          + model.bias.X * model.x_train_num[i,s] \
                                          - model.y_train[i] * model.x_train_num[i,s]  \
                              for i in model.samples_set)                      

    #Component involving the sum of poison samples errors
    poison_samples_component = sum(
                                          sum(model.tnn_ln_times_numsamples[k,r,s].X
                                          for r in model.numfeatures_set)
                                          + sum(sum(model.tcn_lc_times_numsamples[k,j,z,s].X
                                                        for z in range(1, model.no_categories[j] + 1)) 
                                            for j in model.catfeatures_set) \
                                          + model.bias.X * multiplier[k,s].X \
                                          - model.y_poison[k] * multiplier[k,s].X  \
                              for k in model.psamples_set)
                                
    #Component involving regularization
    regularization_component = 2 * model.regularization * model.weights_num[s].X

    if function == 'MSE':
      derivative_num_weights = (2 / (model.no_samples + model.no_psamples)) \
                              * (train_samples_component + poison_samples_component) \
                              + regularization_component

    elif function == 'SLS':
      derivative_num_weights = 2 * (train_samples_component + poison_samples_component) \
                              + regularization_component
                        
    print(derivative_num_weights)


# mydevfunc(bilevel_model, True, "MSE", 1)
# mydevfunc(bilevel_model, True, "MSE", 2)