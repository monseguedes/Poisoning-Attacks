"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Main script for the paper of poisoning attacks of categorical variables. 
"""

# Self-created libraries
import model.model_class as model
from algorithm.solution_approaches import *
from solutions_handler.regression_comparison import * 

# Python Libraries
import gurobipy as gp
import os
import pandas as pd


model_parameters = {'dataset_name': '1num1cat', 
                    'poison_rate': 20,
                    'training_samples': 300,
                    'seed': 2,
                    'function': 'MSE',
                    'no_psubsets': 2,
                    'datatype': 'test'}

folder = '_'.join([model_parameters['dataset_name'], 
                    str(model_parameters['poison_rate']),
                    str(model_parameters['training_samples']),
                    str(model_parameters['no_psubsets']),
                    str(model_parameters['seed'])])

# TODO: have this code inside solution and plot generation.
if not os.path.exists(os.path.join('results/scores', folder)):
    os.mkdir(os.path.join('results/scores', folder))

if not os.path.exists(os.path.join('plots', folder)):   
    os.mkdir(os.path.join('plots', folder))

# Solve models
bilevel_model, bilevel_instance, bilevel_solution = solve_model('bilevel', model_parameters)
ridge_model, ridge_instance, ridge_solution = solve_model('ridge', model_parameters)
benchmark_model, benchmark_instance, benchmark_solution = solve_model('benchmark', model_parameters)

# comparison = ComparisonModel(bilevel_instance_data=bilevel_instance,
#                             bilevel_model=bilevel_model,
#                             ridge_instance_data=ridge_instance,
#                             ridge_model=ridge_model,
#                             datatype=model_parameters['datatype'],
#                             folder=folder)

# comparison.make_poisoned_predictions()
# comparison.make_non_poisoned_predictions()
# comparison.make_benchmark_predictions(benchmark_model=benchmark_model, benchmark_intance=benchmark_instance)
# comparison.plot_actual_vs_pred()
# comparison.plot_actual_vs_pred_benchmark()
# comparison.plot_actual_vs_predicted_all()
# comparison.store_comparison_metrics()
