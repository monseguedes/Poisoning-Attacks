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
                    'training_samples': 10,
                    'seed': 2,
                    'function': 'MSE',
                    'no_psubsets': 1,
                    'datatype': 'test'}

# Solve models
bilevel_model, bilevel_instance, bilevel_solution = solve_model('bilevel', model_parameters)
ridge_model, ridge_instance, ridge_solution = solve_model('ridge', model_parameters)
benchmark_model, benchmark_instance, benchmark_solution = solve_model('benchmark', model_parameters)

# Compare models
comparison = ComparisonModel(model_parameters)
comparison.make_poisoned_predictions(bilevel_instance=bilevel_instance,
                                     bilevel_model=bilevel_model)
comparison.make_non_poisoned_predictions(ridge_instance=ridge_instance,
                                         ridge_model=ridge_model)
comparison.make_benchmark_predictions(benchmark_intance=benchmark_instance,
                                      benchmark_model=benchmark_model)
comparison.plot_actual_vs_pred('bilevel')
comparison.plot_actual_vs_pred('benchmark')
comparison.plot_actual_vs_predicted_all()
comparison.store_comparison_metrics()
