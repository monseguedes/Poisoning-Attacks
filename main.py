"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Main script for the paper of poisoning attacks of categorical variables. 
"""

# Self-created libraries
import model.model_class as model
from algorithm.solution_approaches import *
from solutions_handler.regression_comparison import * 


model_parameters = {'dataset_name': '1num1cat',
                    'no_nfeatures': int,
                    'no_cfeatures': int, 
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
comparison.compare_everything(bilevel_instance=bilevel_instance, bilevel_model=bilevel_model,
                              ridge_instance=ridge_instance,ridge_model=ridge_model,
                              benchmark_instance=benchmark_instance, benchmark_model=benchmark_model)

