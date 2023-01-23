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

model_parameters = {'dataset_name': 'smallvtest', 
                    'poison_rate': 20,
                    'training_samples': 12,
                    'seed': 0}


# Solve using bilevel model
bilevel_model, bilevel_instance, bilevel_solutions = solving_MINLP(dataset_name=model_parameters['dataset_name'],
                                                                   poison_rate=model_parameters['poison_rate'],
                                                                   training_samples=model_parameters['training_samples'],
                                                                   seed=model_parameters['seed'])

pridge_model, pridge_instance, pridge_solutions = ridge_regression(dataset_name=model_parameters['dataset_name'],
                                                                   training_samples=model_parameters['training_samples'],
                                                                   seed=model_parameters['seed'],
                                                                   poisoned=True,
                                                                   poison_solutions=bilevel_solutions,
                                                                   bilevel_instance=bilevel_instance)

ridge_model, ridge_instance, ridge_solutions = ridge_regression(dataset_name=model_parameters['dataset_name'],
                                                                training_samples=model_parameters['training_samples'],
                                                                seed=model_parameters['seed'])

comparison = ComparisonModel(bilevel_instance_data=bilevel_instance,
                             bilevel_model=bilevel_model,
                             ridge_instance_data=ridge_instance,
                             ridge_model=ridge_model)

comparison.make_poisoned_predictions()
comparison.make_non_poisoned_predictions()
comparison.plot_actual_vs_pred()
comparison.store_comparison_metrics()
print(bilevel_solutions)
print(pridge_solutions)
print(bilevel_model.upper_bound)
