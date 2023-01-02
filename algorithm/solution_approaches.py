"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Auxiliary module of package 'algorithm' with functions to apply solution approaches 
and solve models.
"""

# Self-created modules
from model.model_class import *
from model.ridge_model_class import *
import model.instance_class as data

# Python Libraries
from os import path
import pandas as pd
import numpy as np
import os
import gurobipy as gp
import time
import csv
import matplotlib.pyplot as plt


# Main Functions to solve model

def solving_MINLP(dataset_name: str, 
                  poison_rate: int, 
                  training_samples: int,
                  seed: int):
    """
    Algorithm for solving the MINLP bilevel model.
    """

    # Initialise data (ready for building first instance)
    instance_data = data.InstanceData(dataset_name=dataset_name)
    instance_data.prepare_instance(poison_rate=poison_rate, 
                                   training_samples=training_samples,
                                   seed=seed)
    
    # Create model
    gurobi_model = gp.Model('Poisoning_Attack')
    my_model = PoisonAttackModel(gurobi_model, instance_data)
    m = my_model.model
    print('Model has been built')

    # Solve model
    print('Solving the model...')
    m.params.NonConvex = 2
    m.params.FeasibilityTol = 0.01
    results = m.optimize()
    print('Model has been solved')

    ### Store results
    index = pd.MultiIndex.from_tuples(my_model.x_poison_num.keys(), 
                                      names=('sample', 'feature'))   # Create index from the keys (indexes) of the solutions of x_poison_num
    num_poison_solution = pd.Series([element.X for element in my_model.x_poison_num.values()], 
                                    index=index)  # Make a dataframe with solutions and desires index

    index = pd.MultiIndex.from_tuples(my_model.x_poison_cat.keys(), 
                                      names=('sample', 'feature', 'category'))   # Create index from the keys (indexes) of the solutions of x_poison_cat
    cat_poison_solution = pd.Series([element.X for element in my_model.x_poison_cat.values()], 
                                    index=index)  # Make a dataframe with solutions and desires index

    num_poison_solution.name = 'x_train_num'
    cat_poison_solution.name = 'x_train_cat'

    solutions_dict = {'x_poison_num': num_poison_solution.to_dict(),
                      'x_poison_cat': cat_poison_solution.to_dict(),
                      'weights_num': [element.X for element in my_model.weights_num.values()],
                      'weights_cat': [element.X for element in my_model.weights_cat.values()],
                      'bias': my_model.bias.X}

    print('Objective value is ', m.ObjVal)

    return my_model, instance_data, solutions_dict
    
def ridge_regression(dataset_name: str,
                     training_samples: int, 
                     seed: int, 
                     initialized_solution=0,
                     poisoned=False,
                     poison_solutions=None,
                     bilevel_instance=None):
    """
    Fits a fit regression model. The goal of this function is to be able to 
    compare the performance of poisoned and nonpoisoned models.
    """

    # Initialise data (ready for building first instance)
    instance_data = data.InstanceData(dataset_name=dataset_name)
    instance_data.create_dataframes(training_samples=training_samples, seed=seed)
    instance_data.split_dataframe()
    instance_data.regularization_parameter()

    if poisoned == True:   # Take as data the training samples + poisoning samples instead
        instance_data = bilevel_instance
        instance_data.append_poisoning_attacks(poison_solutions)
    
    # Create model
    gurobi_model = gp.Model('Regression_Model')
    my_model = RegressionModel(gurobi_model, instance_data)
    m = my_model.model
    print('Model has been built')

    print('Solving the model...')
    # Solve model
    m.params.NonConvex = 2
    m.params.FeasibilityTol = 0.01
    results = m.optimize()
    print('Model has been solved')

    ###
    solutions_dict = {'weights': [element.X for element in my_model.weights.values()],
                      'bias': my_model.bias.X}

    print('Objective value is ', m.ObjVal)

    return my_model, instance_data, solutions_dict






