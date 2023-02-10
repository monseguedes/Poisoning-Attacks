"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Auxiliary module of package 'algorithm' with functions to apply solution approaches 
and solve models.

- solving_MINLP
- ridge_regression
- single_attack_strategy
- iterative_attack_strategy

"""

# Self-created modules
from model.model_class import *
import model.instance_class as data
import model.pyomo_instance_class as benchmark_data


# Python Libraries
from os import path
import pandas as pd
import numpy as np
import os
import gurobipy as gp
import time
import csv
import matplotlib.pyplot as plt
import pyomo.environ as pyo


# Main Functions to solve model

def solving_MINLP(dataset_name: str, 
                  poison_rate: int, 
                  training_samples: int,
                  seed: int,
                  function = 'MSE'):
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
    my_model = PoisonAttackModel(gurobi_model, instance_data, function=function)
    m = my_model.model
    print('Model has been built')

    def data_cb(model, where):
        if where == gp.GRB.Callback.MIP:
            cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

            # Did objective value or best bound change?
            if model._obj != cur_obj or model._bd != cur_bd:
                model._obj = cur_obj
                model._bd = cur_bd
                model._data.append([time.time() - m._start, cur_obj, cur_bd])

    # Solve model
    m._obj = None
    m._bd = None
    m._data = []
    m._start = time.time()

    print('Solving the model...')
    m.params.NonConvex = 2
    m.params.FeasibilityTol = 0.0001
    m.params.TimeLimit = 600
    results = m.optimize(callback=data_cb)

    file_name = '_'.join(['bounds',
                          str(my_model.no_numfeatures),
                          str(my_model.no_catfeatures),
                          str(len(instance_data.x_train_dataframe)),
                          str(int(instance_data.poison_rate * 100))])

    with open('bounds/' + file_name + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(m._data)

    # Plot bounds
    bounds_df = pd.read_csv('bounds/' + file_name + '.csv', names=['Time', 'Objective', 'Bound'], header=None)
    bounds_df = bounds_df.iloc[5:]
    bounds_df.set_index('Time', drop=True)
    bounds_df[['Objective', 'Bound']].plot(marker='.')
    plt.title('Evolution of Incumbent and Upper-Bound')
    plt.ylabel('Objective value')
    plt.xlabel('Time (s)')
    plt.savefig('bounds/' + 'plot_' + file_name + '.png')
    plt.close()

    print('Model has been solved')

    m.write('out.sol')
    # m.read('out.sol')
    # m.update()
    # results = m.optimize()

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
                     function='MSE',
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
    my_model = RegressionModel(gurobi_model, instance_data, function=function)
    m = my_model.model
    print('Model has been built')

    print('Solving the model...')
    # Solve model
    # m.params.NonConvex = 2
    m.params.FeasibilityTol = 0.0001
    results = m.optimize()
    print('Model has been solved')

    ###
    solutions_dict = {'weights': [element.X for element in my_model.weights.values()],
                      'bias': my_model.bias.X}

    print('Objective value is ', m.ObjVal)

    return my_model, instance_data, solutions_dict

def single_attack_strategy(opt: pyo.SolverFactory, 
                           dataset_name: str, 
                           poison_rate: int, 
                           seed: int, 
                           initialized_solution=0):
    """
    Algorithm for single attack strategy of benchmark paper.

    Input parameters are: dataframe name, poison rate, seed (to control randomness), and initialized solution.
    """

    # Initialise data (ready for building first instance)
    instance_data = benchmark_data.InstanceData(dataset_name=dataset_name, 
                                                seed=seed)
    instance_data.prepare_instance(poison_rate=poison_rate, 
                                   N=1)

    # Create abstract model
    model = BenchmarkPoisonAttackModel(instance_data)
    
    print('Model has been built')

    print('Solving the model...')
    # Solve model
    results = opt.solve(model, load_solutions=True, tee=True)
    # results = solver_manager.solve(model, load_solutions=True, tee=True, opt=opt)
    print('Model has been solved')

    ### Store results of the poison subset found during this iteration     
    index = pd.MultiIndex.from_tuples(model.x_poison.keys(), names=('sample', 'feature'))   # Create index from the keys (indexes) of the solutions of x_poison
    poison_solution = pd.Series(model.x_poison.values(), index=index)  # Make a dataframe with solutions and desires index
    new_x_train = poison_solution
    new_x_train.name = 'x_train'
    ###

    solutions_dict = {'x_poison': poison_solution.to_dict(),
                     'weights': model.weights.value,
                     'bias': model.bias.value}
    
    return model, instance_data, solutions_dict

def iterative_attack_strategy(opt: pyo.SolverFactory, 
                              dataset_name: str, 
                              poison_rate: int,
                              training_samples: int, 
                              no_psubsets: int, 
                              seed: int, 
                              initialized_solution=0):
    """
    Algorithm for iterative attack strategy. 

    It starts by creating the abstract model, and an initial data object for creating the first 
    instance. After this, while the iteration count is smaller than the number of subsets (there
    is an iteration per subset), the model instance is created with the intance data object and the 
    model is solved for current instance. After that, solutions are stored in a dataframe, and data
    object for instance is updated to that current iteration becomes data. Then, we go back to start
    of while loop and process is repeated for all subsets/iterations.
    """

    # Initializa data (ready for building first instance)
    instance_data = benchmark_data.InstanceData(dataset_name=dataset_name)
    instance_data.prepare_instance(poison_rate=poison_rate,
                                   training_samples=training_samples,
                                   N=no_psubsets, 
                                   seed=seed)

    # Iteration count
    iteration = instance_data.iteration_count

    iterations_solutions = []
    
    while iteration <= no_psubsets: # There is an iteration for each poison subset
        # Build instance for current iteration data
        # Create abstract model
        model = BenchmarkPoisonAttackModel(instance_data)

        # Solve model
        results = opt.solve(model, load_solutions=True, tee=True)

        ### Store results of the poison subset found during this iteration     
        index = pd.MultiIndex.from_tuples(model.x_poison_num.keys(), names=('sample', 'feature'))   # Create index from the keys (indexes) of the solutions of x_poison
        poison_solution = pd.Series([variable._value for variable in model.x_poison_num.values()], index=index)  # Make a dataframe with solutions and desires index
        new_x_train_num = poison_solution
        new_x_train_num.name = 'x_train_num'
        ###

        solutions_dict = {'x_poison_num': poison_solution.to_dict(),
                         'weights_num': model.weights_num,
                         'weights_cat': model.weights_cat,
                         'bias': model.bias.value}
        iterations_solutions.append(solutions_dict)

        # Modify data dataframes with results
        instance_data.update_data(new_x_train_num=new_x_train_num)

        iteration = instance_data.iteration_count
        print('Iteration no. {} is finished'.format(iteration - 1))
        print('Objective value is ', pyo.value(model.objective_function))

    solutions = {'iteration no.' + str(iteration): solution for iteration, solution in enumerate(iterations_solutions)}

    return model, instance_data, solutions

