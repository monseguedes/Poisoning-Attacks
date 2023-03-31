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
def solve_model(model_name: str, model_parameters: dict, checking_bilevel=False):
    """
    Function that solves any of the models
    """
    model_functions = {
        'bilevel': solve_bilevel,
        'ridge': solve_ridge,
        'benchmark': solve_benchmark
    }

    if model_name in model_functions:
        model_func = model_functions[model_name]
        model, intance, solution = model_func(model_parameters, checking_bilevel)

    return model, intance, solution

def solve_bilevel(model_parameters: dict, checking_bilevel: bool):
    """
    Function to solve and store bilevel results
    """
    # Solve using bilevel model
    bilevel_model, bilevel_instance, bilevel_solution = solving_MINLP(dataset_name=model_parameters['dataset_name'],
                                                                      no_nfeatures=model_parameters['no_nfeatures'],
                                                                      no_cfeatures=model_parameters['no_cfeatures'], 
                                                                      poison_rate=model_parameters['poison_rate'],
                                                                      training_samples=model_parameters['training_samples'],
                                                                      seed=model_parameters['seed'],
                                                                      function=model_parameters['function'])

    if checking_bilevel:
        # To check that solutions of lower level are optimal
        pridge_model, pridge_instance, pridge_solution = solve_ridge(dataset_name=model_parameters['dataset_name'],
                                                                     training_samples=model_parameters['training_samples'],
                                                                     seed=model_parameters['seed'],
                                                                     function=model_parameters['function'],
                                                                     poisoned=True,
                                                                     poison_solutions=bilevel_solution,
                                                                     bilevel_instance=bilevel_instance)
        
    return bilevel_model, bilevel_instance, bilevel_solution

def solve_ridge(model_parameters: dict, checking_bilevel: bool):
    """
    Function to solve and store ridge results
    """
    # Normal ridge regression
    ridge_model, ridge_instance, ridge_solution = ridge_regression(dataset_name=model_parameters['dataset_name'],
                                                                            training_samples=model_parameters['training_samples'],
                                                                            seed=model_parameters['seed'],
                                                                            function=model_parameters['function'])

    return ridge_model, ridge_instance, ridge_solution

def solve_benchmark(model_parameters: dict, checking_bilevel: bool):
    """
    Function to solve and store benchmark results
    """
    # Solve benchmark 
    opt = pyo.SolverFactory('ipopt')
    benchmark_model, benchmark_instance, benchmark_solution = iterative_attack_strategy(opt=opt, 
                                                                                        dataset_name=model_parameters['dataset_name'], 
                                                                                        poison_rate=model_parameters['poison_rate'],
                                                                                        training_samples=model_parameters['training_samples'],
                                                                                        no_psubsets = model_parameters['no_psubsets'], 
                                                                                        seed=model_parameters['seed'])
    
    return benchmark_model, benchmark_instance, benchmark_solution


# Algorithmic approaches
def solving_MINLP(dataset_name: str,
                  no_nfeatures: int,
                  no_cfeatures: int, 
                  poison_rate: int, 
                  training_samples: int,
                  seed: int,
                  function = 'MSE',
                  feasibility=0.0001,
                  time_limit=600):
    """
    Algorithm for solving the MINLP bilevel model.
    """

    # Initialise data (ready for building first instance)
    instance_data = data.InstanceData(dataset_name=dataset_name)
    instance_data.prepare_instance(poison_rate=poison_rate, 
                                   training_samples=training_samples,
                                   no_nfeatures=no_nfeatures,
                                   no_cfeatures=no_cfeatures, 
                                   seed=seed)
    
    # Create model
    gurobi_model = gp.Model('Poisoning_Attack')
    my_model = PoisonAttackModel(gurobi_model, instance_data, function=function)
    m = my_model.model

    # Callback function for bounds
    def data_cb(model, where):
        if where == gp.GRB.Callback.MIP:
            cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

            # Did objective value or best bound change?
            if model._obj != cur_obj or model._bd != cur_bd:
                model._obj = cur_obj
                model._bd = cur_bd
                model._data.append([time.time() - m._start, cur_obj, cur_bd])

    # Prepare objects for callback
    m._obj = None
    m._bd = None
    m._data = []
    m._start = time.time()

    print('Solving model...')
    m.params.NonConvex = 2
    m.params.FeasibilityTol = feasibility
    m.params.TimeLimit = time_limit
    results = m.optimize(callback=data_cb)
    print('Model has been solved')
    # m.write('out.lp')

    # Save bounds in file
    file_name = '_'.join(['bounds',
                          str(my_model.no_numfeatures),
                          str(my_model.no_catfeatures),
                          str(len(instance_data.x_train_dataframe)),
                          str(int(instance_data.poison_rate * 100))])
    with open('results/bounds/files/' + file_name + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(m._data)
    
    def plot_bounds(file_name):
        """
        Function to plot upper and lower bounds.
        """
        # Plot bounds
        bounds_df = pd.read_csv('results/bounds/files/' + file_name + '.csv', names=['Time', 'Primal', 'Dual'], header=None)
        bounds_df = bounds_df.iloc[5:]
        bounds_df.set_index('Time', drop=True, inplace=True)
        bounds_df[['Primal', 'Dual']].plot(style={'Primal': '-k', 'Dual': '--k'}, figsize=(8.5,6.5))
        plt.title('Evolution of Primal and Dual Bounds', fontsize=30)
        plt.ylabel('Bounds', fontsize=20)
        plt.xlabel('Time (s)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig('results/bounds/plots/' + 'plot_' + file_name + '.pdf', transparent=True, bbox_inches = "tight")
        #plt.show()
        plt.close()

    plot_bounds(file_name)

    def initialise_opt_solution(m):
        """
        Initialise model with known optimal solution.
        """
        m.write('out.sol')
        m.read('out.sol')
        m.update()
        results = m.optimize()

        return results
    
    # results = initialise_opt_solution(m)

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
                              seed: int):
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

    model = BenchmarkPoisonAttackModel(instance_data)

    # Initialise solutions
    for psample, numfeature in itertools.product(model.psamples_set, model.numfeatures_set):
        model.x_poison_num[psample, numfeature].value = instance_data.num_x_poison_dataframe.to_dict()[psample, numfeature]

    while iteration <= no_psubsets: # There is an iteration for each poison subset
        # Build instance for current iteration data
        # Create model
        #model = BenchmarkPoisonAttackModel(instance_data)
        #model.x_train_num[1,1].value = instance_data.num_x_train_dataframe.to_dict()[1,1]

        # Solve model
        results = opt.solve(model, load_solutions=True, tee=True)

        ### Store results of the poison subset found during this iteration
        index = pd.MultiIndex.from_tuples(model.x_poison_num.keys(), names=('sample', 'feature'))   # Create index from the keys (indexes) of the solutions of x_poison
        poison_solution = pd.Series([variable._value for variable in model.x_poison_num.values()], index=index)  # Make a dataframe with solutions and desires index
        new_x_train_num = poison_solution
        new_x_train_num.name = 'x_train_num'
        ###

        solutions_dict = {'x_poison_num': poison_solution.to_dict(),
                         'weights_num': {index : model.weights_num[index].value for index in model.numfeatures_set},
                         'weights_cat': {(cat_feature,category) : model.weights_cat[(cat_feature,category)].value for cat_feature in model.catfeatures_set for category in model.categories_sets[cat_feature]},
                         'bias': model.bias.value}
        iterations_solutions.append(solutions_dict)

        # Modify data dataframes with results
        instance_data.update_data(new_x_train_num=new_x_train_num)

        iteration = instance_data.iteration_count
        print('Iteration no. {} is finished'.format(iteration - 1))
        print('Objective value is ', pyo.value(model.objective_function))

    solutions = {'iteration no.' + str(iteration + 1): solution for iteration, solution in enumerate(iterations_solutions)}

    return model, instance_data, solutions

def benchmark_plus_optimising_heuristic(opt: pyo.SolverFactory, 
                                        model_parameters: dict, 
                                        feasibility=0.0001,
                                        time_limit=600):
    """
    This is the heuristic algorithm we use to get feasible 
    solutions. It works as follows. We quickly optimise numerical 
    features (locally) using ipopt. Then take these solutions and 
    matrix accordingly. Then we optimize subsets of features on top
    of this first optimization. Features are chosen using LASSO.
    """
    
    # Solve model using benchmark method.
    benchmark_model, benchmark_instance, benchmark_solution = iterative_attack_strategy(opt=opt, 
                                                                                        dataset_name=model_parameters['dataset_name'], 
                                                                                        poison_rate=model_parameters['poison_rate'],
                                                                                        training_samples=model_parameters['training_samples'],
                                                                                        no_psubsets = model_parameters['no_psubsets'], 
                                                                                        seed=model_parameters['seed'])

    # Add poisoning samples as data


    # Optimise now categorical features separately (MINLP)
    instance_data = data.InstanceData(dataset_name=model_parameters['dataset_name'])
    instance_data.prepare_instance(poison_rate=model_parameters['poison_rate'],
                                   training_samples=model_parameters['training_samples'],
                                   no_nfeatures='all',
                                   no_cfeatures='all', 
                                   seed=model_parameters['seed'])
    
    # Change the necessary dataframes to have benchmark as columns in data dataframes (x_poison_dataframe)
    # update x_poison_dataframe
    instance_data.feature_selection(no_nfeatures=model_parameters['no_nfeatures'],
                                    no_cfeatures=model_parameters['no_cfeatures'])
    instance_data.poison_samples()

    # Create model
    gurobi_model = gp.Model('Poisoning_Attack')
    my_model = PoisonAttackModel(gurobi_model, instance_data, function=function)
    m = my_model.model

    print('Solving model...')
    m.params.NonConvex = 2
    m.params.FeasibilityTol = feasibility
    m.params.TimeLimit = time_limit
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