"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Collection of auxiliary functions to define the objective and constraints of the model in model_class.
"""

import numpy as np
from statistics import mean
import gurobipy as gp

def objective_function(model, function):
    """
    Defines the objective function expression. If function is MSE, it calculates the mean 
    squared error. If function is SLS, it finds the sum of least squared. The only difference 
    is whether the objective is divided by the number of samples. 
    MSE = 1 / n * summation((predicted - target)^2) 
    SLS = summation((predicted - target)^2) 
    This is the attacker's objective, so it only uses training data.
    """

    #Get sum of squared error of regression prediction and target
    sum_square_errors = gp.quicksum((gp.quicksum(model.weights_num[r] * model.x_train_num[k, r] for r in model.numfeatures_set) 
                        + gp.quicksum(gp.quicksum(model.weights_cat[j, z] * model.x_train_cat[k, j, z] for z in range(1, model.no_categories[j] + 1)) 
                                                                                                       for j in model.catfeatures_set) 
                        + model.bias 
                        - model.y_train[k]) ** 2 
                        for k in model.samples_set)

    if function == 'MSE':
      obj = 1 / model.no_samples * sum_square_errors

    elif function == 'SLS':
      obj = sum_square_errors

    return obj

def SOS_constraints(model, k, j):
    """
    SOS1 contraint that ensures that one and only one category by summing over all 
    categories within a categorical feature and making the sum equal to 1.
    """
    sum_of_categories = sum(model.x_poison_cat[k,j,i] for i in range(1, model.no_categories[j] + 1))

    return sum_of_categories 

def loss_function_derivative_num_weights(model, poisoned, function, s):
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
    train_samples_component = gp.quicksum(
                                          gp.quicksum(model.weights_num[r] * model.x_train_num[i, r] 
                                          for r in model.numfeatures_set)
                                          * model.x_train_num[i,s] \
                                          + gp.quicksum(gp.quicksum(model.weights_cat[j, z] * model.x_train_cat[i, j, z] 
                                                      for z in range(1, model.no_categories[j] + 1)) 
                                            for j in model.catfeatures_set) \
                                          * model.x_train_num[i,s] \
                                          + model.bias * model.x_train_num[i,s] \
                                          - model.y_train[i] * model.x_train_num[i,s]  \
                              for i in model.samples_set)                      

    #Component involving the sum of poison samples errors
    poison_samples_component = gp.quicksum(
                                          gp.quicksum(model.tnn_ln_times_numsamples[k,r,s]
                                          for r in model.numfeatures_set)
                                          + gp.quicksum(gp.quicksum(model.tcn_lc_times_numsamples[k,j,z,s]
                                                        for z in range(1, model.no_categories[j] + 1)) 
                                            for j in model.catfeatures_set) \
                                          + model.bias * multiplier[k,s] \
                                          - model.y_poison[k] * multiplier[k,s]  \
                              for k in model.psamples_set)
                                
    #Component involving regularization
    regularization_component = 2 * model.regularization * model.weights_num[s] 

    if function == 'MSE':
      derivative_num_weights = (2 / (model.no_samples + model.no_psamples)) \
                              * (train_samples_component + poison_samples_component) \
                              + regularization_component

    elif function == 'SLS':
      derivative_num_weights = 2 * (train_samples_component + poison_samples_component) \
                              + regularization_component
                        
    return  derivative_num_weights

def loss_function_derivative_cat_weights(model, poisoned, function, l, h):
    """
    Finds the derivetive of the loss function (follower's objective) with respect to 
    the categorical weights of the linear regression model (to get first order optimality 
    condition).
    """
    if poisoned:
       multiplier = model.x_poison_cat
    else:
       multiplier = model.x_data_poison_cat

    #Component involving the sum of training samples errors
    train_samples_component = gp.quicksum(
                                          gp.quicksum(model.weights_num[r] * model.x_train_num[i, r] 
                                          for r in model.numfeatures_set)
                                          * model.x_train_cat[i,l,h] \
                                          + gp.quicksum(gp.quicksum(model.weights_cat[j, z] * model.x_train_cat[i, j, z] 
                                                      for z in range(1, model.no_categories[j] + 1)) 
                                            for j in model.catfeatures_set) \
                                          * model.x_train_cat[i,l,h] \
                                          + model.bias * model.x_train_cat[i,l,h] \
                                          - model.y_train[i] * model.x_train_cat[i,l,h]  \
                              for i in model.samples_set)

    #Component involving the sum of poison samples errors
    poison_samples_component = gp.quicksum(
                                          gp.quicksum(model.tnc_ln_times_catsamples[k,r,l,h]
                                          for r in model.numfeatures_set)
                                          + gp.quicksum(gp.quicksum(model.tcc_lc_times_catsample[k,j,z,l,h]
                                                        for z in range(1, model.no_categories[j] + 1)) 
                                            for j in model.catfeatures_set) \
                                          + model.bias * multiplier[k,l,h] \
                                          - model.y_poison[k] * multiplier[k,l,h]  \
                              for k in model.psamples_set)

    #Component involving regularization
    regularization_component = 2 * model.regularization * model.weights_cat[l,h] 

    if function == 'MSE':
      derivative_num_weights = (2 / (model.no_samples + model.no_psamples)) \
                                * (train_samples_component + poison_samples_component) \
                                + regularization_component

    elif function == 'SLS':
      derivative_num_weights = 2 * (train_samples_component + poison_samples_component) \
                                + regularization_component
    
    return  derivative_num_weights 

def loss_function_derivative_bias(model, function):
    """
    Finds the derivetive of the loss function (follower's objective) with respect to 
    the bias of the linear regression model (to get first order optimality condition).
    """

    train_samples_component = gp.quicksum(
                                          gp.quicksum(model.weights_num[r] * model.x_train_num[i, r] 
                                          for r in model.numfeatures_set)
                                          + gp.quicksum(gp.quicksum(model.weights_cat[j, z] * model.x_train_cat[i, j, z] 
                                                      for z in range(1, model.no_categories[j] + 1)) 
                                            for j in model.catfeatures_set) \
                                          + model.bias \
                                          - model.y_train[i]  \
                              for i in model.samples_set)

    poison_samples_component = gp.quicksum(
                                          gp.quicksum(model.ln_numweight_times_numsample[k,r]
                                          for r in model.numfeatures_set)
                                          + gp.quicksum(gp.quicksum(model.lc_catweight_times_catsample[k,j,z]
                                                        for z in range(1, model.no_categories[j] + 1)) 
                                            for j in model.catfeatures_set) \
                                          + model.bias \
                                          - model.y_poison[k] \
                              for k in model.psamples_set)
    
    if function == 'MSE':
        derivative_bias =  (2 / (model.no_samples + model.no_psamples)) \
                        * (train_samples_component + poison_samples_component)
    elif function == 'SLS':
        derivative_bias = 2  * (train_samples_component + poison_samples_component)

    return derivative_bias  

def ridge_objective_function(model, function):
    """
    Objective to solve standard ridge regression. Can use either MSE or SLS.
    """

    #Get sum of squared error of regression prediction and target
    sum_square_errors = gp.quicksum((gp.quicksum(model.weights[r] * model.x_train[k, r] for r in model.features_set) \
                        + model.bias \
                        - model.y_train[k]) ** 2 \
                        for k in model.samples_set)

    regularization_component = model.regularization * 1 / 2 * gp.quicksum(model.weights[r] ** 2 for r in model.features_set)

    if function == 'MSE':
      obj = ((1 / model.no_samples) * sum_square_errors) + regularization_component

    elif function == 'SLS':
      obj = sum_square_errors + regularization_component

    return obj