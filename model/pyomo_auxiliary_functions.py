"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Collection of auxiliary functions defines the objective and constraints of the model.
"""

import numpy as np
from statistics import mean

def linear_regression_function(no_sample, features_set, features, weights, bias):
    """
    Given the sample, the set of features, the features, and the regression
    parameters weights and bias, this function finds the predicted value for 
    a sample.
    LRF (prediction) = weight * sample + bias
    """

    #Predict values using linear regression
    y_hat = sum(features[no_sample, j] * weights[j] for j in features_set) + bias
    
    return y_hat

def mean_squared_error(model):
    """
    Gets mean squared error, which is the mean of sum of the square of the difference
    between predicted values (regression) and target values for all samples.
    MSE = 1 / n * summation( (predicted - target)^2 ) 
    """

    #Get sum of squared error of regression prediction and target
    sum_square_errors = sum((linear_regression_function(i, model.total_features_set , model.x_train, model.weights, model.bias) - model.y_train[i]) ** 2 
       for i in model.samples_set)

    #sum_square_errors = sum( (sum(model.x_train[i, j] * model.weights[j] for j in model.features_set) + model.bias - model.y_train[i]) ** 2 
     #   for i in model.samples_set)

    #Get mean of squared errors
    mse = 1 / model.no_samples * sum_square_errors

    return mse

def loss_function_derivative_num_weights(model, j):
    """
    Finds the derivetive of the loss function (follower's objective) with respect to 
    the weights of the linear regression model, and sets it to 0 (first order optimality
    condition).
    """

    train_samples_component = sum((linear_regression_function(i, model.total_features_set, model.x_train, model.weights, model.bias) - model.y_train[i]) * model.x_train[i,j] 
                                  for i in model.samples_set) #Component involving the sum of training samples errors
    poison_samples_component = sum((linear_regression_function(q, model.num_features_set, model.x_poison, model.weights, model.bias) + 
                                    sum(model.x_poison_cat[q, k] * model.weights[model.no_total_features - model.no_num_features + k] for k in model.cat_features_set) 
                                    - model.y_poison[q]) 
                                    * model.x_poison[q,j] 
                                    for q in model.psamples_set) #Component involving the sum of poison samples errors
    regularization_component = model.regularization * model.weights[j] #Component involving the regularization
    
    return  (2 / (model.no_samples + model.no_psamples)) * (train_samples_component + poison_samples_component) + regularization_component 

def loss_function_derivative_cat_weights(model, j):
    """
    Finds the derivetive of the loss function (follower's objective) with respect to 
    the weights of the linear regression model, and sets it to 0 (first order optimality
    condition).
    """

    train_samples_component = sum((linear_regression_function(i, model.total_features_set, model.x_train, model.weights, model.bias) - model.y_train[i]) * model.x_train[i,model.no_total_features - model.no_num_features + j] 
        for i in model.samples_set) #Component involving the sum of training samples errors
    poison_samples_component = sum((linear_regression_function(q, model.num_features_set, model.x_poison, model.weights, model.bias) + 
                               sum(model.x_poison_cat[q, k] * model.weights[model.no_total_features - model.no_num_features + k] for k in model.cat_features_set) 
                               - model.y_poison[q]) 
                               * model.x_poison_cat[q,j] 
                               for q in model.psamples_set) #Component involving the sum of poison samples errors
    regularization_component = model.regularization * model.weights[j] #Component involving the regularization
    
    return  (2 / (model.no_samples + model.no_psamples)) * (train_samples_component + poison_samples_component) + regularization_component 

def loss_function_derivative_bias(model):
    """
    Finds the derivetive of the loss function (follower's objective) with respect to 
    the bias of the linear regression model, and sets it to 0 (first order optimality
    condition).
    """

    train_samples_component = sum((linear_regression_function(i, model.total_features_set, model.x_train, model.weights, model.bias) - model.y_train[i]) 
        for i in model.samples_set)
    poison_samples_component = sum((linear_regression_function(q, model.num_features_set, model.x_poison, model.weights, model.bias) + 
                               sum(model.x_poison_cat[q, k] * model.weights[model.no_total_features - model.no_num_features + k] for k in model.cat_features_set) 
                               - model.y_poison[q]) 
                               for q in model.psamples_set)

    return  (2 / (model.no_samples + model.no_psamples)) * (train_samples_component + poison_samples_component)