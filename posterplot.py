"""
Functions for plots for ICERM poster.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


import model.instance_class as data

from sklearn.linear_model import LinearRegression

sns.set_style("whitegrid")

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def plot_regression_lines(dataset_name, feature):
    """
    Plot to show the motivation of lower-level bounds
    """
     # Create dataset
    instance_data = data.InstanceData(dataset_name=dataset_name)
    instance_data.prepare_instance(poison_rate=20, 
                                   training_samples=15,
                                   seed=4)
    
    # Pick feature
    feature_dataframe = instance_data.test_dataframe[str(feature)]
    target_series = instance_data.test_y

    np.random.seed(1)
    # Convert to numpy format
    X_train =  np.random.uniform(0, 1, len(feature_dataframe))
    y_train = 0.6 * X_train
    X_train = X_train.reshape(-1, 1)
    perturbation = np.random.uniform(-0.15, 0.15, len(y_train))
    y_train = y_train + perturbation

    # Fit model
    reg = LinearRegression().fit(X_train, y_train)

    # Plot regression line and data
    y_pred = reg.predict(X_train)
    plt.scatter(X_train, y_train, color='lightskyblue', label='Original Data Points')
    plt.plot(X_train, y_pred, color='navy', label='Original Regression Line')

    # Add poisoning samples
   
    X_poisoning = np.random.uniform(0, 0.4, np.ceil(0.2 * len(y_train)).astype(int)).reshape(-1, 1)
    y_poisoning = np.random.uniform(0.8, 1, np.ceil(0.2 * len(y_train)).astype(int))

    # Fit model again
    full_X_train = np.concatenate((X_train,X_poisoning), axis=0)
    full_y_train = np.concatenate((y_train, y_poisoning))
    poireg = LinearRegression().fit(full_X_train, full_y_train)

    # Add regression line and poisoned points
    full_y_pred = poireg.predict(full_X_train)
    plt.scatter(X_poisoning, y_poisoning, color='red', label='Poisoning Smaples')
    plt.plot(full_X_train, full_y_pred, color='darkred', label='Regression Line with Poisoning Smaples')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2)

    plt.title('Regression Model')

    plt.savefig('bounding_coeff.png', transparent=True, bbox_inches = "tight")

    plt.show()

plot_regression_lines('house', 6)