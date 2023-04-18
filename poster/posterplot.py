"""
Functions for plots for ICERM poster.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import os

from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Add path to be able to use module from other folder
import sys
sys.path.append('./model')                
import instance_class as data
# import model.instance_class as data

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
    plt.scatter(X_train, y_train, color='lightskyblue', label='Original Data')
    plt.plot(X_train, y_pred, color='mediumblue', label='Original Regression Line')

    # Add poisoning samples
   
    X_poisoning = np.random.uniform(0, 0.4, np.ceil(0.2 * len(y_train)).astype(int)).reshape(-1, 1)
    y_poisoning = np.random.uniform(0.8, 1, np.ceil(0.2 * len(y_train)).astype(int))

    # Fit model again
    full_X_train = np.concatenate((X_train,X_poisoning), axis=0)
    full_y_train = np.concatenate((y_train, y_poisoning))
    poireg = LinearRegression().fit(full_X_train, full_y_train)

    # Add regression line and poisoned points
    full_y_pred = poireg.predict(full_X_train)
    plt.scatter(X_poisoning, y_poisoning, color='darkorange', label='Additional Data')
    plt.plot(full_X_train, full_y_pred, color='darkorange', label='Regression Line with Additional Data')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=1, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Effect of Additional Data', fontsize=30)

    plt.savefig('poster_plots/plots/bounding_coeff.pdf', transparent=True, bbox_inches = "tight")
    plt.show()

def plot_3D_regression():
    """
    Plot example of 3D linear regression.
    """

    X_train = np.random.rand(2000).reshape(1000,2)*60
    y_train = (X_train[:, 0]**2)+(X_train[:, 1]**2)
    X_test = np.random.rand(200).reshape(100,2)*60
    y_test = (X_test[:, 0]**2)+(X_test[:, 1]**2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:,0], X_train[:,1], y_train, marker='.', color='darkviolet')
    ax.set_xlabel("feature 1", fontsize=16)
    ax.set_ylabel("feature 2", fontsize=16)
    ax.set_zlabel("target", fontsize=16)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("MAE: {}".format(np.abs(y_test-y_pred).mean()))
    print("RMSE: {}".format(np.sqrt(((y_test-y_pred)**2).mean())))

    coefs = model.coef_
    intercept = model.intercept_
    xs = np.tile(np.arange(61), (61,1))
    ys = np.tile(np.arange(61), (61,1)).T
    zs = xs*coefs[0]+ys*coefs[1]+intercept
    print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(intercept, coefs[0],
                                                            coefs[1]))
    ax.plot_surface(xs,ys,zs, alpha=0.5, color='darkorange')
    plt.title('3D Regression Model', fontsize=24)

    plt.savefig('poster_plots/plots/3D_regression_example.pdf', transparent=True, bbox_inches = "tight")
    plt.show() 
    
plot_regression_lines('house', 6)
plot_3D_regression()