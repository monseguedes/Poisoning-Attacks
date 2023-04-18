"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

TODO:fill
"""

# Self-created imports 
import model.instance_class as data

# Python libraries
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

# Set parameters
dataset_name = 'pharm'
possible_paramaters = np.linspace(0.6,0.7)

# Create data
instance_data = data.InstanceData(dataset_name=dataset_name, seed=1)
instance_data.create_dataframes()

# Convert data into sklearn format (numpy)
target = instance_data.train_dataframe.pop('target').values # Numpy array with target values
feature_matrix = instance_data.train_dataframe.to_numpy()   # Numpy 2D array with feature matrix

# Grid search with CV 
grid_space = {'alpha': possible_paramaters}
ridge = Ridge()
grid_search = GridSearchCV(ridge, grid_space, scoring='neg_mean_squared_error', cv=10)
grid_search.fit(feature_matrix, target)

# Print results
print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n", grid_search.best_estimator_)
print("\n The best score across ALL searched params:\n", grid_search.best_score_)
print("\n The best parameters across ALL searched params:\n", grid_search.best_params_)


