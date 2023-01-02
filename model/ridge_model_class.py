"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

This is the model building file for the ridge regression model. It created a model class, and inside that class 
all gurobipy objects, as well as a model class, are stored. It imports all functions from another scripts,
data is given to the model as an input class.
"""

# Python Libraries
from os import path
import gurobipy as gp
from gurobipy import GRB
import itertools

# Self-created modules
import model.auxiliary_functions as aux
import model.instance_class
import algorithm.bounding_procedure as bnd


class RegressionModel():
    """
    This is the class of a unconstrained ridge regression model, 
    which has all parameters, variables, and objective.
    """ 

    def __init__(self, m: gp.Model,  instance_data: model.instance_class.InstanceData, function='MSE', **kwds,):  # Use initialisation parameters from AbtractModel class
        """
        m: a gurobipy empty model.
        instance_data: a class object with all data.
        function: type of objective function to be used.
        """

        super().__init__(**kwds)  # Gives access to methods in a superclass from the subclass that inherits from it
        self.model = m
        self.function = function
        self.build_parameters(instance_data)
        self.build_variables(instance_data)
        self.build_constraints()
        self.build_objective()
        self.model.update()
        
    def __repr__(self) -> str:
        return super().__repr__()

    def build_parameters(self, instance_data: model.instance_class.InstanceData):
        """
        Parameters of the single level model: 
        - number of training samples.
        - number of features.
        - sets for each of the above numbers.
        - data for features.
        - response variable of training data.
        - regularization parameter.
        """

        print('Defining parameters')
        
        # Order of sets
        self.no_samples = instance_data.no_samples  #No. of non-poisoned samples
        self.no_features = instance_data.no_features  # No. of numerical features
        
        print('No. training samples is:', self.no_samples)
        
        # Sets
        self.samples_set = range(1, self.no_samples + 1)   # Set of non-poisoned samples 
        self.features_set = range(1, self.no_features + 1)   # Set of numerical features

        # Parameters
        self.x_train = instance_data.ridge_x_train_dataframe.to_dict()
        self.y_train = instance_data.y_train_dataframe.to_dict()['y_train']
        self.regularization = instance_data.regularization

        print('Parameters have been defined')

    def build_variables(self, instance_data: model.instance_class.InstanceData):
        """
        Decision variables of single level model: 
        - weights.
        - bias term.
        """

        print('Creating variables')
        
        self.weights = self.model.addVars(self.features_set, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY ,name='weights')
        self.bias = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='bias')

        print('Variables have been created')

    def build_constraints(self):
        """
        There are no constraints.
        """

        print('There are no constraints')

    def build_objective(self):
        """
        Objective function of ridge regression. Maximize the mean squared error or the 
        sum of least squares.
        """

        self.model.setObjective(aux.ridge_objective_function(self, self.function), GRB.MINIMIZE)

        print('Objective has been built')


