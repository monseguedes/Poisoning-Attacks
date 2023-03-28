"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper
"""

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import Lasso

#TODO: make depend on dataframe instead of whole data

def create_dataframe(name: str):
    """
    Create dataframe with dataset
    """

    file = 'data-binary.csv'
    path = os.path.join('data', name, file) 

    dataframe = pd.read_csv(path, index_col=0)

    return dataframe

class DataSet():
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def get_features_lists(self,no_numerical: int, no_categorical: int):
        """
        Run all necessary functions to get list of most 
        important features
        """
        self.format_data()
        self.fit_lasso()
        self.get_used_features()
        self.chosen_numerical, self.chosen_categorical = self.get_num_cat_features(no_numerical=no_numerical, no_categorical=no_categorical)

        return self.chosen_numerical, self.chosen_categorical

    def format_data(self):
        """
        Convert data to format that sklearn can handle
        """
        # Get feature columns and convert them to array
        self.features_dataframe = self.dataframe.drop(columns=['target'])
        self.features_array =  np.array(self.features_dataframe)
        self.name_features = self.features_dataframe

        # Get target column 
        self.target_array = np.array(self.dataframe.target)
        
        return self.features_array, self.target_array

    def fit_lasso(self, alpha=0.003):
        """
        Fit LASSO model to data.
        """
        self.model = Lasso(alpha=alpha)
        self.model.fit(self.features_array, self.target_array)

        return self.model

    def get_used_features(self, alpha=0.001):
        """
        Get the names of the features that LASSO assigns weights
        different than 0.
        """
        self.fit_lasso(alpha=alpha)
        coeffs = self.model.coef_  
        bool_coeff = [False if coef == 0 else True for coef in coeffs]     
        self.used_features = [name for name, nonzero in zip(self.name_features, bool_coeff) if nonzero]
        self.coeffs_used_features = {name : coeff for name in self.used_features for coeff in [coeff for coeff in coeffs if coeff != 0]}

        return self.coeffs_used_features

    def get_num_cat_features(self, no_numerical: int, no_categorical: int):
        """
        Takes the n most important features for numerical and categorical.
        For categorical, the whole categorical feature is chosen. 
        """

        ### Get numerical features------------------------
        numerical_features = {key : value for key, value in self.coeffs_used_features.items() if ':' not in key}
        # Make sure LASSO selects enough features
        alpha = 0.004
        while no_numerical > len(numerical_features):
            self.get_used_features(alpha)
            alpha += -0.001
        chosen_numerical = sorted(numerical_features, key=numerical_features.get, reverse=True)[:no_numerical]
        self.chosen_numerical = list(chosen_numerical)

        ### Get categorical features-----------------------
        categorical_features = {key : value for key, value in self.coeffs_used_features.items() if ':' in key}
        max_dict = {}
        for key, value in categorical_features.items():
            key_type = key.split(':')[0]
            if key_type not in max_dict or value > max_dict[key_type]:
                max_dict[key_type] = value
        # Make sure LASSO selects enough features
        alpha = 0.004
        while no_categorical > len(max_dict):
            self.get_used_features(alpha)
            alpha += -0.001
        chosen_categorical = sorted(max_dict, key=max_dict.get, reverse=True)[:no_categorical]
        self.chosen_categorical = list(chosen_categorical)

        return self.chosen_numerical, self.chosen_categorical

dataframe = create_dataframe('house')
test = DataSet(dataframe)
print(test.get_features_lists(5,5))

