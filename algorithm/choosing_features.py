"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper
"""

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import Lasso

def create_dataframe(name: str):
    """
    Create dataframe with dataset
    """

    file = 'data-binary.csv'
    path = os.path.join('data', name, file) 

    dataframe = pd.read_csv(path, index_col=0)

    return dataframe

class LASSOdataframe():
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def get_features_lists(self,no_numerical: int, no_categorical: int):
        """
        Run all necessary functions to get list of most 
        important features
        """
        alpha = 0.005
        self.format_data()
        self.fit_lasso()
        self.get_used_features(alpha=alpha)
        self.chosen_numerical, self.chosen_categorical = self.get_num_cat_features(no_numerical=no_numerical, no_categorical=no_categorical, alpha=alpha)

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

    def get_num_cat_features(self, no_numerical: int, no_categorical: int, alpha: float):
        """
        Takes the n most important features for numerical and categorical.
        For categorical, the whole categorical feature is chosen. 
        """

        ### Get numerical features------------------------
        numerical_features = {int(key) : abs(value) for key, value in self.coeffs_used_features.items() if ':' not in key}
        if no_numerical == 'all':
            self.chosen_numerical = [column for column in self.features_dataframe.columns if ':' not in column] 
        else:
            # Make sure LASSO selects enough features
            while no_numerical > len(numerical_features):
                self.get_used_features(alpha)
                alpha -= 0.00001
                numerical_features = {int(key) : abs(value) for key, value in self.coeffs_used_features.items() if ':' not in key}
            chosen_numerical = sorted(numerical_features, key=numerical_features.get, reverse=True)[:no_numerical]
            self.chosen_numerical = list(chosen_numerical)

        ### Get categorical features-----------------------
        categorical_features = {key : abs(value) for key, value in self.coeffs_used_features.items() if ':' in key}
        if no_categorical == 'all':
            self.chosen_categorical = list(set([column.split(':')[0] for column in self.features_dataframe.columns if ':' in column]))
        else:
            max_dict = {}
            for key, value in categorical_features.items():
                key_type = key.split(':')[0]
                if key_type not in max_dict or value > max_dict[key_type]:
                    max_dict[int(key_type)] = value
            # Make sure LASSO selects enough features
            while no_categorical > len(max_dict):
                self.get_used_features(alpha)
                alpha -= 0.0001
            chosen_categorical = sorted(max_dict, key=max_dict.get, reverse=True)[:no_categorical]
            self.chosen_categorical = list(chosen_categorical)

        return self.chosen_numerical, self.chosen_categorical


