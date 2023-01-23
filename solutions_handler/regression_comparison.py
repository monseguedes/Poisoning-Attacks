"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

TODO:fill
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import pandas as pd

import model.instance_class
import model.model_class

class ComparisonModel():
    """
    This class builds all necessary objects to compare models.
    """

    def __init__(self, bilevel_instance_data: model.instance_class.InstanceData, 
                       bilevel_model: model.model_class.PoisonAttackModel,
                       ridge_instance_data: model.instance_class.InstanceData,
                       ridge_model: model.model_class.RegressionModel):  
        """
        Given some data class and some model class, build a model to
        then compare to other model.
        bilevel_instance_data: the data class for the bilevel model
        bilevel_model: the bilevel model class
        ridge_instance_data: the data class for the ridge nopoisoned model
        ridge_model: the nonpoisoned ridge model classs
        """

        # Store data and model classes.
        self.bilevel_instance_data = bilevel_instance_data
        self.bilevel_model = bilevel_model
        self.ridge_instance_data = ridge_instance_data
        self.ridge_model = ridge_model

        self.y_train = list(self.bilevel_model.y_train.values())
        self.pred_bilevel_y_train = None
        self.pred_ridge_y_train = None
        self.data_dataframe = self.bilevel_instance_data.x_train_dataframe.copy(deep=True)
        self.ridge_data_dataframe = self.ridge_instance_data.ridge_x_train_dataframe.copy(deep=True).unstack()

    def make_poisoned_predictions(self):
        """
        Take the regression coefficents given by solving the bilevel model
        and use them to make predictions on training dataset.
        """
        
        # Define vector of size rows of data_dataframe, and with bias in all terms
        self.pred_bilevel_y_train = np.repeat(self.bilevel_model.bias.X, len(self.data_dataframe))

        # Take columns one by one, convert them to vector, and multiply them by the corresponding weights
        for column in self.data_dataframe.columns:
            if ':' not in column:
                column_index = int(column)
                self.pred_bilevel_y_train += self.data_dataframe[column] * self.bilevel_model.weights_num[column_index].X
            else:
                column_index = [int(index) for index in column.split(':')]
                self.pred_bilevel_y_train += self.data_dataframe[column] * self.bilevel_model.weights_cat[(column_index[0], column_index[1])].X
        
        self.bilevel_dataframe = self.data_dataframe.copy(deep=True)
        self.bilevel_dataframe['actual_y_train'] = self.y_train 
        self.bilevel_dataframe['pred_bilevel_y_train'] = self.pred_bilevel_y_train

        return self.pred_bilevel_y_train

    def make_non_poisoned_predictions(self):
        """
        Take the regression coefficents given by solving the nonpoisoned model
        and use them to make predictions on training dataset.
        """
        
        # Define vector of size rows of data_dataframe, and with bias in all terms
        self.pred_ridge_y_train = np.repeat(self.ridge_model.bias.X, len(self.ridge_data_dataframe))
        
        # Take columns one by one, convert them to vector, and multiply them by the corresponding weights
        for column in self.ridge_data_dataframe.columns:
            column_index = int(column)
            self.pred_ridge_y_train += self.ridge_data_dataframe[column] * self.ridge_model.weights[column_index].X
        
        self.ridge_dataframe = self.ridge_data_dataframe.copy(deep=True)
        self.ridge_dataframe['actual_y_train'] = self.y_train 
        self.ridge_dataframe['pred_ridge_y_train'] = self.pred_ridge_y_train

        return self.pred_ridge_y_train

    def plot_actual_vs_pred(self):
        """
        Take the predictions of both models
        and plot them vs actual.
        """

        # Plot bilevel model
        figure = sns.scatterplot(data=self.bilevel_dataframe, x='actual_y_train', y='pred_bilevel_y_train', label='Poisoned')
        sns.scatterplot(data=self.ridge_dataframe, x='actual_y_train', y='pred_ridge_y_train', label='Non-poisoned')
        figure.set_aspect('equal', adjustable='box')
        max_value = max([max(self.bilevel_dataframe['actual_y_train']), 
                         max(self.bilevel_dataframe['pred_bilevel_y_train']), 
                         max(self.ridge_dataframe['pred_ridge_y_train'])])
        plt.xlim([-0.05, max_value + 0.05])
        plt.ylim([-0.05,max_value + 0.05])
        plt.title('Actual vs Predicted for Training Data')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.savefig('actual_vs_predicted.png')
        plt.show()

    def store_comparison_metrics(self):
        """
        Finds and stores main regression metrics for the poisoned
        and nonpoisoned models.
        """

        # Create dataframe with metrics
        self.metrics_dataframe = pd.DataFrame({'metric': ['MSE', 'RMSE', 'MAE'],
                                               'nonpoisoned': [mean_squared_error(self.y_train, self.pred_ridge_y_train, squared=False), 
                                                               mean_squared_error(self.y_train, self.pred_ridge_y_train, squared=True),
                                                               mean_absolute_error(self.y_train, self.pred_ridge_y_train)],
                                                'poisoned': [mean_squared_error(self.y_train, self.pred_bilevel_y_train, squared=False), 
                                                             mean_squared_error(self.y_train, self.pred_bilevel_y_train, squared=True),
                                                             mean_absolute_error(self.y_train, self.pred_bilevel_y_train)]})

        # Last column as increment between both models
        self.metrics_dataframe['increase'] = (self.metrics_dataframe['poisoned'] - self.metrics_dataframe['nonpoisoned']) / self.metrics_dataframe['nonpoisoned'] * 100

        print(self.metrics_dataframe)


        








