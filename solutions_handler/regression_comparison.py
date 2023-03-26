"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Functions necessary to evaluate the performance of poisoning attacks. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pyomo.environ as pyo

import pandas as pd

import model.instance_class
import model.model_class

sns.set_style("whitegrid")

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


class ComparisonModel():
    """
    This class builds all necessary objects to compare models.
    """

    def __init__(self, bilevel_instance_data: model.instance_class.InstanceData, 
                       bilevel_model: model.model_class.PoisonAttackModel,
                       ridge_instance_data: model.instance_class.InstanceData,
                       ridge_model: model.model_class.RegressionModel,
                       datatype: str,
                       folder):  
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

        self.datatype = datatype
        self.folder = folder

        if self.datatype == 'train':
            self.y = list(self.bilevel_model.y_train.values())
            self.data_dataframe = self.bilevel_instance_data.x_train_dataframe.copy(deep=True)
            self.ridge_data_dataframe = self.ridge_instance_data.ridge_x_train_dataframe.copy(deep=True).unstack()
        
        elif self.datatype == 'test':
            self.y = list(self.bilevel_instance_data.test_y)
            self.data_dataframe = self.bilevel_instance_data.test_dataframe.copy(deep=True)
            self.ridge_data_dataframe = self.ridge_instance_data.test_ridge_x_train_dataframe.copy(deep=True).unstack()

        self.pred_bilevel_y_train = None
        self.pred_ridge_y_train = None

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
        self.bilevel_dataframe['actual_y_train'] = self.y 
        self.bilevel_dataframe['pred_bilevel_y_train'] = self.pred_bilevel_y_train

        return self.pred_bilevel_y_train

    def make_benchmark_predictions(self, benchmark_model, benchmark_intance):   
        """
        Take the regression coefficents given by solving the model
        and use them to make predictions.
        """

        self.benchmark_intance = benchmark_intance
        self.benchmark_model = benchmark_model
        
        # Define vector of size rows of data_dataframe, and with bias in all terms.
        self.pred_benchmark_y_train = np.repeat(pyo.value(self.benchmark_model.bias), len(self.data_dataframe))

        # Take columns one by one, convert them to vector, and multiply them by the corresponding weights
        for column in self.data_dataframe.columns:
            if ':' not in column:
                column_index = int(column)
                self.pred_benchmark_y_train += self.data_dataframe[column] * self.benchmark_model.weights_num[column_index]._value
            else:
                column_index = [int(index) for index in column.split(':')]
                self.pred_benchmark_y_train += self.data_dataframe[column] * self.benchmark_model.weights_cat[(column_index[0], column_index[1])]._value
        
        self.benchmark_dataframe = self.data_dataframe
        self.benchmark_dataframe['actual_y_train'] = self.y 
        self.benchmark_dataframe['pred_benchmark_y_train'] = self.pred_benchmark_y_train

        return self.pred_benchmark_y_train

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
        self.ridge_dataframe['actual_y_train'] = self.y 
        self.ridge_dataframe['pred_ridge_y_train'] = self.pred_ridge_y_train

        return self.pred_ridge_y_train

    def plot_actual_vs_pred(self):
        """
        Take the predictions of both models
        and plot them vs actual.
        """

        # Plot bilevel model
        figure = sns.scatterplot(data=self.bilevel_dataframe, x='actual_y_train', y='pred_bilevel_y_train', label='Poisoned', color='red')
        sns.scatterplot(data=self.ridge_dataframe, x='actual_y_train', y='pred_ridge_y_train', label='Non-poisoned', color='lightskyblue')
        figure.set_aspect('equal', adjustable='box')
        max_value = max([max(self.bilevel_dataframe['actual_y_train']), 
                         max(self.bilevel_dataframe['pred_bilevel_y_train']), 
                         max(self.ridge_dataframe['pred_ridge_y_train'])])
        plt.xlim([-0.05, max_value + 0.05])
        plt.ylim([-0.05,max_value + 0.05])
        # plt.xlim([-0.05, 0.6])
        # plt.ylim([-0.05,0.6])
        if self.datatype  == 'train':
            plt.title('Actual vs Predicted for Training Data', fontsize=20)
        elif self.datatype == 'test':
            plt.title('Actual vs Predicted for Test Data', fontsize=20)
        plt.xlabel('Actual', fontsize=14)
        plt.ylabel('Predicted', fontsize=14)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=14)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.1), fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig('plots/' + self.folder +
                     '/actual_vs_predicted' + '_' +
                     self.datatype + '_' +
                     str(self.bilevel_model.no_numfeatures) + '_' + 
                     str(self.bilevel_model.no_catfeatures) + '_' +
                     str(len(self.bilevel_dataframe)) + '_' +
                     str(int(self.bilevel_instance_data.poison_rate * 100)) + '_' +
                     str(self.bilevel_instance_data.seed) +
                     '.pdf', transparent=True, bbox_inches = "tight")
        plt.show()

    def plot_actual_vs_pred_benchmark(self):
        """
        Take the predictions of both models
        and plot them vs actual.
        """

        figure = sns.scatterplot(data=self.benchmark_dataframe, x='actual_y_train', y='pred_benchmark_y_train', label='Poisoned')
        sns.scatterplot(data=self.ridge_dataframe, x='actual_y_train', y='pred_ridge_y_train', label='Non-poisoned')
        figure.set_aspect('equal', adjustable='box')
        max_value = max([max(self.benchmark_dataframe['actual_y_train']), 
                         max(self.benchmark_dataframe['pred_benchmark_y_train']), 
                         max(self.ridge_dataframe['pred_ridge_y_train'])])
        plt.xlim([-0.05, max_value + 0.05])
        plt.ylim([-0.05,max_value + 0.05])
        plt.title('Actual vs Predicted for Training Data')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.savefig('plots/' + self.folder +
                     '/benchmark_actual_vs_predicted' + '_' +
                     self.datatype + '_' +
                     str(self.benchmark_model.no_numfeatures) + '_' + 
                     str(self.benchmark_model.no_catfeatures) + '_' +
                     str(len(self.benchmark_dataframe)) + '_' +
                     str(int(self.benchmark_intance.poison_rate * 100)) + '_' +
                     str(self.benchmark_intance.seed) + 
                     '.png')
        plt.show()

    def plot_actual_vs_predicted_all(self):
        """
        Take the predictions of both models
        and plot them vs actual.
        """

        figure = sns.scatterplot(data=self.bilevel_dataframe, x='actual_y_train', y='pred_bilevel_y_train', label='Poisoned')
        sns.scatterplot(data=self.ridge_dataframe, x='actual_y_train', y='pred_ridge_y_train', label='Non-poisoned')
        sns.scatterplot(data=self.benchmark_dataframe, x='actual_y_train', y='pred_benchmark_y_train', label='Benchmark')
        
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
        plt.savefig('plots/' + self.folder +
                     '/all_actual_vs_predicted' + '_' +
                     self.datatype + '_' +
                     str(self.bilevel_model.no_numfeatures) + '_' + 
                     str(self.bilevel_model.no_catfeatures) + '_' +
                     str(len(self.bilevel_dataframe)) + '_' +
                     str(int(self.bilevel_instance_data.poison_rate * 100)) + '_' + 
                     str(self.bilevel_instance_data.seed) +  
                     '.png')
        plt.show()

    def store_comparison_metrics(self):
        """
        Finds and stores main regression metrics for the poisoned
        and nonpoisoned models.
        """

        # Create dataframe with metrics
        self.metrics_dataframe = pd.DataFrame({'metric': ['MSE', 'RMSE', 'MAE'],
                                               'nonpoisoned': [mean_squared_error(self.y, self.pred_ridge_y_train, squared=False), 
                                                               mean_squared_error(self.y, self.pred_ridge_y_train, squared=True),
                                                               mean_absolute_error(self.y, self.pred_ridge_y_train)],
                                                'benchmark': [mean_squared_error(self.y, self.pred_benchmark_y_train, squared=False), 
                                                              mean_squared_error(self.y, self.pred_benchmark_y_train, squared=True),
                                                              mean_absolute_error(self.y, self.pred_benchmark_y_train)],
                                                'poisoned': [mean_squared_error(self.y, self.pred_bilevel_y_train, squared=False), 
                                                             mean_squared_error(self.y, self.pred_bilevel_y_train, squared=True),
                                                             mean_absolute_error(self.y, self.pred_bilevel_y_train)]})

        # Last columns as increment between models
        self.metrics_dataframe['non-benchmark increase'] = (self.metrics_dataframe['benchmark'] - self.metrics_dataframe['nonpoisoned']) / self.metrics_dataframe['nonpoisoned'] * 100
        self.metrics_dataframe['non-MINLP increase'] = (self.metrics_dataframe['poisoned'] - self.metrics_dataframe['nonpoisoned']) / self.metrics_dataframe['nonpoisoned'] * 100
        self.metrics_dataframe['benchmark-MINLP increase'] = (self.metrics_dataframe['poisoned'] - self.metrics_dataframe['benchmark']) / self.metrics_dataframe['benchmark'] * 100

        self.metrics_dataframe.to_csv('solutions/' + self.folder +  
                                      '/all_actual_vs_predicted' + '_' +
                                      self.datatype + '_' +
                                      str(self.bilevel_model.no_numfeatures) + '_' + 
                                      str(self.bilevel_model.no_catfeatures) + '_' +
                                      str(len(self.bilevel_dataframe)) + '_' +
                                      str(int(self.bilevel_instance_data.poison_rate * 100)) + 
                                      str(self.bilevel_instance_data.seed) +
                                      '.csv')
        
        print(self.metrics_dataframe)


        








