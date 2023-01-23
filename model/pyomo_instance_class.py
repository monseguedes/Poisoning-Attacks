"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Class for data input. TODO: fill
"""

# Python imports
import pandas as pd
import numpy as np
from os import path
from math import floor

class InstanceData():
    def __init__(self, dataset_name: str, seed: int):    # Right now names are either 'pharm', or 'house'
        """
        The initialization corresponds to the data for the first iteration. If there are no iterations (single attack strategy), 
        it still works with N=1.

        dataset_name: 'pharm', or 'house'
        poisson_rate: 4, 8, 12, 16, 20
        N: number of poison subsets (if total number of poison samples is not divisible by N, then some samples are ignored)
        seed: seed for different random splits of training, validation and testing sets
        """

        self.iteration_count = 1    # To keep track of iteration count
        self.dataset_directory = ''.join(['data/', dataset_name])    # e.g., data/pharm
        self.seed = seed
    
    def prepare_instance(self, poison_rate: int, N: int):
        """
        TODO: fill
        """
    
        # Poisoning parameters
        self.poison_rate = poison_rate / 100    # 4, 8, 12, 16, or 20
        self.no_poisson_subsets = N    # no. of subsets in which the total poisson samples (gotten after applying rate to training data) is divided

        self.create_dataframes()
        self.split_dataframe()
        self.poison_samples()
        self.inital_sets_size()
        self.regularization_parameter()

    def create_dataframes(self):
        """
        TODO: fill
        """
        
        # Main dataframes
        self.whole_dataframe = pd.read_csv(path.join(self.dataset_directory, 'data-binary.csv'), index_col=[0]) # Whole dataframe with features as columns and target column, as in file.
        
        no_samples = len(self.whole_dataframe)

        if no_samples > 300:
            # Pick fixed number of trainign samples.
            self.train_dataframe = self.whole_dataframe.sample(frac=None, 
                                                               n=12, 
                                                               random_state=self.seed) # The indexes are not reset, but randomly shuffled 
        else:
            # Randomly pick half of the samples for training data.
            self.train_dataframe = self.whole_dataframe.sample(frac=1/2, 
                                                           random_state=self.seed) # The indexes are not reset, but randomly shuffled 
        
        self.test_validation_dataframe = self.whole_dataframe.drop(self.train_dataframe.index)  # Store rest of samples, which will be further divided into testing and validating sets (1:1 ratio)

    def split_dataframe(self):
        """
        Split training dataframe into features dataframe and target dataframe.
        """

        # FEATURES (x_train)
        # Get only feature columns and reset index
        self.x_train_dataframe = self.train_dataframe.drop(columns=['target'], 
                                                           inplace=False).reset_index(drop=True)    
        self.x_train_dataframe.index.name = 'sample'  
        self.x_train_dataframe.index += 1   # Index starts at 1
        self.original_x_train_dataframe = self.x_train_dataframe
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.processed_x_train_dataframe =  self.x_train_dataframe.rename(columns={x:y for x,y in zip(self.x_train_dataframe.columns,range(1,len(self.x_train_dataframe.columns) + 1))})
        self.processed_x_train_dataframe = self.processed_x_train_dataframe[self.processed_x_train_dataframe.columns].stack().rename_axis(index={None: 'feature'})    
        self.processed_x_train_dataframe.name = 'x_train'

        # TARGET (y_train)
        self.y_train_dataframe = self.train_dataframe[['target']].reset_index(drop=True)    # Get only target column
        self.y_train_dataframe.rename(columns={'target': 'y_train'}, inplace=True)  # Rename column as y_train, which is the name of the pyomo parameter
        self.y_train_dataframe.index.name = 'sample'    # Rename new index as 'sample'   
        self.y_train_dataframe.index += 1   # Make index start at 1 instead of 0 
        self.original_y_train_dataframe = self.y_train_dataframe

    def poison_samples(self):
        # Select poison samples (from training data)
        self.poison_dataframe = self.train_dataframe.sample(frac= self.poison_rate, random_state=self.seed).reset_index(drop=True)   # Dataframe with all samples ot be poisoned
        self.no_total_psamples = self.poison_dataframe.index.size    # Total number of poisoned samples (rate applied to training data)
        self.no_psamples_per_subset = floor(self.no_total_psamples / self.no_poisson_subsets)   # Gets the biggest number of samples per subset that makes possible the desired number of subsets
        self.no_total_psamples = self.no_psamples_per_subset * self.no_poisson_subsets  # Now multiplies the no. samples per subset and no. of subset to get total poisoned samples
        self.poison_dataframe = self.poison_dataframe.iloc[:self.no_total_psamples]   # If the initial poison data had a non divisible number of samples, update it to be divisible

        # X_POISON_CAT 
        self.complete_cat_poison_dataframe = self.poison_dataframe.drop(columns=['target'], 
                                                               inplace=False).reset_index(drop=True) 
        self.cat_columns = [name for name in self.poison_dataframe.columns if ':' in name]
        self.complete_cat_poison_dataframe = self.poison_dataframe[self.cat_columns]
        self.complete_cat_poison_dataframe.index.name = 'sample'  
        self.complete_cat_poison_dataframe.index += 1   # Index starts at 1
        # Define poison data (x_poison_cat) for initial iteration
        self.cat_poison_dataframe = self.complete_cat_poison_dataframe.iloc[:self.no_psamples_per_subset].reset_index(drop=True)
        self.cat_poison_dataframe.index.name = 'sample' 
        self.cat_poison_dataframe.index += 1
        
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.cat_poison_dataframe =  self.cat_poison_dataframe.rename(columns={x:y for x,y in zip(self.cat_poison_dataframe.columns,range(1,len(self.cat_poison_dataframe.columns) + 1))})
        self.cat_poison_dataframe = self.cat_poison_dataframe[self.cat_poison_dataframe.columns].stack().rename_axis(index={None: 'feature'})    
        self.cat_poison_dataframe.name = 'x_poison_cat'
        
        # TARGET (y_poison)
        self.complete_y_poison_dataframe = self.poison_dataframe[['target']].reset_index(drop=True) # Get only target column
        self.complete_y_poison_dataframe.rename(columns={'target': 'y_poison'}, inplace=True)  # Rename column as y_poison, which is the name of the pyomo parameter
        self.complete_y_poison_dataframe.index += 1

        # Define poison data (y_poison) for initial iteration
        self.y_poison_dataframe = self.complete_y_poison_dataframe.iloc[:self.no_psamples_per_subset].reset_index(drop=True)
        self.y_poison_dataframe.index += 1
        self.y_poison_dataframe = round(1 - self.y_poison_dataframe)
        self.attack_initialization = self.y_poison_dataframe    # Might need to change format, we'll see
        
    def inital_sets_size(self):
        """
        TODO: fill
        """
        
        # Initial size of sets
        self.no_samples = self.processed_x_train_dataframe.index.levshape[0]
        self.no_psamples = self.y_poison_dataframe.index.size
        self.no_cat_features = self.cat_poison_dataframe.index.levshape[1]
        self.no_total_features = self.processed_x_train_dataframe.index.levshape[1]
        self.no_num_features = self.no_total_features - self.no_cat_features

    def regularization_parameter(self):
        """
        TODO: fill
        """
        
        # Other parameters
        self.regularization = 0.6612244897959183

    def update_data(self, new_x_train: pd.DataFrame):
        """
        Updates instance data in order to incorporate solutions of previous iteration.

        The input is a dataframe with new data in teh following format: multiindex dataframe with sample and feature as index
        and solutions of solving model for x_poison as column. Here, x_poison becomes x_train since solutions to previous 
        iterations become datapoints.
        """

        # Increase iteration count every time this method is called
        self.iteration_count += 1

        # Add categorical features to new_x_train
        new_x_train = new_x_train.unstack(level=1)
        new_x_train.columns = [str(column) for column in new_x_train.columns]

        # Make categorical features have matching indices to concatenate with numerical features.
        categorical_features = self.cat_poison_dataframe.unstack(level=1)
        categorical_features.columns = self.cat_columns
        new_x_train = pd.concat([new_x_train, categorical_features], axis=1)
        
        ### Update parameters dataframe
        # DATA FEATURES (x_train)
        self.x_train_dataframe = pd.concat([self.x_train_dataframe, new_x_train], ignore_index=True)  # Add new x_train data (solutions) to old x_train data
        self.x_train_dataframe.index += 1

        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.processed_x_train_dataframe =  self.x_train_dataframe.rename(columns={x:y for x,y in zip(self.x_train_dataframe.columns,range(1,len(self.x_train_dataframe.columns) + 1))})
        self.processed_x_train_dataframe = self.processed_x_train_dataframe[self.processed_x_train_dataframe.columns].stack().rename_axis(index={None: 'feature'})    
        self.processed_x_train_dataframe.name = 'x_train'
        self.processed_x_train_dataframe.index.rename('sample', level=0, inplace=True)

        # DATA TARGET (y_train)
        self.y_train_dataframe = pd.concat([self.y_train_dataframe, self.y_poison_dataframe.rename(columns={'y_poison':'y_train'})]).reset_index(drop=True) # Add poison target to main target dataframe since attacks from previpus iterations become data for next iteration
        self.y_train_dataframe.index.rename('sample')
        self.y_train_dataframe.index += 1

        # ATTACK CAT FEATURES
        self.cat_poison_dataframe = self.complete_cat_poison_dataframe[(self.iteration_count - 2) * self.no_psamples_per_subset:    # Get next poison samples (by slicing whole poison samples in order)
                                                                       (self.iteration_count - 1) * self.no_psamples_per_subset].reset_index(drop=True)    # Depends on iteration        
        
        self.cat_poison_dataframe.index.rename('sample')
        self.cat_poison_dataframe.index += 1
        self.cat_poison_dataframe =  self.cat_poison_dataframe.rename(columns={x:y for x,y in zip(self.cat_poison_dataframe.columns,range(1,len(self.cat_poison_dataframe.columns) + 1))})
        self.cat_poison_dataframe = self.cat_poison_dataframe[self.cat_poison_dataframe.columns].stack().rename_axis(index={None: 'feature'})    
        self.cat_poison_dataframe.name = 'x_poison_cat'

        # ATTACK TARGET (y_poison)
        self.y_poison_dataframe = self.complete_y_poison_dataframe[(self.iteration_count - 1) * self.no_psamples_per_subset:    # Get next poison samples (by slicing whole poison samples in order)
                                                                    self.iteration_count * self.no_psamples_per_subset].reset_index(drop=True)    # Depends on iteration        
        self.y_poison_dataframe.index.rename('sample')
        self.y_poison_dataframe.index += 1
        ###

        # Update sets
        self.no_samples = self.processed_x_train_dataframe.index.levshape[0]
        self.no_psamples = self.y_poison_dataframe.index.size
        self.no_cat_features = self.cat_poison_dataframe.index.levshape[1]
        self.no_total_features = self.processed_x_train_dataframe.index.levshape[1]
        self.no_num_features = self.no_total_features - self.no_cat_features

    def format_data(self):
        # Order of sets
        self.no_samples =   #No. of non-poisoned samples
        self.no_features =   # No. of numerical features
                
        # Parameters
        self.ridge_x_train_dataframe = 
        self.y_train_dataframe = 
        self.regularization = 
           
       



        