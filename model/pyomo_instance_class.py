"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

This script creates the class with all the data that is then given to the benckmark model.
"""

# Python imports
import pandas as pd
import numpy as np
from os import path
from math import floor

class InstanceData():
    def __init__(self, dataset_name: str):    # Right now names are either 'pharm', or 'house'
        """
        The initialization corresponds to the data for the first iteration. If there are no iterations (single attack strategy).

        dataset_name: 'pharm', or 'house'
        """

        self.iteration_count = 1    # To keep track of iteration count
        self.dataset_directory = ''.join(['data/', dataset_name])    # e.g., data/pharm
        
    def prepare_instance(self, poison_rate: int, training_samples: int, N: int, seed: int):
        """
        Prepares the instance by creating dataframe, dividing it into poisoning samples and 
        standard samples, defining the sizes of the sets involved in the model, and the 
        regularisation parameter. This depends on the poison rate.
        poisson_rate: 4, 8, 12, 16, 20.
        training_samples: no. training samples chosen from the whole data.
        N: number of poisoning subsets.
        seed: seed for different random splits of training, validation and testing sets.
        """

        self.seed = seed

        # Poisoning parameters
        self.poison_rate = poison_rate / 100    # 4, 8, 12, 16, or 20
        self.no_poisson_subsets = N    # no. of subsets in which the total poisson samples (gotten after applying rate to training data) is divided
        
        # Run all necessary methods
        self.create_dataframes(training_samples, self.seed)
        self.split_dataframe()
        self.num_cat_split()
        self.poison_samples()
        self.inital_sets_size()
        self.regularization_parameter()

    def create_dataframes(self, training_samples: int, seed: int):
        """
        Creates a dataframe with all the data, which has features and traget as columns, 
        and samples as rows. Numerical columns are integers, while categorical columns are 
        of the form '1:1' for 'no.catfeature:no.category'. Response variable is names as
        'target'. These files are prepared by preprocessing. 
        """
        
        # Whole dataframe with features as columns and target column, as in file.
        self.whole_dataframe = pd.read_csv(path.join(self.dataset_directory, 
                                                     'data-binary.csv'), 
                                           index_col=[0]) 
        print(len(self.whole_dataframe.columns))

        # Pick fixed number of trainig samples.
        self.train_dataframe = self.whole_dataframe.sample(frac=None, 
                                                           n=training_samples, 
                                                           random_state=seed) # The indexes are not reset, but randomly shuffled 
        
        # Store rest of samples, which will be further divided into testing and validating sets
        self.test_validation_dataframe = self.whole_dataframe.drop(self.train_dataframe.index)
        self.test_dataframe =  self.test_validation_dataframe.sample(frac=None, 
                                                                     n = min(5 * training_samples,len(self.test_validation_dataframe.index)), 
                                                                     random_state=seed) # The indexes are not reset, but randomly shuffled 
        self.test_dataframe = self.test_dataframe.reset_index(drop=True)
        self.test_dataframe.index.name = 'sample'  
        self.test_dataframe.index += 1   # Index starts at 1 
        self.test_y = self.test_dataframe['target']  
        self.test_dataframe = self.test_dataframe.drop(columns=['target'], 
                                                       inplace=False)

    def split_dataframe(self):
        """
        Splits training dataframe into features dataframe and target dataframe.
        This function has two main outputs: 
        - a dataframe with response variables,
        - a dataframe with just the features which mantains the '1:1' notation for 
        the categorical features, 
        - a multiindexed dataframe with all features numbered as integers (not 
        distingushing between numerical and categorical). This last dataframe is 
        used for the ridge regression model.
        """

        ### FEATURES (x_train)------------------------
        # Get only feature columns and reset index. Columns are still 1,2,.. 1:1,...
        self.x_train_dataframe = self.train_dataframe.drop(columns=['target'], 
                                                           inplace=False).reset_index(drop=True)    
        self.x_train_dataframe.index.name = 'sample'  
        self.x_train_dataframe.index += 1   # Index starts at 1

        # Set no. of samples variables.
        self.no_samples = len(self.x_train_dataframe.index)
        self.no_total_features = len(self.x_train_dataframe.columns)

        # Change dataframe column names to create dataframe for ridge model.
        self.ridge_x_train_dataframe = self.x_train_dataframe.copy()
        self.ridge_x_train_dataframe.columns = [count + 1 for count, value in enumerate(self.x_train_dataframe.columns)]
        self.ridge_x_train_dataframe = self.ridge_x_train_dataframe.stack().rename_axis(index={None: 'feature'})  

        ### TARGET (y_train)-------------------------
        self.y_train_dataframe = self.train_dataframe[['target']].reset_index(drop=True)   
        self.y_train_dataframe.rename(columns={'target': 'y_train'}, inplace=True)  # Rename column as y_train, which is the name of the pyomo parameter
        self.y_train_dataframe.index.name = 'sample'      
        self.y_train_dataframe.index += 1   

        return self.x_train_dataframe, self.y_train_dataframe

    def num_cat_split(self):
        """
        Splits the features dataframe into one multiindexed dataframe for numerical 
        features and one multiindexed dataframe for categorical features.
        """

        ### NUMERICAL FEATURES-------------------------------
        # Get only numerical columns (those that are just integers) and convert them to integer type
        self.numerical_columns = [name for name in self.x_train_dataframe.columns if ':' not in name]
        self.num_x_train_dataframe = self.x_train_dataframe[self.numerical_columns]
        self.num_x_train_dataframe.columns = self.num_x_train_dataframe.columns.astype(int) # Make column names integers so that 
                                                                                            # they can later be used as pyomo indices
        # Stack dataframe to get multiindex, indexed by sample and feature
        self.num_x_train_dataframe = self.num_x_train_dataframe.stack().rename_axis(index={None: 'feature'})    
        self.num_x_train_dataframe.name = 'x_train_num'
        
        ### CATEGORICAL FEATURES------------------------------
        # Get only categorical columns (those that include ':' in name)
        self.categorical_columns = [name for name in self.x_train_dataframe.columns if name not in self.numerical_columns]
        self.cat_x_train_dataframe = self.x_train_dataframe[self.categorical_columns]
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.cat_x_train_dataframe = self.cat_x_train_dataframe.stack().rename_axis(index={None: 'column'})   
        self.cat_x_train_dataframe.name = 'x_train_cat'
        self.cat_x_train_dataframe = self.cat_x_train_dataframe.reset_index()   # This resets index so that current index becomes columns
        # Split multiindex of the form '1:2' into one index for 1 and another index for 2
        self.cat_x_train_dataframe[['feature', 'category']] = self.cat_x_train_dataframe.column.str.split(':', expand=True).astype(int)
        self.cat_x_train_dataframe = self.cat_x_train_dataframe.drop(columns=['column'])   # Drops the columns wirth '1:1' names 
        self.cat_x_train_dataframe = self.cat_x_train_dataframe.set_index(['sample', 'feature', 'category'])   # Sets relevant columns as indices.

        return self.num_x_train_dataframe, self.cat_x_train_dataframe

    def poison_samples(self):
        """
        Takes the dataframe for training data and gets data for poisoning samples
        depending on poisoning rate
        """

        # Dataframe with all samples to be poisoned
        self.poison_dataframe = self.train_dataframe.sample(frac= self.poison_rate, 
                                                            random_state=self.seed).reset_index(drop=True)  
        # Total number of poisoned samples (rate applied to training data)
        self.no_total_psamples = self.poison_dataframe.index.size   
        # Get the biggest number of samples per subset that makes possible the desired number of subsets
        self.no_psamples_per_subset = floor(self.no_total_psamples / self.no_poisson_subsets) 
        # Now multiplies the no. samples per subset and no. of subset to get total poisoned samples 
        self.no_total_psamples = self.no_psamples_per_subset * self.no_poisson_subsets 
        # If the initial poison data had a non divisible number of samples, update it to be divisible
        self.poison_dataframe = self.poison_dataframe.iloc[:self.no_total_psamples]   

        # X_POISON_CAT------------------------------------------
        self.complete_x_poison_dataframe = self.poison_dataframe.drop(columns=['target'], 
                                                               inplace=False).reset_index(drop=True) 
        self.cat_columns = [name for name in self.poison_dataframe.columns if ':' in name]
        self.complete_cat_poison_dataframe = self.poison_dataframe[self.cat_columns]
        self.complete_cat_poison_dataframe.index.name = 'sample'  
        self.complete_cat_poison_dataframe.index += 1  
        # Define poison data (x_poison_cat) for initial iteration
        self.cat_poison_dataframe = self.complete_cat_poison_dataframe.iloc[:self.no_psamples_per_subset].reset_index(drop=True)
        self.cat_poison_dataframe.index.name = 'sample' 
        self.cat_poison_dataframe.index += 1
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.cat_poison_dataframe = self.cat_poison_dataframe.stack().rename_axis(index={None: 'column'})   
        self.cat_poison_dataframe.name = 'x_poison_cat'
        self.cat_poison_dataframe = self.cat_poison_dataframe.reset_index()   # This resets index so that current index becomes columns
        # Split multiindex of the form '1:2' into one index for 1 and another index for 2
        self.cat_poison_dataframe[['feature', 'category']] = self.cat_poison_dataframe.column.str.split(':', expand=True).astype(int)
        self.cat_poison_dataframe = self.cat_poison_dataframe.drop(columns=['column'])   # Drops the columns wirth '1:1' names 
        self.cat_poison_dataframe = self.cat_poison_dataframe.set_index(['sample', 'feature', 'category'])   # Sets relevant columns as indices.

        ### Initial poisoning samples-------------------------
        self.num_x_poison_dataframe = self.complete_x_poison_dataframe[self.numerical_columns]
        self.num_x_poison_dataframe.index += 1
        # Initialise those to be poisoned to be opposite
        def flip_nearest(x):
            if x < 0.5:
                return 1
            else:
                return 0
        # for feature in self.numerical_columns:
        #     self.num_x_poison_dataframe[feature]= self.num_x_poison_dataframe[feature].apply(lambda x: flip_nearest(x))
        self.num_x_poison_dataframe.columns = self.num_x_poison_dataframe.columns.astype(int) # Make column names integers so that 
                                                                                                    # they can later be used as pyomo indices
        # Stack dataframe to get multiindex, indexed by sample and feature, this is nice when converted
        # to dictionary and used as data since matched gurobi's format.
        self.num_x_poison_dataframe = self.num_x_poison_dataframe.stack().rename_axis(index={None: 'feature'})    
        self.num_x_poison_dataframe.name = 'x_data_poison_num'
        
        ### TARGET (y_poison)---------------------------------
        self.complete_y_poison_dataframe = self.poison_dataframe[['target']].reset_index(drop=True) 
        self.complete_y_poison_dataframe.rename(columns={'target': 'y_poison'}, inplace=True)  
        self.complete_y_poison_dataframe.index += 1
        # Define poison data (y_poison) for initial iteration
        self.y_poison_dataframe = self.complete_y_poison_dataframe.iloc[:self.no_psamples_per_subset].reset_index(drop=True)
        self.y_poison_dataframe.index += 1
        self.y_poison_dataframe = round(1 - self.y_poison_dataframe)
        self.attack_initialization = self.y_poison_dataframe  
        
    def inital_sets_size(self):
        """
        Extracts size of sets from all dataframes.
        """
        
        # Initial size of sets
        self.no_samples = len(self.x_train_dataframe.index)
        self.no_psamples = self.y_poison_dataframe.index.size
        self.no_numfeatures = self.num_x_train_dataframe.index.levshape[1]
        self.no_catfeatures = self.cat_x_train_dataframe.index.levshape[1]
        # Create dictionary with number of categories per categorical feature
        categorical_names = set([name.split(':')[0] for name in self.categorical_columns]) 
        self.categories_dict = {int(cat_name) : [int(category.split(':')[1]) for category in self.categorical_columns if category.startswith(cat_name + ':')] for cat_name in categorical_names}
        self.no_categories_dict = {int(cat_name) : len(self.categories_dict[int(cat_name)]) for cat_name in categorical_names}
        print(self.no_categories_dict)

    def regularization_parameter(self):
        """
        Sets the value of the regularization parameter of the regression
        model.
        """
        
        # Other parameters
        self.regularization = 0.6612244897959183
        self.regularization = 0.1

    def update_data(self, new_x_train_num: pd.DataFrame):
        """
        Updates instance data in order to incorporate solutions of previous iteration.

        The input is a dataframe with new data in the following format: multiindex dataframe with sample and feature as index
        and solutions of solving model for x_poison_num as column. Here, x_poison_num becomes x_train_num since solutions to previous 
        iterations become datapoints.
        """

        self.iteration_count += 1
        new_x_train_num = new_x_train_num.unstack(level=1)
        
        ### Update parameters dataframe
        # DATA FEATURES (x_train_num)-------------------------------
        self.num_x_train_dataframe = pd.concat([self.num_x_train_dataframe.unstack(level=1), new_x_train_num], ignore_index=True)  # Add new x_train data (solutions) to old x_train data
        self.num_x_train_dataframe.index += 1
        self.num_x_train_dataframe = self.num_x_train_dataframe.stack().rename_axis(index={None: 'feature'}) 

        extra_cat_dataframe = self.cat_poison_dataframe.copy().rename(columns={'x_poison_cat': 'x_train_cat'})
        first_level = [element for element in range(self.no_samples + 1 , self.no_samples + len(extra_cat_dataframe.index.get_level_values(0)) + 1)]
        extra_cat_dataframe.index.set_levels(first_level, level=0, inplace=True)
        self.cat_x_train_dataframe = pd.concat([self.cat_x_train_dataframe, extra_cat_dataframe], axis=0)  # Add new x_train data (solutions) to old x_train data

        # DATA TARGET (y_train)-------------------------------------
        self.y_train_dataframe = pd.concat([self.y_train_dataframe, self.y_poison_dataframe.rename(columns={'y_poison':'y_train'})]).reset_index(drop=True) # Add poison target to main target dataframe since attacks from previpus iterations become data for next iteration
        self.y_train_dataframe.index.rename('sample')
        self.y_train_dataframe.index += 1

        # ATTACK CAT FEATURES----------------------------------------
        self.cat_poison_dataframe = self.complete_cat_poison_dataframe[(self.iteration_count - 2) * self.no_psamples_per_subset:    # Get next poison samples (by slicing whole poison samples in order)
                                                                       (self.iteration_count - 1) * self.no_psamples_per_subset].reset_index(drop=True)    # Depends on iteration   
        self.cat_poison_dataframe.index.name = 'sample'
        self.cat_poison_dataframe.index += 1
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.cat_poison_dataframe = self.cat_poison_dataframe.stack().rename_axis(index={None: 'column'})   
        self.cat_poison_dataframe.name = 'x_poison_cat'
        self.cat_poison_dataframe = self.cat_poison_dataframe.reset_index()   # This resets index so that current index becomes columns
        # Split multiindex of the form '1:2' into one index for 1 and another index for 2
        self.cat_poison_dataframe[['feature', 'category']] = self.cat_poison_dataframe.column.str.split(':', expand=True).astype(int)
        self.cat_poison_dataframe = self.cat_poison_dataframe.drop(columns=['column'])   # Drops the columns wirth '1:1' names 
        self.cat_poison_dataframe = self.cat_poison_dataframe.set_index(['sample', 'feature', 'category'])   # Sets relevant columns as indices.  

        # ATTACK TARGET (y_poison)------------------------------------
        self.y_poison_dataframe = self.complete_y_poison_dataframe[(self.iteration_count - 1) * self.no_psamples_per_subset:    # Get next poison samples (by slicing whole poison samples in order)
                                                                    self.iteration_count * self.no_psamples_per_subset].reset_index(drop=True)    # Depends on iteration        
        self.y_poison_dataframe.index.rename('sample')
        self.y_poison_dataframe.index += 1

        # Update sets
        self.no_samples = len(self.y_train_dataframe.index)
        self.no_psamples = self.y_poison_dataframe.index.size
     



        