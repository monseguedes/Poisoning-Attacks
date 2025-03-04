"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

This script creates the class with all the data that is then given to either the bilevel
model or the ridge regression model.
"""

import itertools

# Add path to be able to use module from other folder
import sys
from math import floor
from os import path

import numpy as np

# Python imports
import pandas as pd

sys.path.append("./programs/minlp/algorithm")
import choosing_features as choose

long_space = 80
short_space = 60


class InstanceData:
    """
    This class is the instance that is the fed into either the bilevel model
    or the ridge regression model.
    """

    def __init__(self, dataset_name: str):
        """
        dataset_name: 'pharm', or 'house'
        """

        self.dataset_directory = "".join(["data/", dataset_name])

    def prepare_instance(
        self,
        poison_rate: int,
        training_samples: int,
        no_poison_subsets: int,
        no_nfeatures: int,
        no_cfeatures: int,
        seed: int,
    ):
        """
        Prepares the instance by creating dataframe, dividing it into poisoning samples and
        standard samples, defining the sizes of the sets involved in the model, and the
        regularisation parameter. This depends on the poison rate.
        poisson_rate: 4, 8, 12, 16, 20.
        training_samples: no. training samples chosen from the whole data.
        seed: seed for different random splits of training, validation and testing sets.
        """

        self.seed = seed

        # Poisoning parameters
        self.poison_rate = poison_rate / 100  # 4, 8, 12, 16, or 20
        self.no_poison_subsets = no_poison_subsets
        self.no_nfeatures = no_nfeatures
        self.no_cfeatures = no_cfeatures

        # Run all necessary methods
        self.create_dataframes(training_samples, self.seed)
        print("Splitting daframe")
        self.split_dataframe()
        print("Numerical and categorical split")
        self.num_cat_split()
        print("Choosing features")
        if not no_nfeatures == "all" and not no_cfeatures == "all":
            self.feature_selection(no_nfeatures, no_cfeatures)
        print("Splitting poisoning data")
        self.poison_samples()
        print("Defining sets")
        self.inital_sets_size()
        self.regularization_parameter()

    def create_dataframes(self, training_samples: int, seed: int):
        """
        Creates a dataframe with all the data, which has features and traget as columns,
        and samples as rows. Numerical columns are integers, while categorical columns are
        of the form '1:1' for 'no.catfeature:no.category'. Response variable is names as
        'target'. These files are prepared by preprocessing.
        """

        ### Main dataframes
        # Whole dataframe with features as columns and target column, as in file.
        self.whole_dataframe = pd.read_csv(
            path.join(self.dataset_directory, "data-binary.csv"), index_col=[0]
        )

        # Pick fixed number of trainig samples.
        self.train_dataframe = self.whole_dataframe.sample(
            frac=None, n=training_samples, random_state=seed
        )  # The indexes are not reset, but randomly shuffled

        # Store rest of samples, which will be further divided into testing and validating sets
        self.test_validation_dataframe = self.whole_dataframe.drop(
            self.train_dataframe.index
        )

        self.test_dataframe = self.test_validation_dataframe.sample(
            frac=None,
            n=min(
                5 * training_samples, len(self.test_validation_dataframe.index)
            ),
            random_state=seed,
        )  # The indexes are not reset, but randomly shuffled

        self.test_dataframe = self.test_dataframe.reset_index(drop=True)
        self.test_dataframe.index.name = "sample"
        self.test_dataframe.index += 1  # Index starts at 1
        self.test_y = self.test_dataframe["target"]
        self.test_dataframe = self.test_dataframe.drop(
            columns=["target"], inplace=False
        )

        # Change dataframe column names to create dataframe for ridge model.
        self.test_ridge_x_train_dataframe = self.test_dataframe.copy()
        self.test_ridge_x_train_dataframe.columns = [
            count + 1
            for count, value in enumerate(self.test_dataframe.columns)
        ]
        self.test_ridge_x_train_dataframe = (
            self.test_ridge_x_train_dataframe.stack().rename_axis(
                index={None: "feature"}
            )
        )

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

        ### FEATURES (x_train)
        # Get only feature columns and reset index. Columns are still 1,2,.. 1:1,...
        self.x_train_dataframe = self.train_dataframe.drop(
            columns=["target"], inplace=False
        ).reset_index(drop=True)
        self.x_train_dataframe.index.name = "sample"
        self.x_train_dataframe.index += 1  # Index starts at 1

        ### Select POISON SAMPLES (from training data)-----------------------
        # Dataframe with all samples to be poisoned

        # Dataframe with all samples to be poisoned
        self.poison_dataframe = self.train_dataframe.sample(
            frac=self.poison_rate, random_state=self.seed
        ).reset_index(drop=True)
        # Total number of poisoned samples (rate applied to training data)
        self.no_total_psamples = self.poison_dataframe.index.size
        # Get the biggest number of samples per subset that makes possible the desired number of subsets
        self.no_psamples_per_subset = floor(
            self.no_total_psamples / self.no_poison_subsets
        )
        # Now multiplies the no. samples per subset and no. of subset to get total poisoned samples
        self.no_total_psamples = (
            self.no_psamples_per_subset * self.no_poison_subsets
        )
        # If the initial poison data had a non divisible number of samples, update it to be divisible
        self.poison_dataframe = self.poison_dataframe.iloc[
            : self.no_total_psamples
        ]
        self.x_poison_dataframe = self.poison_dataframe.drop(
            columns=["target"], inplace=False
        ).reset_index(drop=True)
        self.x_poison_dataframe.index.name = "sample"
        self.x_poison_dataframe.index += 1
        # Number of poisoned samples (rate applied to training data)
        self.no_psamples = self.poison_dataframe.index.size

        # Set no. of samples variables.
        self.no_samples = len(self.x_train_dataframe.index)
        self.no_features = len(self.x_train_dataframe.columns)

        # Change dataframe column names to create dataframe for ridge model.
        self.ridge_x_train_dataframe = self.x_train_dataframe.copy()
        self.ridge_x_train_dataframe.columns = [
            count + 1
            for count, value in enumerate(self.x_train_dataframe.columns)
        ]
        self.ridge_x_train_dataframe = (
            self.ridge_x_train_dataframe.stack().rename_axis(
                index={None: "feature"}
            )
        )

        ### TARGET (y_train)
        # Get only target column
        self.y_train_dataframe = self.train_dataframe[["target"]].reset_index(
            drop=True
        )
        self.y_train_dataframe.rename(
            columns={"target": "y_train"}, inplace=True
        )  # Rename as y_train, which is the name of the pyomo parameter
        self.y_train_dataframe.index.name = "sample"
        self.y_train_dataframe.index += 1

        return self.x_train_dataframe, self.y_train_dataframe

    def num_cat_split(self):
        """
        Splits the features dataframe into one multiindexed dataframe for numerical
        features and one multiindexed dataframe for categorical features.
        """

        ### NUMERICAL FEATURES
        # Get only numerical columns (those that are just integers) and convert them to integer type
        self.numerical_columns = [
            name for name in self.x_train_dataframe.columns if ":" not in name
        ]
        self.num_x_train_dataframe = self.x_train_dataframe[
            self.numerical_columns
        ]
        self.num_x_train_dataframe.columns = (
            self.num_x_train_dataframe.columns.astype(int)
        )  # Make column names integers so that
        # they can later be used as pyomo indices
        # Stack dataframe to get multiindex, indexed by sample and feature, this is nice when converted
        # to dictionary and used as data since matched gurobi's format.
        self.num_x_train_dataframe = (
            self.num_x_train_dataframe.stack().rename_axis(
                index={None: "feature"}
            )
        )
        self.num_x_train_dataframe.name = "x_train_num"

        ### CATEGORICAL FEATURES
        # Get only categorical columns (those that include ':' in name)
        self.categorical_columns = [
            name
            for name in self.x_train_dataframe.columns
            if name not in self.numerical_columns
        ]
        self.cat_x_train_dataframe = self.x_train_dataframe[
            self.categorical_columns
        ]
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.cat_x_train_dataframe = (
            self.cat_x_train_dataframe.stack().rename_axis(
                index={None: "column"}
            )
        )
        self.cat_x_train_dataframe.name = "x_train_cat"
        self.cat_x_train_dataframe = (
            self.cat_x_train_dataframe.reset_index()
        )  # This resets index so that current index becomes columns
        # Split multiindex of the form '1:2' into one index for 1 and another index for 2
        if len(self.cat_x_train_dataframe) == 0:
            self.cat_x_train_dataframe["feature"] = []
            self.cat_x_train_dataframe["category"] = []
        else:
            self.cat_x_train_dataframe[
                ["feature", "category"]
            ] = self.cat_x_train_dataframe.column.str.split(
                ":", expand=True
            ).astype(
                int
            )
        self.cat_x_train_dataframe = self.cat_x_train_dataframe.drop(
            columns=["column"]
        )  # Drops the columns wirth '1:1' names
        self.cat_x_train_dataframe = self.cat_x_train_dataframe.set_index(
            ["sample", "feature", "category"]
        )  # Sets relevant columns as indices.

        return self.num_x_train_dataframe, self.cat_x_train_dataframe

    def feature_selection(self, no_nfeatures: int, no_cfeatures: int):
        """
        Run LASSO model to pick most important features.
        """
        self.chosen_numerical, self.chosen_categorical = choose.LASSOdataframe(
            self.train_dataframe
        ).get_features_lists(no_nfeatures, no_cfeatures)

        return self.chosen_numerical, self.chosen_categorical

    def poison_samples(self, only_num=False):
        """
        Takes the dataframe for training data and gets data for poisoning samples
        depending on poisoning rate
        """

        ### NUMERICAL FEATURES (x_data_poison_num)------------------------------------
        self.num_x_poison_dataframe = self.x_poison_dataframe[
            self.numerical_columns
        ]

        # Initialise those to be poisoned to be opposite
        def flip_nearest(x):
            if x < 0.5:
                return 1
            else:
                return 0

        # if self.no_nfeatures != 'all':
        #     to_flip = self.chosen_numerical
        # else:
        #     to_flip = self.numerical_columns
        # for numerical in to_flip:
        #     numerical = str(numerical)
        #     self.num_x_poison_dataframe[numerical] = self.num_x_poison_dataframe[numerical].apply(lambda x: flip_nearest(x))

        self.num_x_poison_dataframe.columns = (
            self.num_x_poison_dataframe.columns.astype(int)
        )  # Make column names integers so that
        # they can later be used as pyomo indices
        # Stack dataframe to get multiindex, indexed by sample and feature, this is nice when converted
        # to dictionary and used as data since matched gurobi's format.
        self.num_x_poison_dataframe = (
            self.num_x_poison_dataframe.stack().rename_axis(
                index={None: "feature"}
            )
        )
        self.num_x_poison_dataframe.name = "x_data_poison_num"

        if only_num:
            return

        ### CATEGORICAL FEATURES (x_data_poison_num)----------------------------------
        # Get only categorical columns (those that include ':' in name)
        self.cat_x_poison_dataframe = self.x_poison_dataframe[
            self.categorical_columns
        ]
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.cat_x_poison_dataframe = (
            self.cat_x_poison_dataframe.stack().rename_axis(
                index={None: "column"}
            )
        )
        self.cat_x_poison_dataframe.name = "x_data_poison_cat"
        self.cat_x_poison_dataframe = (
            self.cat_x_poison_dataframe.reset_index()
        )  # This resets index so that current index becomes columns
        # Split multiindex of the form '1:2' into one index for 1 and another index for 2
        if len(self.cat_x_poison_dataframe) == 0:
            self.cat_x_poison_dataframe["feature"] = []
            self.cat_x_poison_dataframe["category"] = []
        else:
            self.cat_x_poison_dataframe[
                ["feature", "category"]
            ] = self.cat_x_poison_dataframe.column.str.split(
                ":", expand=True
            ).astype(
                int
            )
        self.cat_x_poison_dataframe = self.cat_x_poison_dataframe.drop(
            columns=["column"]
        )  # Drops the columns wirth '1:1' names
        self.cat_x_poison_dataframe = self.cat_x_poison_dataframe.set_index(
            ["sample", "feature", "category"]
        )  # Sets relevant columns as indices.

        ### TARGET (y_poison)---------------------------------------------------------
        # Get only target column from poison dataframe
        self.y_poison_dataframe = self.poison_dataframe[
            ["target"]
        ].reset_index(drop=True)
        self.y_poison_dataframe.rename(
            columns={"target": "y_poison"}, inplace=True
        )  # y_poison is the name of the pyomo parameter
        self.y_poison_dataframe.index += 1
        # self.y_poison_dataframe = round(1 - self.y_poison_dataframe)

        return self.y_poison_dataframe

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
        categorical_names = set(
            [name.split(":")[0] for name in self.categorical_columns]
        )
        self.categories_dict = {
            int(cat_name): [
                int(category.split(":")[1])
                for category in self.categorical_columns
                if category.startswith(cat_name + ":")
            ]
            for cat_name in categorical_names
        }
        self.no_categories_dict = {
            int(cat_name): len(self.categories_dict[int(cat_name)])
            for cat_name in categorical_names
        }

    def regularization_parameter(self):
        """
        Sets the value of the regularization parameter of the regression
        model.
        """

        # Other parameters
        self.regularization = 0.6612244897959183
        # self.regularization = 0.01

    def append_poisoning_attacks(self, solutions):
        """
        Takes the solutions from the poisoning attacks and concatenates them
        to the data so that a regression model can be fit on the whole dataset.
        solutions: solution output from the bilevel solution algorithm.
        """

        # Add poisoning samples to training dataframe
        df = self.x_train_dataframe.copy()

        # Create dataframe with solutions of numerical features
        num_dict = {
            (str(triple[0]), str(triple[1])): solutions["x_poison_num"][triple]
            for triple in solutions["x_poison_num"].keys()
        }
        index = pd.MultiIndex.from_tuples(
            num_dict.keys(), names=("sample", "feature")
        )  # Create index from the keys (indexes) of the solutions of x_poison
        num_features = pd.Series(
            solutions["x_poison_num"].values(), index=index
        )  # Make a dataframe with solutions and desires index
        num_df = num_features.unstack()

        # Create dataframe with solutions of categorical features
        cat_dict = {
            (str(triple[0]), str(triple[1]) + ":" + str(triple[2])): solutions[
                "x_poison_cat"
            ][triple]
            for triple in solutions["x_poison_cat"].keys()
        }
        index = pd.MultiIndex.from_tuples(
            cat_dict.keys(), names=("sample", "feature")
        )  # Create index from the keys (indexes) of the solutions of x_poison
        cat_features = pd.Series(
            cat_dict.values(), index=index
        )  # Make a dataframe with solutions and desires index
        cat_df = cat_features.unstack()

        # Join numerical and categorical dataframes
        poisoning_df = pd.concat([num_df, cat_df], axis=1)

        # Join original and poisoned dataframe
        self.whole_df = pd.concat([df, poisoning_df], axis=0)
        self.whole_df.reset_index(inplace=True, drop=True)
        self.whole_df.index += 1
        self.whole_df.columns = [
            count + 1 for count, value in enumerate(self.whole_df.columns)
        ]
        self.no_features = len(self.whole_df.columns)
        self.no_samples = len(self.whole_df.index)
        self.whole_df = self.whole_df.stack().rename_axis(
            index={None: "feature"}
        )

        # Add poisoning y to training y
        y_train = self.y_train_dataframe.copy(deep=True)
        y_poison = self.y_poison_dataframe.copy(deep=True)
        y_poison.rename(columns={"y_poison": "y_train"}, inplace=True)
        self.whole_y = pd.concat([y_train, y_poison])
        self.whole_y.reset_index(inplace=True, drop=True)
        self.whole_y.index += 1

        # Update objects that will be used as parameters.
        self.ridge_x_train_dataframe = self.whole_df
        self.y_train_dataframe = self.whole_y
