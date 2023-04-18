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


class InstanceData:
    def __init__(
        self, dataset_name: str
    ):  # Right now names are either 'pharm', or 'house'
        """
        The initialization corresponds to the data for the first iteration. If there are no iterations (single attack strategy).

        dataset_name: 'pharm', or 'house'
        """

        self.dataset_directory = "".join(["data/", dataset_name])  # e.g., data/pharm

    def prepare_instance(
        self,
        poison_rate: int,
        training_samples: int,
        no_psubsets: int,
        seed: int,
    ):
        """
        Prepares the instance by creating dataframe, dividing it into poisoning samples and
        standard samples, defining the sizes of the sets involved in the model, and the
        regularisation parameter. This depends on the poison rate.
        poison_rate: 4, 8, 12, 16, 20.
        training_samples: no. training samples chosen from the whole data.
        N: number of poisoning subsets.
        seed: seed for different random splits of training, validation and testing sets.
        """

        self.seed = seed

        # Poisoning parameters
        self.poison_rate = poison_rate / 100  # 4, 8, 12, 16, or 20
        self.no_psubsets = no_psubsets  # no. of subsets in which the total poison samples (gotten after applying rate to training data) is divided

        # Run all necessary methods
        self.create_dataframes(training_samples, self.seed)
        # print('Splitting daframe')
        # self.split_dataframe()
        # print('Numerical and categorical split')
        # self.num_cat_split()
        # print('Splitting poisoning data')
        # self.poison_samples()
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

        # Whole dataframe with features as columns and target column,
        # as in file (1,2,3..,1:1,1:2,...,target).
        whole_dataframe = pd.read_csv(
            path.join(self.dataset_directory, "data-binary.csv"), index_col=[0]
        )

        # Pick fixed number of trainig samples.
        # The indexes are not reset, but randomly shuffled
        self.train_dataframe = whole_dataframe.sample(
            frac=None, n=training_samples, random_state=seed
        )  # TODO make same format as test

        # Set no. of samples variables.
        # self.no_samples = len(self.train_dataframe.index)
        self.no_total_features = (
            len(self.train_dataframe.columns) - 1
        )  # Remove target column

        # Store rest of samples, which will be further divided into testing and validating sets
        self.test_dataframe = whole_dataframe.drop(self.train_dataframe.index)
        self.test_dataframe = self.test_dataframe.reset_index(drop=True)

        self.train_dataframe.reset_index(drop=True, inplace=True)

        self.poison_dataframe = self.train_dataframe.sample(
            frac=self.poison_rate, random_state=self.seed
        ).reset_index(drop=True)

    # def split_dataframe(self):
    #     """
    #     Splits training dataframe into features dataframe and target dataframe.
    #     This function has two main outputs:
    #     - a dataframe with response variables,
    #     - a dataframe with just the features which mantains the '1:1' notation for
    #     the categorical features,
    #     - a multiindexed dataframe with all features numbered as integers (not
    #     distingushing between numerical and categorical). This last dataframe is
    #     used for the ridge regression model.
    #     """

    #     ### FEATURES (x_train)------------------------
    #     # Get only feature columns and reset index. Columns are still 1,2,.. 1:1,...
    #     self.x_train_dataframe = self.train_dataframe.drop(columns=['target'],
    #                                                        inplace=False)
    #     self.x_train_dataframe.index.name = 'sample'
    #     self.x_train_dataframe.index += 1   # Index starts at 1

    #     ### TARGET (y_train)-------------------------
    #     self.y_train_dataframe = self.train_dataframe[['target']].reset_index(drop=True)
    #     self.y_train_dataframe.rename(columns={'target': 'y_train'}, inplace=True)  # Rename column as y_train, which is the name of the pyomo parameter
    #     self.y_train_dataframe.index.name = 'sample'
    #     self.y_train_dataframe.index += 1

    #     return self.x_train_dataframe, self.y_train_dataframe

    # def num_cat_split(self): """
    #     Splits the features dataframe into one multiindexed dataframe for numerical
    #     features and one multiindexed dataframe for categorical features.
    #     """
    #
    #     ### NUMERICAL FEATURES-------------------------------
    #     # Get only numerical columns (those that are just integers) and convert them to integer type
    #     self.numerical_columns = [name for name in self.x_train_dataframe.columns if ':' not in name]
    #     # self.num_x_train_dataframe = self.x_train_dataframe[self.numerical_columns]
    #     # self.num_x_train_dataframe.columns = self.num_x_train_dataframe.columns.astype(int) # Make column names integers so that
    #     #                                                                                     # they can later be used as pyomo indices
    #     # # Stack dataframe to get multiindex, indexed by sample and feature
    #     # self.num_x_train_dataframe = self.num_x_train_dataframe.stack().rename_axis(index={None: 'feature'})
    #     # self.num_x_train_dataframe.name = 'x_train_num'
    #
    #     ### CATEGORICAL FEATURES------------------------------
    #     # Get only categorical columns (those that include ':' in name)
    #     self.categorical_columns = [name for name in self.x_train_dataframe.columns if name not in self.numerical_columns]
    #     self.cat_x_train_dataframe = self.x_train_dataframe[self.categorical_columns]
    #     # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
    #     self.cat_x_train_dataframe = self.cat_x_train_dataframe.stack().rename_axis(index={None: 'column'})
    #     self.cat_x_train_dataframe.name = 'x_train_cat'
    #     self.cat_x_train_dataframe = self.cat_x_train_dataframe.reset_index()   # This resets index so that current index becomes columns
    #     # Split multiindex of the form '1:2' into one index for 1 and another index for 2
    #     if len(self.cat_x_train_dataframe) == 0:
    #         self.cat_x_train_dataframe['feature'] = []
    #         self.cat_x_train_dataframe['category'] = []
    #     else:
    #         self.cat_x_train_dataframe[['feature', 'category']] = self.cat_x_train_dataframe.column.str.split(':', expand=True).astype(int)
    #     self.cat_x_train_dataframe = self.cat_x_train_dataframe.drop(columns=['column'])   # Drops the columns wirth '1:1' names
    #     self.cat_x_train_dataframe = self.cat_x_train_dataframe.set_index(['sample', 'feature', 'category'])   # Sets relevant columns as indices.
    #
    #     return self.num_x_train_dataframe, self.cat_x_train_dataframe

    # def poison_samples(self):
    #     """
    #     Takes the dataframe for training data and gets data for poisoning samples
    #     depending on poisoning rate
    #     """
    #
    #     # Dataframe with all samples to be poisoned
    #     self.poison_dataframe = self.train_dataframe.sample(frac= self.poison_rate,
    #                                                         random_state=self.seed).reset_index(drop=True)
    #
    #     # Total number of poisoned samples (rate applied to training data)
    #     self.no_total_psamples = self.poison_dataframe.index.size
    #     # Get the biggest number of samples per subset that makes possible the desired number of subsets
    #     if self.no_psubsets == 0:
    #         self.no_psamples_per_subset = 0
    #     else:
    #         self.no_psamples_per_subset = floor(self.no_total_psamples / self.no_psubsets)
    #         if self.no_psamples_per_subset == 0:
    #             raise SystemError('The ratio between poisoning samples and poisoning subset is not feasible')
    #     # Now multiplies the no. samples per subset and no. of subset to get total poisoned samples
    #     self.no_total_psamples = self.no_psamples_per_subset * self.no_psubsets
    #     # If the initial poison data had a non divisible number of samples, update it to be divisible
    #     self.poison_dataframe = self.poison_dataframe.iloc[:self.no_total_psamples]
    #
    #     # ### Initial poisoning samples-------------------------
    #     # self.num_x_poison_dataframe = self.complete_x_poison_dataframe[self.numerical_columns]
    #     # self.num_x_poison_dataframe.index += 1
    #     # # Initialise those to be poisoned to be opposite
    #     # # def flip_nearest(x):
    #     # #     if x < 0.5:
    #     # #         return 1
    #     # #     else:
    #     # #         return 0
    #     # # for feature in self.numerical_columns:
    #     # #     self.num_x_poison_dataframe[feature]= self.num_x_poison_dataframe[feature].apply(lambda x: flip_nearest(x))
    #     # self.num_x_poison_dataframe.columns = self.num_x_poison_dataframe.columns.astype(int)
    #     # self.num_x_poison_dataframe = self.num_x_poison_dataframe.stack().rename_axis(index={None: 'feature'})
    #     # self.num_x_poison_dataframe.name = 'x_data_poison_num'
    #
    #     ### TARGET (y_poison)---------------------------------
    #     self.complete_y_poison_dataframe = self.poison_dataframe[['target']].reset_index(drop=True)
    #     self.complete_y_poison_dataframe.rename(columns={'target': 'y_poison'}, inplace=True)
    #     self.complete_y_poison_dataframe.index += 1
    #     # self.complete_y_poison_dataframe = round(1 - self.complete_y_poison_dataframe)
    #
    #     self.y_poison_dataframe = self.complete_y_poison_dataframe[:self.no_psamples_per_subset]
    #
    #     ### FLAG
    #     self.flag_array = np.ones(self.no_total_psamples)
    #     self.flag_array[:self.no_psamples_per_subset]=0
    #
    #
    def get_numerical_columns(self):
        return [
            name
            for name in self.train_dataframe.columns
            if ":" not in name and name != "target"
        ]

    def get_categorical_columns(self):
        return [name for name in self.poison_dataframe.columns if ":" in name]

    # def get_num_x_train_dataframe(self):
    #     numerical_columns = self.get_numerical_columns()
    #     num_x_train_dataframe = self.train_dataframe[numerical_columns]
    #     num_x_train_dataframe.columns = num_x_train_dataframe.columns.astype(int) # Make column names integers so that
    #                                                                                         # they can later be used as pyomo indices
    #     # Stack dataframe to get multiindex, indexed by sample and feature
    #     num_x_train_dataframe = num_x_train_dataframe.stack().rename_axis(index={None: 'feature'})
    #     # num_x_train_dataframe.name = 'x_train_num'
    #     return num_x_train_dataframe
    #
    def get_num_x_train_dataframe(self, unstack):
        return get_numerical_features(df=self.train_dataframe, unstack=unstack)

    # def get_cat_x_train_dataframe(self, stack):
    #     categorical_columns = self.get_categorical_columns()
    #     cat_x_train_dataframe = self.train_dataframe[categorical_columns]
    #
    #     if not stack: return cat_x_train_dataframe # TODO remove, complete one is unstacked, data on is stacked
    #
    #     # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
    #     cat_x_train_dataframe = cat_x_train_dataframe.unstack()
    #     cat_x_train_dataframe = cat_x_train_dataframe.reset_index()
    #     cat_x_train_dataframe = cat_x_train_dataframe.rename(columns={'level_0': 'feature_category', 'level_1': 'sample'})
    #     # Split multiindex of the form '1:2' into one index for 1 and another index for 2
    #     if len(cat_x_train_dataframe) == 0:
    #         cat_x_train_dataframe['feature'] = []
    #         cat_x_train_dataframe['category'] = []
    #     else:
    #         cat_x_train_dataframe[['feature', 'category']] = cat_x_train_dataframe.feature_category.str.split(':', expand=True).astype(int)
    #     cat_x_train_dataframe = cat_x_train_dataframe.drop(columns=['feature_category'])   # Drops the columns wirth '1:1' names
    #     cat_x_train_dataframe = cat_x_train_dataframe.set_index(['sample', 'feature', 'category'])   # Sets relevant columns as indices.
    #     cat_x_train_dataframe = cat_x_train_dataframe.sort_index()
    #     return cat_x_train_dataframe

    def get_cat_x_train_dataframe(self, unstack):
        return get_categorical_features(df=self.train_dataframe, unstack=unstack)

    # def get_num_x_poison_dataframe(self):
    #     numerical_columns = self.get_numerical_columns()
    #     num_x_poison_dataframe = self.poison_dataframe[numerical_columns]
    #     num_x_poison_dataframe.columns = num_x_poison_dataframe.columns.astype(int) # Make column names integers so that
    #                                                                                         # they can later be used as pyomo indices
    #     # Stack dataframe to get multiindex, indexed by sample and feature
    #     num_x_poison_dataframe = num_x_poison_dataframe.stack().rename_axis(index={None: 'feature'})
    #     num_x_poison_dataframe.name = 'x_poison_num'
    #
    def get_num_x_poison_dataframe(self, unstack):
        return get_numerical_features(df=self.poison_dataframe, unstack=unstack)

    def get_cat_x_poison_dataframe(self, unstack):
        return get_categorical_features(df=self.poison_dataframe, unstack=unstack)

    def get_complete_cat_poison_dataframe(self):
        raise NotImplementedError

    def get_cat_poison_dataframe_data(self):
        raise NotImplementedError

    def get_cat_poison_dataframe(self):
        # Define poison data (x_poison_cat) for initial iteration
        cat_poison_dataframe = self.complete_cat_poison_dataframe.iloc[
            : self.no_psamples_per_subset
        ].reset_index(drop=True)
        cat_poison_dataframe.index.name = "sample"
        cat_poison_dataframe.index += 1
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        cat_poison_dataframe = cat_poison_dataframe.stack().rename_axis(
            index={None: "column"}
        )
        cat_poison_dataframe.name = "x_poison_cat"
        cat_poison_dataframe = (
            cat_poison_dataframe.reset_index()
        )  # This resets index so that current index becomes columns
        # Split multiindex of the form '1:2' into one index for 1 and another index for 2
        if len(cat_poison_dataframe) == 0:
            cat_poison_dataframe["feature"] = []
            cat_poison_dataframe["category"] = []
        else:
            cat_poison_dataframe[
                ["feature", "category"]
            ] = cat_poison_dataframe.column.str.split(":", expand=True).astype(int)
        cat_poison_dataframe = cat_poison_dataframe.drop(
            columns=["column"]
        )  # Drops the columns wirth '1:1' names
        cat_poison_dataframe = cat_poison_dataframe.set_index(
            ["sample", "feature", "category"]
        )  # Sets relevant columns as indices.

        return cat_poison_dataframe

    @property
    def no_samples(self):
        import warnings

        warnings.warn("no_samples is deprecated. Use no_train_samples", stacklevel=2)
        return self.no_train_samples

    @property
    def no_psamples(self):
        import warnings

        warnings.warn("no_psamples is deprecated. Use no_poison_samples", stacklevel=2)
        return self.no_poison_samples

    def inital_sets_size(self):
        """
        Extracts size of sets from all dataframes.
        """
        # Initial size of sets
        self.no_train_samples = len(self.train_dataframe)
        self.no_poison_samples = len(self.poison_dataframe)
        self.no_numfeatures = len(
            get_numerical_feature_column_names(self.train_dataframe)
        )
        self.no_catfeatures = len(
            get_categorical_feature_column_names(self.train_dataframe)
        )
        # Create dictionary with number of categories per categorical feature
        categorical_names = get_categorical_feature_names(self.train_dataframe)
        self.categories_dict = get_categorical_feature_to_categories(
            self.train_dataframe
        )
        self.no_categories_dict = get_categorical_feature_to_no_categories(
            self.train_dataframe
        )

    def regularization_parameter(self):
        """
        Sets the value of the regularization parameter of the regression
        model.
        """

        # Other parameters
        self.regularization = 0.6612244897959183

    def update_data(self, iteration: int, new_x_poison_num: pd.DataFrame):
        """
        Updates instance data in order to incorporate solutions of previous iteration.

        The input is a dataframe with new data in the following format: multiindex dataframe with sample and feature as index
        and solutions of solving model for x_poison_num as column. Here, x_poison_num becomes x_train_num since solutions to previous
        iterations become datapoints.
        """
        if iteration == self.no_psubsets:
            next_iteration = 0
        else:
            next_iteration = iteration

        ### NUMERICAL POISON (x_poison_num_data)-------------------------------
        self.num_x_poison_dataframe.to_numpy()[
            (iteration - 1)
            * self.no_numfeatures
            * self.no_psamples_per_subset : iteration
            * self.no_numfeatures
            * self.no_psamples_per_subset
        ] = new_x_poison_num.to_numpy()

        ### ATTACK CATEGORICAL FEATURES (x_poison_cat)
        self.cat_poison_dataframe = self.complete_cat_poison_dataframe.iloc[
            next_iteration
            * self.no_psamples_per_subset : (next_iteration + 1)
            * self.no_psamples_per_subset
        ].reset_index(drop=True)
        self.cat_poison_dataframe.index.name = "sample"
        self.cat_poison_dataframe.index += 1
        # Stack dataframe to get multiindex, indexed by sample and feature, useful for pyomo format.
        self.cat_poison_dataframe = self.cat_poison_dataframe.stack().rename_axis(
            index={None: "column"}
        )
        self.cat_poison_dataframe.name = "x_poison_cat"
        self.cat_poison_dataframe = (
            self.cat_poison_dataframe.reset_index()
        )  # This resets index so that current index becomes columns
        # Split multiindex of the form '1:2' into one index for 1 and another index for 2
        self.cat_poison_dataframe[
            ["feature", "category"]
        ] = self.cat_poison_dataframe.column.str.split(":", expand=True).astype(int)
        self.cat_poison_dataframe = self.cat_poison_dataframe.drop(
            columns=["column"]
        )  # Drops the columns wirth '1:1' names
        self.cat_poison_dataframe = self.cat_poison_dataframe.set_index(
            ["sample", "feature", "category"]
        )  # Sets relevant columns as indices.

        ### ATTACK TARGET (y_poison)------------------------------------
        self.y_poison_dataframe = self.complete_y_poison_dataframe[
            next_iteration
            * self.no_psamples_per_subset : (  # Get next poison samples (by slicing whole poison samples in order)
                next_iteration + 1
            )
            * self.no_psamples_per_subset
        ].reset_index(
            drop=True
        )  # Depends on iteration
        self.y_poison_dataframe.index.rename("sample")
        self.y_poison_dataframe.index += 1

        ### UPDATE FLAG
        self.flag_array = np.ones(self.no_psamples_per_subset)
        self.flag_array[
            next_iteration
            * self.no_psamples_per_subset : (next_iteration + 1)
            * self.no_psamples_per_subset
        ] = 0


def get_numerical_feature_column_names(df):
    """Extract the column names of numerical features

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ...     "1:1":    [ 1,  0,  0],
    ...     "1:2":    [ 0,  1,  0],
    ...     "2:1":    [ 0,  0,  1],
    ...     "2:2":    [ 0,  1,  0],
    ...     "2:3":    [ 0,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> get_numerical_feature_column_names(df)
    ['1', '2']

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    names : list
    """
    return [x for x in df.columns if ":" not in x and x != "target"]


def get_categorical_feature_column_names(df):
    """Extract the column names of categorical features

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ...     "1:1":    [ 1,  0,  0],
    ...     "1:2":    [ 0,  1,  0],
    ...     "2:1":    [ 0,  0,  1],
    ...     "2:2":    [ 0,  1,  0],
    ...     "2:3":    [ 0,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> get_categorical_feature_column_names(df)
    ['1:1', '1:2', '2:1', '2:2', '2:3']

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    names : list
    """
    return [x for x in df.columns if ":" in x]


def get_categorical_feature_names(df):
    """Extract the column names of categorical features

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ...     "1:1":    [ 1,  0,  0],
    ...     "1:2":    [ 0,  1,  0],
    ...     "2:1":    [ 0,  0,  1],
    ...     "2:2":    [ 0,  1,  0],
    ...     "2:3":    [ 0,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> get_categorical_feature_names(df)
    ['1', '2']

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    names : list
    """
    return sorted(
        set([x.split(":")[0] for x in get_categorical_feature_column_names(df)])
    )


def get_categorical_feature_to_categories(df):
    """Extract the column names of categorical features

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ...     "1:1":    [ 1,  0,  0],
    ...     "1:2":    [ 0,  1,  0],
    ...     "2:1":    [ 0,  0,  1],
    ...     "2:2":    [ 0,  1,  0],
    ...     "2:3":    [ 0,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> get_categorical_feature_to_categories(df)
    {1: [1, 2], 2: [1, 2, 3]}

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    names : list
    """
    out = dict()
    for column_name in get_categorical_feature_column_names(df):
        feature, category = map(int, column_name.split(":"))
        out.setdefault(feature, [])
        out[feature].append(category)
    return out


def get_categorical_feature_to_no_categories(df):
    """Extract the column names of categorical features

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ...     "1:1":    [ 1,  0,  0],
    ...     "1:2":    [ 0,  1,  0],
    ...     "2:1":    [ 0,  0,  1],
    ...     "2:2":    [ 0,  1,  0],
    ...     "2:3":    [ 0,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> get_categorical_feature_to_no_categories(df)
    {1: 2, 2: 3}

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    names : list
    """
    dct = get_categorical_feature_to_categories(df)
    return {k: len(v) for k, v in dct.items()}


def get_numerical_features(df, unstack):
    """Extract numerical features

    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ...     "1:1":    [ 1,  0,  0],
    ...     "1:2":    [ 0,  1,  0],
    ...     "2:1":    [ 0,  0,  1],
    ...     "2:2":    [ 0,  1,  0],
    ...     "2:3":    [ 0,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> get_numerical_features(df, unstack=False)
       1  2
    0  0  3
    1  1  4
    2  2  5
    >>> get_numerical_features(df, unstack=True)
    sample  feature
    0       1          0
            2          3
    1       1          1
            2          4
    2       1          2
            2          5
    Name: 0, dtype: int64
    """
    df = df[get_numerical_feature_column_names(df)]
    if not unstack:
        return df
    df = df.unstack()
    df = df.reset_index()
    df = df.rename(columns={"level_0": "feature", "level_1": "sample"})
    df = df.set_index(["sample", "feature"])[0]
    df = df.sort_index()
    return df


def get_categorical_features(df, unstack):
    """Extract categorical features

    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ...     "1:1":    [ 1,  0,  1],
    ...     "1:2":    [ 0,  1,  0],
    ...     "2:1":    [ 0,  0,  0],
    ...     "2:2":    [ 0,  1,  0],
    ...     "2:3":    [ 1,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> get_categorical_features(df, unstack=False)
       1:1  1:2  2:1  2:2  2:3
    0    1    0    0    0    1
    1    0    1    0    1    0
    2    1    0    0    0    1

    >>> get_categorical_features(df, unstack=True)
    sample  feature  category
    0       1        1           1
                     2           0
            2        1           0
                     2           0
                     3           1
    1       1        1           0
                     2           1
            2        1           0
                     2           1
                     3           0
    2       1        1           1
                     2           0
            2        1           0
                     2           0
                     3           1
    Name: 0, dtype: int64

    Parameters
    """
    df = df[get_categorical_feature_column_names(df)]
    if not unstack:
        return df
    # The first index makes le
    df.index.name = "sample"
    df = df.unstack()
    # df.name = "value"
    df = df.reset_index()
    df[["feature", "category"]] = df.iloc[:, 0].str.split(":", expand=True).astype(int)
    df = df.drop(columns=df.columns[0])
    df = df.set_index(keys=["sample", "feature", "category"]).iloc[:, 0]
    df = df.sort_index()
    return df
