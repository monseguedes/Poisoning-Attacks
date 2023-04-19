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
    def __init__(self, model_parameters):
        """
        The initialization corresponds to the data for the first iteration.
        If there are no iterations (single attack strategy).

        dataset_name: 'pharm', or 'house'
        """
        self.regularization = 0.6612244897959183

        # Whole dataframe with features as columns and target column,
        # as in file (1,2,3..,1:1,1:2,...,target).
        dataset_directory = path.join("data", model_parameters["dataset_name"])
        whole_dataframe = pd.read_csv(
            path.join(dataset_directory, "data-binary.csv"), index_col=[0]
        )

        _cast_column_names_to_int(whole_dataframe, inplace=True)

        # Pick fixed number of trainig samples.
        # The indexes are not reset, but randomly shuffled
        training_samples = model_parameters["training_samples"]
        seed = model_parameters["seed"]
        self.train_dataframe = whole_dataframe.sample(
            frac=None, n=training_samples, random_state=seed
        )

        # Store rest of samples, which will be further divided into testing and
        # validating sets
        self.test_dataframe = whole_dataframe.drop(self.train_dataframe.index)
        self.test_dataframe = self.test_dataframe.reset_index(drop=True)

        self.train_dataframe.reset_index(drop=True, inplace=True)

        poison_rate = model_parameters["poison_rate"] / 100
        self.poison_dataframe = self.train_dataframe.sample(
            frac=poison_rate, random_state=seed
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

    def get_y_train_dataframe(self):
        return get_targets(df=self.train_dataframe)

    #
    def get_num_x_poison_dataframe(self, unstack):
        return get_numerical_features(df=self.poison_dataframe, unstack=unstack)

    def get_cat_x_poison_dataframe(self, unstack):
        return get_categorical_features(df=self.poison_dataframe, unstack=unstack)

    def get_y_poison_dataframe(self):
        return get_targets(df=self.poison_dataframe)

    # def get_complete_cat_poison_dataframe(self):
    #     raise NotImplementedError
    #
    # def get_cat_poison_dataframe_data(self):
    #     raise NotImplementedError
    #
    # def get_cat_poison_dataframe(self):
    #     # Define poison data (x_poison_cat) for initial iteration
    #     cat_poison_dataframe = self.complete_cat_poison_dataframe.iloc[
    #         : self.no_psamples_per_subset
    #     ].reset_index(drop=True)
    #     cat_poison_dataframe.index.name = "sample"
    #     cat_poison_dataframe.index += 1
    #
    #     # Stack dataframe to get multiindex, indexed by sample and feature,
    #     # useful for pyomo format.
    #     cat_poison_dataframe = cat_poison_dataframe.stack().rename_axis(
    #         index={None: "column"}
    #     )
    #     cat_poison_dataframe.name = "x_poison_cat"
    #     # This resets index so that current index becomes columns
    #     cat_poison_dataframe = cat_poison_dataframe.reset_index()
    #     # Split multiindex of the form '1:2' into one index for 1 and another
    #     # index for 2
    #     if len(cat_poison_dataframe) == 0:
    #         cat_poison_dataframe["feature"] = []
    #         cat_poison_dataframe["category"] = []
    #     else:
    #         cat_poison_dataframe[
    #             ["feature", "category"]
    #         ] = cat_poison_dataframe.column.str.split(":", expand=True).astype(int)
    #     # Drops the columns wirth '1:1' names
    #     cat_poison_dataframe = cat_poison_dataframe.drop(columns=["column"])
    #     # Sets relevant columns as indices.
    #     cat_poison_dataframe = cat_poison_dataframe.set_index(
    #         ["sample", "feature", "category"]
    #     )
    #
    #     return cat_poison_dataframe
    #
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

    @property
    def no_train_samples(self):
        return len(self.train_dataframe)

    @property
    def no_poison_samples(self):
        return len(self.poison_dataframe)

    @property
    def no_numfeatures(self):
        return len(get_numerical_feature_column_names(self.train_dataframe))

    @property
    def no_catfeatures(self):
        return len(get_categorical_feature_column_names(self.train_dataframe))

    @property
    def numerical_feature_names(self):
        return get_numerical_feature_column_names(self.train_dataframe)

    @property
    def categorical_feature_names(self):
        return get_categorical_feature_names(self.train_dataframe)

    @property
    def categories_in_categorical_feature(self):
        return get_categorical_feature_to_categories(self.train_dataframe)

    @property
    def no_categories_in_categorical_feature(self):
        return get_categorical_feature_to_no_categories(self.train_dataframe)

    @property
    def no_categories_dict(self):
        import warnings

        warnings.warn(
            "no_categories_dict is deprecated. Use no_categories_in_categorical_feature",
            stacklevel=2,
        )
        return self.no_categories_in_categorical_feature

    def update_numerical_features(self, df):
        if isinstance(df, pd.Series):
            for index, row in df.items():
                self.poison_dataframe.loc[index] = row
        else:
            for index, row in iter:
                self.poison_dataframe[index] = row


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
    [1, 2]

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    names : list[int]
    """
    return [int(x) for x in df.columns if ":" not in str(x) and x != "target"]


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
    df : pandas.DataFrame

    Returns
    -------
    names : list[str]
    """
    return [x for x in df.columns if ":" in str(x)]


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
    [1, 2]

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    names : list[int]
    """
    df = _cast_column_names_to_int(df)
    return sorted(
        set([int(x.split(":")[0]) for x in get_categorical_feature_column_names(df)])
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
    df : pandas.DataFrame

    Returns
    -------
    names : dict[int, list[int]]
    """
    df = _cast_column_names_to_int(df)
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
    df : pandas.DataFrame

    Returns
    -------
    names : dict[int, int]
    """
    df = _cast_column_names_to_int(df)
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

    Parameters
    ----------
    df : pandas.DataFrame
    unstack : bool

    Returns
    -------
    res : pandas.DataFrame
    """
    df = _cast_column_names_to_int(df)
    df = df[get_numerical_feature_column_names(df)]
    if not unstack:
        return df
    df = df.unstack()
    df = df.reset_index()
    df = df.rename(columns={"level_0": "feature", "level_1": "sample"})
    df["feature"] = df["feature"].astype(int)
    df["sample"] = df["sample"].astype(int)
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
    ----------
    df : pandas.DataFrame
    unstack : bool

    Returns
    -------
    res : pandas.DataFrame
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
    df["sample"] = df["sample"].astype(int)
    df["feature"] = df["feature"].astype(int)
    df["category"] = df["category"].astype(int)
    df = df.drop(columns=df.columns[0])
    df = df.set_index(keys=["sample", "feature", "category"]).iloc[:, 0]
    df = df.sort_index()
    return df


def get_targets(df):
    """Extract targets

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
    >>> get_targets(df)
    0    6
    1    7
    2    8
    Name: target, dtype: int64

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    res : pandas.Series
    """
    return df["target"]


def _cast_column_names_to_int(df, inplace=False):
    if inplace:
        out = df
    else:
        out = df.copy()
    out.columns = map(_cast_to_int_if_possible, out.columns)
    return out


def _cast_to_int_if_possible(x):
    try:
        return int(x)
    except ValueError:
        return x
