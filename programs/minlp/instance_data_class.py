"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

This script creates the class with all the data that is then given to the benckmark model.
"""

# TODO Improve consistency of naming.

import copy
from os import path

import numpy as np
import pandas as pd

# import choosing_features


class InstanceData:
    def __init__(self, config, benchmark_data=False, seed=123, thesis=False):
        """
        The initialization corresponds to the data for the first iteration.
        If there are no iterations (single attack strategy).

        dataset_name: 'pharm', or 'house'
        """
        self.regularization = config["regularization"]

        # Whole dataframe with features as columns and target column,
        # as in file (1,2,3..,1:1,1:2,...,target).
        if thesis:
            dataset_directory = path.join("data", "thesis_" + config["dataset"], config["dataset_name"])
            whole_dataframe = pd.read_csv(
                path.join(dataset_directory, "data-binary.csv"), index_col=[0]
            )
        else:
            dataset_directory = path.join("data", config["dataset_name"])
            whole_dataframe = pd.read_csv(
                path.join(dataset_directory, "data-binary.csv"), index_col=[0]
            )

        if benchmark_data:
            # ALTERNATIVE FROM BENCHMARK-----------------
            whole_dataframe = pd.read_csv(
                f"programs/benchmark/manip-ml-master/datasets/house/{seed}_whole_dataframe.csv",
                index_col=[0],
            )


        _cast_column_names_to_int(whole_dataframe, inplace=True)

        # Pick fixed number of trainig samples.
        # The indexes are not reset, but randomly shuffled
        training_samples = config["training_samples"]
        # seed = config["seed"]
        self.train_dataframe = whole_dataframe.sample(
            frac=None, n=training_samples, random_state=seed
        )

        if benchmark_data:
            # ALTERNATIVE FROM BENCHMARK-----------------
            train_array = np.load(
                f"programs/benchmark/manip-ml-master/datasets/house/{seed}_train_array.npy"
            )
            print(train_array.shape)
            print(len(whole_dataframe.columns))
            self.train_dataframe = pd.DataFrame(
                train_array, columns=whole_dataframe.columns
            )
            _cast_column_names_to_int(self.train_dataframe, inplace=True)

        # Store rest of samples, which will be further divided into testing and
        # validating sets
        self.test_dataframe = whole_dataframe.drop(self.train_dataframe.index)
        self.test_dataframe = self.test_dataframe.reset_index(drop=True)

        self.validation_dataframe = self.test_dataframe

        if benchmark_data:
            # ALTERNATIVE FROM BENCHMARK-----------------
            test_array = np.load(
                f"programs/benchmark/manip-ml-master/datasets/house/{seed}_test_array.npy"
            )
            self.test_dataframe = pd.DataFrame(
                test_array, columns=whole_dataframe.columns
            )
            _cast_column_names_to_int(self.test_dataframe, inplace=True)

        self.train_dataframe.reset_index(drop=True, inplace=True)

        poison_rate = config["poison_rate"] / 100
        self.poison_dataframe = self.train_dataframe.sample(
            frac=poison_rate, random_state=seed
        ).reset_index(drop=True)

        if benchmark_data:
            validation_array = np.load(
                f"programs/benchmark/manip-ml-master/datasets/house/{seed}_validation_array.npy"
            )
            self.validation_dataframe = pd.DataFrame(
                validation_array, columns=whole_dataframe.columns
            )
            _cast_column_names_to_int(self.validation_dataframe, inplace=True)

        if benchmark_data:
            # ALTERNATIVE FROM BENCHMARK-----------------
            poison_array = np.load(
                f"programs/benchmark/manip-ml-master/datasets/house/{seed}_{int(config['poison_rate']/100 * 300)}_poison_array.npy"
            )
            self.poison_dataframe = pd.DataFrame(
                poison_array, columns=whole_dataframe.columns
            )
            _cast_column_names_to_int(self.poison_dataframe, inplace=True)

        self.poison_dataframe["target"] = 1 - self.poison_dataframe[
            "target"
        ].round(0)

        self.no_numfeatures = len(
            get_numerical_feature_column_names(self.train_dataframe)
        )
        self.no_catfeatures = len(
            get_categorical_feature_column_names(self.train_dataframe)
        )
        self.no_chosen_numerical_features = config[
            "categorical_attack_no_nfeatures"
        ]
        self.no_chosen_categorical_features = config[
            "categorical_attack_no_cfeatures"
        ]

        # These attributes allows us to access the categorical features easily.
        # For example:
        # instance_data.cat_poison[sample_index, categorical_feature]
        # returns the category as an integer.
        # One can modify the category too by assigning an integer
        # instance_data.cat_poison[sample_index, categorical_feature] = new_cat
        # Similarly, one can get/set category using one-hot encoding
        # via cat_poison_one_hot.
        self.num_poison = NumFeatureAccessor(self.poison_dataframe)
        self.cat_poison_one_hot = CatFeatureAccessor(
            self.poison_dataframe, one_hot=True
        )
        self.cat_poison = CatFeatureAccessor(
            self.poison_dataframe, one_hot=False
        )

    def copy(self):
        """Return a deepcopy of self"""
        return copy.deepcopy(self)

    def get_numerical_columns(self):
        return [
            name
            for name in self.train_dataframe.columns
            if ":" not in name and name != "target"
        ]

    def get_categorical_columns(self):
        return [name for name in self.poison_dataframe.columns if ":" in name]

    def get_num_x_train_dataframe(self, wide=False):
        return get_numerical_features(df=self.train_dataframe, wide=wide)

    def get_cat_x_train_dataframe(self, wide=False):
        return get_categorical_features(df=self.train_dataframe, wide=wide)

    def get_y_train_dataframe(self):
        return get_targets(df=self.train_dataframe)

    def get_y_test_dataframe(self):
        return get_targets(df=self.test_dataframe)

    def get_num_x_poison_dataframe(self, wide=False):
        return get_numerical_features(df=self.poison_dataframe, wide=wide)

    def get_cat_x_poison_dataframe(self, wide=False):
        return get_categorical_features(df=self.poison_dataframe, wide=wide)

    def get_y_poison_dataframe(self):
        return get_targets(df=self.poison_dataframe)

    def get_num_x_test_dataframe(self, wide=False):
        return get_numerical_features(df=self.test_dataframe, wide=wide)

    def get_cat_x_test_dataframe(self, wide=False):
        return get_categorical_features(df=self.test_dataframe, wide=wide)

    def get_y_test_dataframe(self):
        return get_targets(df=self.test_dataframe)

    def get_num_x_validation_dataframe(self, wide=False):
        return get_numerical_features(df=self.validation_dataframe, wide=wide)

    def get_cat_x_validation_dataframe(self, wide=False):
        return get_categorical_features(
            df=self.validation_dataframe, wide=wide
        )

    def get_y_validation_dataframe(self):
        return get_targets(df=self.validation_dataframe)

    @property
    def no_samples(self):
        import warnings

        warnings.warn(
            "no_samples is deprecated. Use no_train_samples", stacklevel=2
        )
        return self.no_train_samples

    @property
    def no_psamples(self):
        import warnings

        warnings.warn(
            "no_psamples is deprecated. Use no_poison_samples", stacklevel=2
        )
        return self.no_poison_samples

    @property
    def no_train_samples(self):
        return len(self.train_dataframe)

    @property
    def no_poison_samples(self):
        return len(self.poison_dataframe)

    @property
    def numerical_feature_names(self):
        return get_numerical_feature_column_names(self.train_dataframe)

    @property
    def categorical_feature_category_tuples(self):
        return get_categorical_feature_category_tuples(self.train_dataframe)

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
    def chosen_numerical_feature_names(self):
        # TODO how do I add config here
        return get_chosen_numerical_feature_names(
            self.train_dataframe, self.no_chosen_numerical_features
        )

    @property
    def chosen_categorical_feature_names(self):
        # TODO how do I add config here
        return get_chosen_categorical_feature_names(
            self.train_dataframe, self.no_chosen_categorical_features
        )

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
            raise NotImplementedError

    def update_categorical_features(self, df):
        if isinstance(df, pd.Series):
            for index, row in df.items():
                # TODO Avoid making string as a column name.
                self.poison_dataframe.loc[
                    index[0], f"{index[1]}:{index[2]}"
                ] = row
        else:
            raise NotImplementedError


def _is_0_based_index(x):
    return np.all(x == np.arange(len(x)))


class NumFeatureAccessor:
    """Utility to facilitate access to numerical features

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "0":      [ 0,  1,  2],
    ...     "1":      [ 3,  4,  5],
    ...     "0:0":    [ 1,  0,  1],
    ...     "0:1":    [ 0,  1,  0],
    ...     "1:0":    [ 0,  0,  0],
    ...     "1:1":    [ 0,  1,  0],
    ...     "1:2":    [ 1,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> print(df)
       0  1  0:0  0:1  1:0  1:1  1:2  target
    0  0  3    1    0    0    0    1       6
    1  1  4    0    1    0    1    0       7
    2  2  5    1    0    0    0    1       8

    >>> num = NumFeatureAccessor(df)
    >>> print(num[0, 1])  # Get inumerical feature 1 of the first data.
    3
    >>> num[0, 1] = -2  # Update the feature
    >>> print(df)
       0  1  0:0  0:1  1:0  1:1  1:2  target
    0  0 -2    1    0    0    0    1       6
    1  1  4    0    1    0    1    0       7
    2  2  5    1    0    0    0    1       8
    """

    def __init__(self, df):
        self.df = df

        if not _is_0_based_index(df.index):
            raise ValueError("expected a dataframe with 0-based index")

        # This maps integer `categorical_feature`
        # to a list of the indices of the columns corresponding to the given
        # category.
        self.column_indices_from_feature = dict()

        def as_num_feature(x):
            try:
                return int(x)
            except ValueError:
                return None

        for column_index, column in enumerate(df.columns):
            num_feature = as_num_feature(column)
            if num_feature is None:
                continue
            self.column_indices_from_feature.setdefault(num_feature, [])
            self.column_indices_from_feature[num_feature].append(column_index)

    def __getitem__(self, key):
        column = self.column_indices_from_feature[key[1]]
        return self.df.iloc[key[0], column].item()

    def __setitem__(self, key, value):
        column = self.column_indices_from_feature[key[1]]
        self.df.iloc[key[0], column] = value


class CatFeatureAccessor:
    """Utility to facilitate access to categorical features

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "0":      [ 0,  1,  2],
    ...     "1":      [ 3,  4,  5],
    ...     "0:0":    [ 1,  0,  1],
    ...     "0:1":    [ 0,  1,  0],
    ...     "1:0":    [ 0,  0,  0],
    ...     "1:1":    [ 0,  1,  0],
    ...     "1:2":    [ 1,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })
    >>> print(df)
       0  1  0:0  0:1  1:0  1:1  1:2  target
    0  0  3    1    0    0    0    1       6
    1  1  4    0    1    0    1    0       7
    2  2  5    1    0    0    0    1       8

    >>> cat = CatFeatureAccessor(df, one_hot=True)
    >>> print(cat[0, 1])  # Get categorical feature 1 of the first data.
    [0 0 1]
    >>> cat[0, 1] = [1, 0, 0]  # Update the category in one-hot encoding
    >>> print(df)
       0  1  0:0  0:1  1:0  1:1  1:2  target
    0  0  3    1    0    1    0    0       6
    1  1  4    0    1    0    1    0       7
    2  2  5    1    0    0    0    1       8

    >>> cat = CatFeatureAccessor(df, one_hot=False)
    >>> print(cat[1, 0])  # Get categorical feature 1 of the first data.
    1
    >>> cat[1, 0] = 0  # Update the category in one-hot encoding
    >>> print(df)
       0  1  0:0  0:1  1:0  1:1  1:2  target
    0  0  3    1    0    1    0    0       6
    1  1  4    1    0    0    1    0       7
    2  2  5    1    0    0    0    1       8
    """

    def __init__(self, df, one_hot):
        self.df = df
        self.one_hot = one_hot

        if not _is_0_based_index(df.index):
            raise ValueError("expected a dataframe with 0-based index")

        # This maps integer `categorical_feature`
        # to a list of the indices of the columns corresponding to the given
        # category.
        self.column_indices_from_feature = dict()
        categories_from_categorical_feature = dict()

        for column_index, column in enumerate(df.columns):
            if not isinstance(column, str) or ":" not in column:
                continue
            # column is something like '2:5'
            categorical_feature, category = map(int, column.split(":"))
            self.column_indices_from_feature.setdefault(
                categorical_feature, []
            )
            self.column_indices_from_feature[categorical_feature].append(
                column_index
            )
            categories_from_categorical_feature.setdefault(
                categorical_feature, []
            )
            categories_from_categorical_feature[categorical_feature].append(
                category
            )

        for tpl in categories_from_categorical_feature.items():
            categorical_feature = tpl[0]
            categories = tpl[1]
            if not _is_0_based_index(categories):
                raise ValueError(
                    f"categorical feature {categorical_feature} has non-0-based categories {categories}"
                )

    def __getitem__(self, key):
        column = self.column_indices_from_feature[key[1]]
        one_hot = self.df.iloc[key[0], column].to_numpy()
        if self.one_hot:
            return one_hot
        return np.argmax(one_hot)

    def __setitem__(self, key, value):
        column = self.column_indices_from_feature[key[1]]
        if self.one_hot:
            self.df.iloc[key[0], column] = value
        else:
            self.df.iloc[key[0], column] = 0
            self.df.iloc[key[0], column[value]] = 1


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


def get_categorical_feature_category_tuples(df):
    """Extract the column names of categorical features as tuple of ints

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
    >>> get_categorical_feature_category_tuples(df)
    [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)]

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    names : list[str]
    """
    return [
        tuple(map(int, x.split(":")))
        for x in get_categorical_feature_column_names(df)
    ]


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
        set(
            [
                int(x.split(":")[0])
                for x in get_categorical_feature_column_names(df)
            ]
        )
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


def get_numerical_features(df, wide=False):
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
    >>> get_numerical_features(df)
    sample  feature
    0       1          0
            2          3
    1       1          1
            2          4
    2       1          2
            2          5
    Name: 0, dtype: int64
    >>> get_numerical_features(df, wide=True)
       1  2
    0  0  3
    1  1  4
    2  2  5

    Parameters
    ----------
    df : pandas.DataFrame
    wide : bool, default False

    Returns
    -------
    res : pandas.DataFrame
    """
    df = _cast_column_names_to_int(df)
    df = df[get_numerical_feature_column_names(df)]
    if wide:
        return df
    df = make_vertical_numerical_dataframe(df)
    return df


def get_categorical_features(df, wide=False):
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
    >>> get_categorical_features(df)
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

    >>> get_categorical_features(df, wide=True)
            1:1  1:2  2:1  2:2  2:3
    sample
    0         1    0    0    0    1
    1         0    1    0    1    0
    2         1    0    0    0    1

    Parameters
    ----------
    df : pandas.DataFrame
    wide : bool, default False

    Returns
    -------
    res : pandas.DataFrame
    """
    df = df[get_categorical_feature_column_names(df)]
    if wide:
        return df
    df = make_vertical_categorical_dataframe(df)
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


def make_vertical_numerical_dataframe(df):
    """Make dataframe vertical

    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ... })

    >>> make_vertical_numerical_dataframe(df)
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

    Returns
    -------
    res : pandas.DataFrame
    """
    df = df.unstack()
    df = df.reset_index()
    df = df.rename(columns={"level_0": "feature", "level_1": "sample"})
    df["feature"] = df["feature"].astype(int)
    df["sample"] = df["sample"].astype(int)
    df = df.set_index(["sample", "feature"])[0]
    df = df.sort_index()
    return df


def make_horizontal_numerical_dataframe(df):
    """Make dataframe horizontal

    >>> df = pd.DataFrame({
    ...     "1":      [ 0,  1,  2],
    ...     "2":      [ 3,  4,  5],
    ... })

    >>> pd.DataFrame({
    ...     "feature": [1, 2, 1, 2, 1, 2],
    ...     "sample": ["0", "0", "1", "1", "2", "2"],
    ... })
           feature sample
    0        1      0
    1        2      0
    2        1      1
    3        2      1
    4        1      2
    5        2      2

    >>> make_vertical_numerical_dataframe(df)
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

    Returns
    -------
    res : pandas.DataFrame
    """
    raise NotImplementedError


def make_vertical_categorical_dataframe(df):
    """Make dataframe vertical

    >>> df = pd.DataFrame({
    ...     "1:1":    [ 1,  0,  1],
    ...     "1:2":    [ 0,  1,  0],
    ...     "2:1":    [ 0,  0,  0],
    ...     "2:2":    [ 0,  1,  0],
    ...     "2:3":    [ 1,  0,  1],
    ...     "target": [ 6,  7,  8],
    ... })

    >>> get_categorical_features(df)
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

    Returns
    -------
    res : pandas.DataFrame
    """
    # The first index makes le
    df.index.name = "sample"
    df = df.unstack()
    df = df.reset_index()
    df[["feature", "category"]] = (
        df.iloc[:, 0].str.split(":", expand=True).astype(int)
    )
    df["sample"] = df["sample"].astype(int)
    df["feature"] = df["feature"].astype(int)
    df["category"] = df["category"].astype(int)
    df = df.drop(columns=df.columns[0])
    df = df.set_index(keys=["sample", "feature", "category"]).iloc[:, 0]
    df = df.sort_index()
    return df


# def get_chosen_numerical_feature_names(df, no_chosen_numerical_features):
#     """Extract the column names of chosen numerical features

#     Examples #TODO how do I add more input
#     --------
#     >>> df = pd.DataFrame({
#     ...     "1":      [ 0,  1,  2],
#     ...     "2":      [ 3,  4,  5],
#     ...     "1:1":    [ 1,  0,  0],
#     ...     "1:2":    [ 0,  1,  0],
#     ...     "2:1":    [ 0,  0,  1],
#     ...     "2:2":    [ 0,  1,  0],
#     ...     "2:3":    [ 0,  0,  1],
#     ...     "target": [ 6,  7,  8],
#     ... })
#     >>> get_numerical_feature_column_names(df)
#     [1, 2]

#     Parameters
#     ----------
#     df : pandas.DataFrame

#     Returns
#     -------
#     names : list[int]
#     """
#     chosen_numerical = choosing_features.LASSOdataframe(df).get_features_lists(
#         no_chosen_numerical_features, 0
#     )[0]

#     return chosen_numerical


# def get_chosen_categorical_feature_names(df, no_chosen_categorical_features):
#     """Extract the column names of chosen numerical features

#     Examples #TODO how do I add more input
#     --------
#     >>> df = pd.DataFrame({
#     ...     "1":      [ 0,  1,  2],
#     ...     "2":      [ 3,  4,  5],
#     ...     "1:1":    [ 1,  0,  0],
#     ...     "1:2":    [ 0,  1,  0],
#     ...     "2:1":    [ 0,  0,  1],
#     ...     "2:2":    [ 0,  1,  0],
#     ...     "2:3":    [ 0,  0,  1],
#     ...     "target": [ 6,  7,  8],
#     ... })
#     >>> get_numerical_feature_column_names(df)
#     [1, 2]

#     Parameters
#     ----------
#     df : pandas.DataFrame

#     Returns
#     -------
#     names : list[int]
#     """
#     chosen_categorical = choosing_features.LASSOdataframe(
#         df
#     ).get_features_lists(0, no_chosen_categorical_features)[1]

#     return chosen_categorical


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
