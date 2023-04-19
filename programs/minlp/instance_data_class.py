"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

This script creates the class with all the data that is then given to the benckmark model.
"""

# TODO Improve consistency of naming.
# TODO Implement utitlity to convert wide <-> long format.
# TODO Implement update of categorical features.
# TODO Improve efficiency of the getter/setter functions.
# In particular, conversion between str <-> integer is very slow
# (e.g. creating string '1:2' from feature=1, category=2). Until the
# performance becomes an issue, we can leave as it is. But when necesarry
# we can think about how to improve it.

import copy
from os import path

import pandas as pd


class InstanceData:
    def __init__(self, config):
        """
        The initialization corresponds to the data for the first iteration.
        If there are no iterations (single attack strategy).

        dataset_name: 'pharm', or 'house'
        """
        self.regularization = config["regularization"]

        # Whole dataframe with features as columns and target column,
        # as in file (1,2,3..,1:1,1:2,...,target).
        dataset_directory = path.join("data", config["dataset_name"])
        whole_dataframe = pd.read_csv(
            path.join(dataset_directory, "data-binary.csv"), index_col=[0]
        )

        _cast_column_names_to_int(whole_dataframe, inplace=True)

        # Pick fixed number of trainig samples.
        # The indexes are not reset, but randomly shuffled
        training_samples = config["training_samples"]
        seed = config["seed"]
        self.train_dataframe = whole_dataframe.sample(
            frac=None, n=training_samples, random_state=seed
        )

        # Store rest of samples, which will be further divided into testing and
        # validating sets
        self.test_dataframe = whole_dataframe.drop(self.train_dataframe.index)
        self.test_dataframe = self.test_dataframe.reset_index(drop=True)

        self.train_dataframe.reset_index(drop=True, inplace=True)

        poison_rate = config["poison_rate"] / 100
        self.poison_dataframe = self.train_dataframe.sample(
            frac=poison_rate, random_state=seed
        ).reset_index(drop=True)

        # TODO Define attributes related to column information, and remove corresponding property.
        # You dont have to remove ones related to rows, such as number of training data, since
        # we may updage dataframe later.
        self.no_numfeatures = len(
            get_numerical_feature_column_names(self.train_dataframe)
        )
        self.no_catfeatures = len(
            get_categorical_feature_column_names(self.train_dataframe)
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

    def get_num_x_poison_dataframe(self, wide=False):
        return get_numerical_features(df=self.poison_dataframe, wide=wide)

    def get_cat_x_poison_dataframe(self, wide=False):
        return get_categorical_features(df=self.poison_dataframe, wide=wide)

    def get_y_poison_dataframe(self):
        return get_targets(df=self.poison_dataframe)

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

    def update_categorical_features(self, df):
        raise NotImplementedError


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
        tuple(map(int, x.split(":"))) for x in get_categorical_feature_column_names(df)
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
    df[["feature", "category"]] = df.iloc[:, 0].str.split(":", expand=True).astype(int)
    df["sample"] = df["sample"].astype(int)
    df["feature"] = df["feature"].astype(int)
    df["category"] = df["category"].astype(int)
    df = df.drop(columns=df.columns[0])
    df = df.set_index(keys=["sample", "feature", "category"]).iloc[:, 0]
    df = df.sort_index()
    return df


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
