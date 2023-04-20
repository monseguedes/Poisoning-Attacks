"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


def create_dataframe(name: str):
    """
    Create dataframe with dataset
    """

    file = "data-binary.csv"
    path = os.path.join("data", name, file)

    dataframe = pd.read_csv(path, index_col=0)

    categorical_columns = [
        name for name in dataframe.columns if ":" in name
    ]  # Names of categorical columns
    categorical_names = set(
        [name.split(":")[0] for name in categorical_columns]
    )  # Names of unique categorical features
    categories_dict = {
        cat_name: [
            category.split(":")[1]
            for category in categorical_columns
            if category.startswith(cat_name + ":")
        ]
        for cat_name in categorical_names
    }

    # Check that all have at least one 1.
    check = []
    for catfeature, categories in categories_dict.items():
        df = dataframe
        df_columns = [catfeature + ":" + category for category in categories]
        has_one = df[df_columns].eq(1).any(axis=1).all()
        check.append(has_one)
    print(np.all(check))

    return dataframe


class LASSOdataframe:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def get_features_lists(self, no_numerical: int, no_categorical: int):
        """
        Run all necessary functions to get list of most
        important features
        """
        self.no_numerical = no_numerical
        self.no_categorical = no_categorical
        alpha = 0.05
        self.format_data()
        (
            self.chosen_numerical,
            self.chosen_categorical,
        ) = self.get_num_cat_features(no_numerical, no_categorical, alpha)

        return self.chosen_numerical, self.chosen_categorical

    def format_data(self):
        """
        Convert data to format that sklearn can handle
        """
        # Get feature columns and convert them to array
        self.features_dataframe = self.dataframe.drop(columns=["target"])
        self.features_array = np.array(self.features_dataframe)
        self.name_features = self.features_dataframe

        # Get target column
        self.target_array = np.array(self.dataframe.target)

        return self.features_array, self.target_array

    def fit_lasso(self, alpha):
        """
        Fit LASSO model to data.
        """
        self.model = Lasso(alpha=alpha)
        self.model.fit(self.features_array, self.target_array)

        return self.model

    def get_used_features(self, alpha):
        """
        Get the names of the features that LASSO assigns weights
        different than 0.
        """
        self.fit_lasso(alpha)
        coeffs = self.model.coef_
        bool_coeff = [False if coef == 0 else True for coef in coeffs]
        self.used_features = [
            name for name, nonzero in zip(self.name_features, bool_coeff) if nonzero
        ]
        self.coeffs_used_features = {
            name: coeff
            for name in self.used_features
            for coeff in [coeff for coeff in coeffs if coeff != 0]
        }

        return self.coeffs_used_features

    def get_num_cat_features(
        self, no_numerical: int, no_categorical: int, alpha: float
    ):
        """
        Takes the n most important features for numerical and categorical.
        For categorical, the whole categorical feature is chosen.
        """

        original_alpha = alpha

        ### Get numerical features------------------------
        if no_numerical == "all":
            self.chosen_numerical = [
                column
                for column in self.features_dataframe.columns
                if ":" not in column
            ]
        elif no_numerical == 0:
            self.chosen_numerical = []
        else:
            numerical_features = {}
            # Make sure LASSO selects enough features
            while no_numerical > len(numerical_features):
                self.get_used_features(alpha)
                alpha -= 0.001
                numerical_features = {
                    int(key): abs(value)
                    for key, value in self.coeffs_used_features.items()
                    if ":" not in key
                }
            chosen_numerical = sorted(
                numerical_features, key=numerical_features.get, reverse=True
            )[:no_numerical]
            self.chosen_numerical = list(chosen_numerical)

        alpha = original_alpha

        ### Get categorical features-----------------------
        if no_categorical == "all":
            self.chosen_categorical = list(
                set(
                    [
                        column.split(":")[0]
                        for column in self.features_dataframe.columns
                        if ":" in column
                    ]
                )
            )
        else:
            max_dict = {}
            while no_categorical > len(max_dict):
                self.get_used_features(alpha)
                categorical_features = {
                    key: abs(value)
                    for key, value in self.coeffs_used_features.items()
                    if isinstance(key, str) and ":" in key
                }
                for key, value in categorical_features.items():
                    key_type = int(key.split(":")[0])
                    if key_type not in max_dict or value > max_dict[key_type]:
                        max_dict[key_type] = value
                alpha -= 0.001

            chosen_categorical = sorted(max_dict, key=max_dict.get, reverse=True)[
                :no_categorical
            ]
            self.chosen_categorical = list(chosen_categorical)

        return self.chosen_numerical, self.chosen_categorical

    def save_new_dataframe(self):
        """
        Filter dataframe and save it
        """
        numerical_dataframe = self.dataframe[
            [str(column) for column in self.chosen_numerical]
        ]
        new_numerical_cols = [str(i) for i in range(len(numerical_dataframe.columns))]
        numerical_dataframe = numerical_dataframe.rename(
            columns=dict(zip(numerical_dataframe.columns, new_numerical_cols))
        )

        if self.chosen_categorical != []:
            categorical_dataframe = pd.concat(
                [
                    self.dataframe.filter(regex=f"^{prefix}:")
                    for prefix in self.chosen_categorical
                ],
                axis=1,
            )
            new_categorical_columns_dict = {
                str(feature) + ":": str(i) + "_"
                for i, feature in enumerate(self.chosen_categorical)
            }
            columns = list(categorical_dataframe.columns)
            for i in range(len(columns)):
                for (
                    old_substring,
                    new_substring,
                ) in new_categorical_columns_dict.items():
                    columns[i] = re.sub("^" + old_substring, new_substring, columns[i])
            columns = [x.replace("_", ":") for x in columns]

            # Currently, the categories are 1-based. We are updating them to be 0-based.
            # For example, if columns are ["1:1", "1:2", "1:3", "2:1", "2:2"],
            # it ['1:0', '1:1', '1:2', '2:0', '2:1'].
            f = lambda x: [x[0], x[1] - 1]
            g = lambda x: f"{x[0]}:{x[1]}"
            lm = lambda f, lst: list(map(f, lst))
            columns = lm(g, lm(f, (lm(int, x.split(":")) for x in columns)))

            categorical_dataframe.columns = columns
            whole_dataframe = pd.concat(
                [
                    numerical_dataframe,
                    categorical_dataframe,
                    self.dataframe["target"],
                ],
                axis=1,
            )
        else:
            whole_dataframe = pd.concat(
                [numerical_dataframe, self.dataframe["target"]], axis=1
            )

        dataset_name = str(self.no_numerical) + "num" + str(self.no_categorical) + "cat"
        directory = os.path.join("data", dataset_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        whole_dataframe.to_csv(os.path.join(directory, "data-binary.csv"))


# dataframe = create_dataframe("house")
# model = LASSOdataframe(dataframe)
# model.get_features_lists(5, 5)
# model.save_new_dataframe()
