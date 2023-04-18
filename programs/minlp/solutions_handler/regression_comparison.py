"""
@author: Monse Guedes Ayala
@project: Poisoning Attacks Paper

Functions necessary to evaluate the performance of poisoning attacks. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import pyomo.environ as pyo

import os

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import model.instance_class
import model.model_class
import model.pyomo_instance_class

sns.set_style("whitegrid")

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


class ComparisonModel:
    """
    This class builds all necessary objects to compare models.
    """

    def __init__(self, model_parameters):
        """
        Given some data class and some model class, build a model to
        then compare to other model.
        bilevel_instance_data: the data class for the bilevel model
        bilevel_model: the bilevel model class
        ridge_instance_data: the data class for the ridge nopoisoned model
        ridge_model: the nonpoisoned ridge model classs
        """

        self.datatype = model_parameters["datatype"]
        self.folder = "_".join(
            [
                model_parameters["dataset_name"],
                str(model_parameters["poison_rate"]),
                str(model_parameters["training_samples"]),
                str(model_parameters["no_psubsets"]),
                str(model_parameters["seed"]),
            ]
        )

        if not os.path.exists(os.path.join("plots", self.folder)):
            os.mkdir(os.path.join("plots", self.folder))

        if not os.path.exists(os.path.join("results/scores", self.folder)):
            os.mkdir(os.path.join("results/scores", self.folder))

    def compare_everything(
        self,
        bilevel_instance,
        bilevel_model,
        ridge_instance,
        ridge_model,
        benchmark_instance,
        benchmark_model,
    ):
        """
        Compares all three models
        """

        self.make_poisoned_predictions(
            bilevel_instance=bilevel_instance, bilevel_model=bilevel_model
        )
        self.make_non_poisoned_predictions(
            ridge_instance=ridge_instance, ridge_model=ridge_model
        )
        self.make_benchmark_predictions(
            benchmark_instance=benchmark_instance,
            benchmark_model=benchmark_model,
        )
        self.plot_actual_vs_pred("bilevel")
        self.plot_actual_vs_pred("benchmark")
        self.plot_actual_vs_predicted_all()
        self.store_comparison_metrics()

    def make_poisoned_predictions(
        self,
        bilevel_model: model.model_class.PoisonAttackModel,
        bilevel_instance: model.instance_class.InstanceData,
    ):
        """
        Take the regression coefficents given by solving the bilevel model
        and use them to make predictions on training dataset.
        """
        self.bilevel_model = bilevel_model
        self.bilevel_instance = bilevel_instance

        if self.datatype == "train":
            self.y = list(self.bilevel_model.y_train)
            self.bilevel_dataframe = self.bilevel_instance.x_train_dataframe.copy(
                deep=True
            )

        elif self.datatype == "test":
            self.y = list(self.bilevel_instance.test_y)
            self.bilevel_dataframe = self.bilevel_instance.test_dataframe.copy(
                deep=True
            )

        # Define vector of size rows of bilevel_dataframe, and with bias in all terms
        self.pred_bilevel_y_train = np.repeat(
            self.bilevel_model.bias.X, len(self.bilevel_dataframe)
        )

        # Take columns one by one, convert them to vector, and multiply them by the corresponding weights
        for column in self.bilevel_dataframe.columns:
            if ":" not in column:
                column_index = int(column)
                self.pred_bilevel_y_train += (
                    self.bilevel_dataframe[column]
                    * self.bilevel_model.weights_num[column_index].X
                )
            else:
                column_index = [int(index) for index in column.split(":")]
                self.pred_bilevel_y_train += (
                    self.bilevel_dataframe[column]
                    * self.bilevel_model.weights_cat[
                        (column_index[0], column_index[1])
                    ].X
                )

        self.bilevel_dataframe["actual_y_train"] = self.y
        self.bilevel_dataframe["pred_bilevel_y_train"] = self.pred_bilevel_y_train

        return self.pred_bilevel_y_train

    def make_benchmark_predictions(
        self,
        benchmark_model: model.model_class.BenchmarkPoisonAttackModel,
        benchmark_instance: model.pyomo_instance_class.InstanceData,
    ):
        """
        Take the regression coefficents given by solving the model
        and use them to make predictions.
        """

        self.benchmark_model = benchmark_model
        self.benchmark_intance = benchmark_instance

        if self.datatype == "train":
            self.y = list(self.benchmark_model.y_train.values())
            self.benchmark_dataframe = self.benchmark_intance.x_train_dataframe.copy(
                deep=True
            )

        elif self.datatype == "test":
            self.y = list(self.benchmark_intance.test_y)
            self.benchmark_dataframe = self.benchmark_intance.test_dataframe.copy(
                deep=True
            )

        # Define vector of size rows of data_dataframe, and with bias in all terms.
        self.pred_benchmark_y_train = np.repeat(
            pyo.value(self.benchmark_model.bias), len(self.benchmark_dataframe)
        )

        # Take columns one by one, convert them to vector, and multiply them by the corresponding weights
        for column in self.benchmark_dataframe.columns:
            if ":" not in column:
                column_index = int(column)
                self.pred_benchmark_y_train += (
                    self.benchmark_dataframe[column]
                    * self.benchmark_model.weights_num[column_index]._value
                )
            else:
                column_index = [int(index) for index in column.split(":")]
                self.pred_benchmark_y_train += (
                    self.benchmark_dataframe[column]
                    * self.benchmark_model.weights_cat[
                        (column_index[0], column_index[1])
                    ]._value
                )

        self.benchmark_dataframe["actual_y_train"] = self.y
        self.benchmark_dataframe["pred_benchmark_y_train"] = self.pred_benchmark_y_train

        return self.pred_benchmark_y_train

    def make_non_poisoned_predictions(
        self,
        ridge_instance: model.instance_class.InstanceData,
        ridge_model: model.model_class.RegressionModel,
    ):
        """
        Take the regression coefficents given by solving the nonpoisoned model
        and use them to make predictions on training dataset.
        """
        self.ridge_instance = ridge_instance
        self.ridge_model = ridge_model

        if self.datatype == "train":
            self.y = list(self.ridge_model.y_train.values())
            self.ridge_dataframe = self.ridge_instance.ridge_x_train_dataframe.copy(
                deep=True
            ).unstack()

        elif self.datatype == "test":
            self.y = list(self.ridge_instance.test_y)
            self.ridge_dataframe = (
                self.ridge_instance.test_ridge_x_train_dataframe.copy(
                    deep=True
                ).unstack()
            )

        # Define vector of size rows of data_dataframe, and with bias in all terms
        self.pred_ridge_y_train = np.repeat(
            self.ridge_model.bias.X, len(self.ridge_dataframe)
        )

        # Take columns one by one, convert them to vector, and multiply them by the corresponding weights
        for column in self.ridge_dataframe.columns:
            column_index = int(column)
            self.pred_ridge_y_train += (
                self.ridge_dataframe[column] * self.ridge_model.weights[column_index].X
            )

        self.ridge_dataframe["actual_y_train"] = self.y
        self.ridge_dataframe["pred_ridge_y_train"] = self.pred_ridge_y_train

        return self.pred_ridge_y_train

    def plot_actual_vs_pred(self, model_name):
        """
        Take the predictions of either model
        and plot them vs actual.
        """

        if model_name == "bilevel":
            dataframe = self.bilevel_dataframe
            y_name = "pred_bilevel_y_train"
            color = "red"
            file_name = "_".join(
                [
                    "/actual_vs_predicted",
                    self.datatype,
                    str(self.bilevel_model.no_numfeatures),
                    str(self.bilevel_model.no_catfeatures),
                    str(len(self.bilevel_dataframe)),
                    str(int(self.bilevel_instance.poison_rate * 100)),
                    str(self.bilevel_instance.seed),
                ]
            )

        elif model_name == "benchmark":
            dataframe = self.benchmark_dataframe
            y_name = "pred_benchmark_y_train"
            color = "green"
            file_name = "_".join(
                [
                    "/benchmark_actual_vs_predicted",
                    self.datatype,
                    str(self.benchmark_model.no_numfeatures),
                    str(self.benchmark_model.no_catfeatures),
                    str(len(self.benchmark_dataframe)),
                    str(int(self.benchmark_intance.poison_rate * 100)),
                    str(self.benchmark_intance.seed),
                ]
            )

        figure = sns.scatterplot(
            data=dataframe,
            x="actual_y_train",
            y=y_name,
            label="Poisoned",
            color=color,
        )
        sns.scatterplot(
            data=self.ridge_dataframe,
            x="actual_y_train",
            y="pred_ridge_y_train",
            label="Non-poisoned",
            color="lightskyblue",
        )
        figure.set_aspect("equal", adjustable="box")
        max_value = max(
            [
                max(dataframe["actual_y_train"]),
                max(dataframe[y_name]),
                max(self.ridge_dataframe["pred_ridge_y_train"]),
            ]
        )
        plt.xlim([-0.05, max_value + 0.05])
        plt.ylim([-0.05, max_value + 0.05])

        if self.datatype == "train":
            plt.title("Actual vs Predicted for Training Data", fontsize=20)
        elif self.datatype == "test":
            plt.title("Actual vs Predicted for Test Data", fontsize=20)

        plt.xlabel("Actual", fontsize=14)
        plt.ylabel("Predicted", fontsize=14)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.1), fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.savefig(
            "plots/" + self.folder + file_name + ".pdf",
            transparent=True,
            bbox_inches="tight",
        )

    def plot_actual_vs_predicted_all(self):
        """
        Take the predictions of both models
        and plot them vs actual.
        """

        figure = sns.scatterplot(
            data=self.bilevel_dataframe,
            x="actual_y_train",
            y="pred_bilevel_y_train",
            label="Poisoned",
        )
        sns.scatterplot(
            data=self.ridge_dataframe,
            x="actual_y_train",
            y="pred_ridge_y_train",
            label="Non-poisoned",
        )
        sns.scatterplot(
            data=self.benchmark_dataframe,
            x="actual_y_train",
            y="pred_benchmark_y_train",
            label="Benchmark",
        )

        figure.set_aspect("equal", adjustable="box")
        max_value = max(
            [
                max(self.bilevel_dataframe["actual_y_train"]),
                max(self.bilevel_dataframe["pred_bilevel_y_train"]),
                max(self.ridge_dataframe["pred_ridge_y_train"]),
            ]
        )
        plt.xlim([-0.05, max_value + 0.05])
        plt.ylim([-0.05, max_value + 0.05])
        plt.title("Actual vs Predicted for Training Data")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.savefig(
            "plots/"
            + self.folder
            + "/all_actual_vs_predicted"
            + "_"
            + self.datatype
            + "_"
            + str(self.bilevel_model.no_numfeatures)
            + "_"
            + str(self.bilevel_model.no_catfeatures)
            + "_"
            + str(len(self.bilevel_dataframe))
            + "_"
            + str(int(self.bilevel_instance.poison_rate * 100))
            + "_"
            + str(self.bilevel_instance.seed)
            + ".png"
        )
        plt.show()

    def store_comparison_metrics(self):
        """
        Finds and stores main regression metrics for the poisoned
        and nonpoisoned models.
        """

        # Create dataframe with metrics
        self.metrics_dataframe = pd.DataFrame(
            {
                "metric": ["MSE", "RMSE", "MAE"],
                "nonpoisoned": [
                    mean_squared_error(self.y, self.pred_ridge_y_train, squared=False),
                    mean_squared_error(self.y, self.pred_ridge_y_train, squared=True),
                    mean_absolute_error(self.y, self.pred_ridge_y_train),
                ],
                "benchmark": [
                    mean_squared_error(
                        self.y, self.pred_benchmark_y_train, squared=False
                    ),
                    mean_squared_error(
                        self.y, self.pred_benchmark_y_train, squared=True
                    ),
                    mean_absolute_error(self.y, self.pred_benchmark_y_train),
                ],
                "poisoned": [
                    mean_squared_error(
                        self.y, self.pred_bilevel_y_train, squared=False
                    ),
                    mean_squared_error(self.y, self.pred_bilevel_y_train, squared=True),
                    mean_absolute_error(self.y, self.pred_bilevel_y_train),
                ],
            }
        )

        # Last columns as increment between models
        self.metrics_dataframe["non-benchmark increase"] = (
            (
                self.metrics_dataframe["benchmark"]
                - self.metrics_dataframe["nonpoisoned"]
            )
            / self.metrics_dataframe["nonpoisoned"]
            * 100
        )
        self.metrics_dataframe["non-MINLP increase"] = (
            (self.metrics_dataframe["poisoned"] - self.metrics_dataframe["nonpoisoned"])
            / self.metrics_dataframe["nonpoisoned"]
            * 100
        )
        self.metrics_dataframe["benchmark-MINLP increase"] = (
            (self.metrics_dataframe["poisoned"] - self.metrics_dataframe["benchmark"])
            / self.metrics_dataframe["benchmark"]
            * 100
        )

        self.metrics_dataframe.to_csv(
            "solutions/"
            + self.folder
            + "/all_actual_vs_predicted"
            + "_"
            + self.datatype
            + "_"
            + str(self.bilevel_model.no_numfeatures)
            + "_"
            + str(self.bilevel_model.no_catfeatures)
            + "_"
            + str(len(self.bilevel_dataframe))
            + "_"
            + str(int(self.bilevel_instance.poison_rate * 100))
            + str(self.bilevel_instance.seed)
            + ".csv"
        )

        print(self.metrics_dataframe)
