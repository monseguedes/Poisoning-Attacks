import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, HTMLWriter

dataframes_attacks = []
percentage_attacks = []
dataframes_numerical_weights = []
dataframes_categorical_weights = []

dataset_name = "5num5cat"

plt.rcParams["figure.figsize"] = [10, 5]


def get_dataframes_attacks(config):
    dataframes_attacks = []
    no_iterations = len(
        [
            name
            for name in os.listdir(f"programs/minlp/attacks/{config['dataset_name']}")
            if os.path.isfile(name) and "poison_dataframe" in name
        ]
    )
    for i in range(no_iterations):
        df = pd.read_csv(
            f"programs/minlp/attacks/{config['dataset_name']}/poison_dataframe{i}.csv"
        )
        dataframes_attacks.append(df)
    return dataframes_attacks


def get_poison_percentage_attacks(config):
    percentage_attacks = []
    no_iterations = len(
        [
            name
            for name in os.listdir(f"programs/minlp/attacks/{config['dataset_name']}")
            if os.path.isfile(name) and "poison_dataframe" in name
        ]
    )
    for i in range(no_iterations):
        df = pd.read_csv(
            f"programs/minlp/attacks/{dataset_name}/poison_dataframe{i}.csv"
        )
        df = df[df.columns[1:-1]]
        df = df.round(0)
        df.loc["sum"] = df.sum()
        df.loc["percentage"] = df.loc["sum"] / df.shape[0] * 100
        percentage_attacks.append(df)
    return percentage_attacks


def get_training_percentage_attacks(config):
    training_percentage = pd.read_csv(
        f"programs/minlp/attacks/{config['dataset_name']}/training_data.csv"
    )
    training_percentage = training_percentage[training_percentage.columns[1:-1]]
    training_percentage = training_percentage.round(0)
    training_percentage.loc["sum"] = training_percentage.sum()
    training_percentage.loc["percentage"] = (
        training_percentage.loc["sum"] / training_percentage.shape[0] * 100
    )
    return training_percentage


def get_dataframes_numerical_weights(config):
    dataframes_numerical_weights = []
    no_iterations = len(
        [
            name
            for name in os.listdir(f"programs/minlp/attacks/{config['dataset_name']}")
            if os.path.isfile(name) and "numerical_weights" in name
        ]
    )
    for i in range(no_iterations):
        df = pd.read_csv(
            f"programs/minlp/attacks/{dataset_name}/numerical_weights{i}.csv",
            header=None,
        )
        df = df[df.columns[1:]]
        df = df.T
        df = df[df.columns[1:]]
        dataframes_numerical_weights.append(df)
    return dataframes_numerical_weights


def get_initial_numerical_weights(config):
    initial_numerical_weights = pd.read_csv(
        f"programs/minlp/attacks/{config['dataset_name']}/initial_numerical_weights.csv",
        header=None,
    )
    initial_numerical_weights = initial_numerical_weights[
        initial_numerical_weights.columns[1:]
    ]
    initial_numerical_weights = initial_numerical_weights.T
    initial_numerical_weights = initial_numerical_weights[
        initial_numerical_weights.columns[1:]
    ]
    return initial_numerical_weights


def get_dataframes_categorical_weights(config):
    dataframes_categorical_weights = []
    categorical_columns = list(percentage_attacks[0].columns)
    categorical_columns = categorical_columns[
        len(dataframes_numerical_weights[0].columns) :
    ]
    no_iterations = len(
        [
            name
            for name in os.listdir(f"programs/minlp/attacks/{config['dataset_name']}")
            if os.path.isfile(name) and "categorical_weights" in name
        ]
    )
    for i in range(1, no_iterations):
        df = pd.read_csv(
            f"programs/minlp/attacks/{dataset_name}/categorical_weights{i}.csv"
        )
        df = df[df.columns[2:]]
        df = df.T
        df.columns = categorical_columns
        dataframes_categorical_weights.append(df)
    return dataframes_categorical_weights


def get_initial_categorical_weights(config):
    initial_categorical_weights = pd.read_csv(
        f"programs/minlp/attacks/{config['dataset_name']}/initial_categorical_weights.csv"
    )
    initial_categorical_weights = initial_categorical_weights[
        initial_categorical_weights.columns[2:]
    ]
    initial_categorical_weights = initial_categorical_weights.T
    return initial_categorical_weights


def plot_attacks(dataframes_attacks):
    fig, ax = plt.subplots()

    def update(frame):
        df = dataframes_attacks[frame]
        ax.clear()
        for i in range(len(df)):
            ax.scatter(list(df.columns), list(df.iloc[i]))
        ax.set_ylim(-0.1, 1.1)  # set the y-axis limit to a fixed range
        ax.set_title(f"Dataframe {frame}")

    anim = FuncAnimation(fig, update, frames=len(dataframes_attacks), interval=300)
    writer = HTMLWriter(fps=2)
    anim.save("attacks.html", writer=writer)
    plt.show()
    plt.close()


def plot_percentage_attacks(percentage_attacks, training_percentage):
    fig, ax = plt.subplots()

    def update(frame):
        df = percentage_attacks[frame]
        initial = training_percentage
        ax.clear()
        for i in range(len(df)):
            ax.bar(list(df.columns), list(df.iloc[-1]))
        for i in range(len(df)):
            ax.scatter(
                list(training_percentage.columns), list(training_percentage.iloc[-1])
            )
        ax.set_ylim(-0.005, 105)  # set the y-axis limit to a fixed range
        plt.xticks(fontsize=5, rotation=45)
        ax.set_title(f"Percentage of 1s at iteration {frame}")

    anim = FuncAnimation(fig, update, frames=len(percentage_attacks), interval=500)
    writer = HTMLWriter(fps=3)
    anim.save("percentage_ones.html", writer=writer)
    plt.show()
    plt.close()


def plot_numerical_weights(dataframes_numerical_weights, initial_numerical_weights):
    fig, ax = plt.subplots()

    def update(frame):
        df = dataframes_numerical_weights[frame]
        initial = initial_numerical_weights
        ax.clear()
        for i in range(len(df)):
            ax.bar(list(df.columns), list(df.iloc[i]))
        for i in range(len(df)):
            ax.scatter(
                list(initial_numerical_weights.columns),
                list(initial_numerical_weights.iloc[i]),
            )
        ax.set_ylim(
            -0.005, max(list(initial_numerical_weights.iloc[i])) + 0.001
        )  # set the y-axis limit to a fixed range
        ax.set_title(f"Numerical weights of iteration {frame}")

    anim = FuncAnimation(
        fig, update, frames=len(dataframes_numerical_weights), interval=500
    )
    writer = HTMLWriter(fps=2)
    anim.save("numerical_weights.html", writer=writer)
    plt.show()
    plt.close()


def plot_categorical_weights(
    dataframes_categorical_weights, initial_categorical_weights
):
    fig, ax = plt.subplots()

    def update(frame):
        df = dataframes_categorical_weights[frame]
        initial = initial_categorical_weights
        ax.clear()
        for i in range(len(df)):
            ax.bar(list(df.columns), list(df.iloc[i]))
        for i in range(len(df)):
            ax.scatter(
                list(initial_categorical_weights.columns),
                list(initial_categorical_weights.iloc[i]),
            )
        ax.set_ylim(
            -0.015, max(list(initial_categorical_weights.iloc[i])) + 0.004
        )  # set the y-axis limit to a fixed range
        plt.xticks(fontsize=5, rotation=45)
        ax.set_title(f"Categorical weights of iteration {frame}")

    anim = FuncAnimation(
        fig, update, frames=len(dataframes_numerical_weights), interval=500
    )
    writer = HTMLWriter(fps=2)
    anim.save("categorical_weights.html", writer=writer)
    plt.show()
    plt.close()
