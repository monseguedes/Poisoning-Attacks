# -*- coding: utf-8 -*-

"""Gradient descent method with pytorch"""

import numpy as np
import torch

def main():
    """Run gradient descent method with pytorch"""
    n_train = 4
    n_poison = 2
    n_num_features = 2
    n_categories = [3, 2]
    np.random.seed(0)
    X_num_train = np.random.rand(n_train, n_num_features)
    X_num_poison = np.random.rand(n_poison, n_num_features)
    X_cat_train = np.array([[np.random.choice(x) for x in n_categories] for i in range(n_train)])
    X_cat_poison = np.array([[np.random.choice(x) for x in n_categories] for i in range(n_poison)])
    X_cat_train_one_hot = to_one_hot(X_cat_train, n_categories)
    X_cat_poison_one_hot = to_one_hot(X_cat_poison, n_categories)
    y_train = np.random.rand(n_train)
    y_poison = np.random.rand(n_poison)
    run(X_num_train, X_cat_train_one_hot, y_train, X_num_poison, X_cat_poison_one_hot, y_poison, optimize_catogorical_features=False)


def run(X_num_train, X_cat_train_one_hot, y_train, X_num_poison, X_cat_poison_one_hot, y_poison, optimize_catogorical_features=False):
    """Run gradient descent method with pytorch"""
    if optimize_catogorical_features:
        raise NotImplementedError
    regularisation_parameter = 1.0
    X_num_train = torch.tensor(X_num_train)
    X_cat_train_one_hot = torch.tensor(X_cat_train_one_hot)
    y_train = torch.tensor(y_train)
    X_num_poison = torch.tensor(X_num_poison, requires_grad=True)
    X_cat_poison_one_hot = torch.tensor(X_cat_poison_one_hot, requires_grad=optimize_catogorical_features)
    y_poison = torch.tensor(y_poison)

    parameters = [X_num_poison]
    if optimize_catogorical_features:
        parameters.append(X_cat_poison_one_hot)
    optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.0, maximize=True)

    X_train = torch.cat([X_num_train, X_cat_train_one_hot], axis=1)

    for i in range(50):
        X_poison = torch.cat([X_num_poison, X_cat_poison_one_hot], axis=1)
        X = torch.cat([X_train, X_poison], axis=0)
        y = torch.cat([y_train, y_poison])
        theta = torch.linalg.solve(X.T @ X + regularisation_parameter * torch.eye(X.shape[1]), X.T @ y)
        loss = torch.nn.functional.mse_loss(y_train, X_train @ theta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            print(f"{'mse':>9s}")
        print(f"{loss.item():9.4f}")

    return X_num_poison, X_cat_poison_one_hot, y_poison


def to_one_hot(X, n_categories):
    """Create a one hot encoding of categorical features

    Examples
    --------
    >>> X = [[1, 1], [2, 0], [2, 1]]
    >>> print(to_one_hot(X, n_categories=[3, 2]))
    [[0. 1. 0. 0. 1.]
     [0. 0. 1. 1. 0.]
     [0. 0. 1. 0. 1.]]
    >>> print(to_one_hot(X, n_categories=[4, 2]))
    [[0. 1. 0. 0. 0. 1.]
     [0. 0. 1. 0. 1. 0.]
     [0. 0. 1. 0. 0. 1.]]

    Parameters
    ----------
    X : (n_samples, n_categorical_features) array
    n_categories : (n_categorical_features,) array
        `n_categories[i]` is  the number of categories categorical feature
        `i` has.

    Returns
    -------
    X_one_hot : (n_samples, sum(n_categories)) array
    """
    X = np.asarray(X)
    n_samples, n_cat_features = X.shape
    X_one_hot = np.zeros((n_samples, np.sum(n_categories)))
    cat_one_hot_start = np.r_[0, np.cumsum(n_categories[:-1])]
    for sample in range(n_samples):
        for cat_feature in range(n_cat_features):
            X_one_hot[sample, cat_one_hot_start[cat_feature] + X[sample, cat_feature]] = 1
    return X_one_hot

if __name__ == "__main__":
    import doctest

    n_failures, _ = doctest.testmod()
    if n_failures > 0:
        raise ValueError(f"{n_failures} tests failed")

    main()

# vimquickrun: python %
