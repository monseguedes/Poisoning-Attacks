"""
Random projections for Polynomial Optimization of Poisoning Attacks.

The original poisoning attack problem is given by:
    max MSE 
    s.t. binary constraints categorical samples (degree 2)
         bound constraints numerical samples (degree 2) (?)
         SOS-1 constraints (degree 1)
         derivative with respect to num weights and trilinear terms (degree 3)
         derivative with respect to cat weights and trilinear terms (degree 3)
         derivative with respect to intercept and bilinear terms (degree 2)

    This means we need a SDP relaxation of degree 4. This is very large and we can
    only solve tiny problems.

    The reformulated poisoning attack problem is given by:
    max MSE
    s.t. binary constraints categorical samples (degree 2)
         bound constraints numerical samples (degree 2) (?)
         SOS-1 constraints (degree 1)
         derivative with respect to num weights with new bilinear variable (degree 2)
         derivative with respect to cat weights with new bilinear variable (degree 2)
         derivative with respect to intercept with new bilinear variable (degree 1)
         variable substitutiosn (degree 2)

    This allows us to solve larger problems.

"""

import numpy as np
import scipy.special as sp
import mosek.fusion as mf
import instance_data_class
from ordered_set import OrderedSet
import time
import math
import flipping_attack
import yaml


def generate_monomials_exact_degree(n, d):
    """
    Generates all monomials of a given degree and dimension.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials.

    Returns
    -------
    monomials : list
        List of monomials of degree d and dimension n in tuple format.

    Examples
    --------
    >>> generate_monomials(2, 2)
    [(2, 0), (1, 1), (0, 2)]

    >>> generate_monomials(3, 2)
    [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]

    """

    if n == 1:
        yield (d,)
    else:
        for value in range(d + 1):
            for permutation in generate_monomials_exact_degree(
                n - 1, d - value
            ):
                yield permutation + (value,) 

                

def generate_monomials_up_to_degree(n, d):
    """
    Generates all monomials up to a given degree and dimension.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials.

    Returns
    -------
    monomials : list
        List of monomials of degree d and dimension n in tuple format.

    Examples
    --------
    >>> generate_monomials_up_to_degree(2, 2)
    [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

    >>> generate_monomials_up_to_degree(3, 2)
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]

    """

    monomials = []
    for i in range(d + 1):
        monomials += list(generate_monomials_exact_degree(n, i))
    return monomials


def outer_product_monomials(monomials_vector, stable_set=False):
    """
    Generates matrix of monomials from outer product of vector of monomials
    of degree d/2 and n variables.

    Parameters
    ----------
    monomials_vector : list
        Vector of monomials.

    Returns
    -------
    monomials_matrix : numpy.ndarray
        Matrix of monomials.

    Examples
    --------
    >>> outer_product_monomials([(0,0), (1, 0), (0, 1)])
    [
    [(0,0), (1, 0), (0, 1)],
    [(1,0), (2, 0), (1, 1)],
    [(0,1), (1, 1), (0, 2)]
    ]

    """

    monomials_matrix = []
    for i in range(len(monomials_vector)):
        monomials_matrix.append([])
        for j in range(len(monomials_vector)):
            monomials_matrix[i].append(
                tuple(
                    [
                        monomials_vector[i][k] + monomials_vector[j][k]
                        for k in range(len(monomials_vector[i]))
                    ]
                )
            )

    return monomials_matrix


def generate_monomials_matrix(n, d, stable_set=False):
    """
    Generates matrix of monomials from outer product of vector of monomials
    of degree d/2 and n variables.

    Parameters
    ----------
    n : int
        Number of variables.
    d : int
        Degree of the monomials in the matrix.

    Returns
    -------
    monomials_matrix : numpy.ndarray
        Matrix of monomials of degree d and dimension n.

    Examples
    --------
    >>> generate_monomials_matrix(2, 2)
    [
    [(0,0), (1, 0), (0, 1)],
    [(1,0), (2, 0), (1, 1)],
    [(0,1), (1, 1), (0, 2)]
    ]

    """

    monomials_vector = generate_monomials_up_to_degree(n, math.floor(d / 2))
    monomials_matrix = outer_product_monomials(
        monomials_vector, stable_set=stable_set
    )

    return monomials_matrix


def pick_specific_monomial(monomials_matrix, monomial, vector=False):
    """
    Picks a specific monomial from the monomials matrix.
    Sets all entries to 0 except those corresponding to the monomial.

    Parameters
    ----------
    monomials_matrix : numpy.ndarray
        Matrix of monomials.
    monomial : tuple
        Monomial to be picked.

    Returns
    -------
    monomial_matrix : numpy.ndarray
        Matrix of monomial.

    Examples
    --------
    >>> pick_specific_monomial(generate_monomials_matrix(2, 2), (1, 1))
    [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
    ]
    """

    if vector:
        monomial_matrix = np.zeros(len(monomials_matrix))
        for i in range(len(monomials_matrix)):
            if monomials_matrix[i] == monomial:
                monomial_matrix[i] = 1

    else:
        monomial_matrix = np.zeros(
            (len(monomials_matrix), len(monomials_matrix))
        )
        for i in range(len(monomials_matrix)):
            for j in range(len(monomials_matrix)):
                if monomials_matrix[i][j] == monomial:
                    monomial_matrix[i][j] = 1


    return monomial_matrix


class SDP_data:
    def __init__(self, instance_data: instance_data_class, degree: int):
        """
        Initializes the SDP_data object.

        Parameters
        ----------
        instance_data : instance_data
            The instance data.
        degree : int
            The degree of the SDP relaxation.

        """

        self.degree = degree
        self.instance_data = instance_data

        # No. of features
        self.no_numfeatures = instance_data.no_numfeatures
        self.no_catfeatures = instance_data.no_catfeatures
        print("No. of numerical features: ", self.no_numfeatures)
        print("No. of categorical features: ", self.no_catfeatures)

        # No. of samples
        self.no_poison_samples = instance_data.no_poison_samples
        self.no_train_samples = instance_data.no_train_samples
        print("No. of poison samples: ", self.no_poison_samples)
        print("No. of training samples: ", self.no_train_samples)

        # Problem dimensions
        self.no_variables = (
            self.no_poison_samples
            * (self.no_numfeatures + self.no_catfeatures)
            + self.no_numfeatures
            + self.no_catfeatures
            + 1
            + self.no_poison_samples
        )
        print("No. of variables: ", self.no_variables)
        self.size_psd_variable = sp.binom(
            self.no_variables + degree / 2, degree / 2
        )
        print("Size of the PSD matrix: ", self.size_psd_variable)
        self.no_constraints = sp.binom(self.no_variables + degree, degree)
        print("No. of constraints: ", self.no_constraints)
        if self.size_psd_variable > 10000 or self.no_constraints > 10000:
            raise ValueError(
                "The size of the PSD matrices or the number of constraints is too large: ",
                self.size_psd_variable,
                self.no_constraints,
            )

        # Monomials vector
        self.distinct_monomials = generate_monomials_up_to_degree(
            self.no_variables, degree
        )
        print("No. of distinct monomials: ", len(self.distinct_monomials))

        # Coefficients dict
        self.coefficients = self.get_coefficients_objective(instance_data)
        print("No. of coefficients: ", len(self.coefficients))

        # Objective bound dict
        self.objective_bound = self.get_objective_bound_dict()

        # Picking dictionaries
        self.A = self.get_SOS_dict()
        print("No. of SOS pick: ", len(self.A))
        self.A_bounds = self.get_num_bound_dict()
        print("No. of numerical bound pick: ", len(self.A_bounds))
        self.A_binary = self.get_cat_binary_dict()
        self.weights_dict = self.get_weights_dict()
        self.intercept_dict = self.get_intercept_dict()
        self.bilinear_dict = self.get_bilinear_dict()
        # self.A_SOS1 = self.get_SOS1_dic() # Make matrix variable

    def get_objective_bound_dict(self):
        """
        Get the coefficients of the objective bound. This will
        be 0 for evertyhing except for the constant term, which
        will be 1.

        Returns
        -------
        coefficients : dict
            The coefficients of the objective bound.

        """

        coefficients = {}

        for monomial in self.distinct_monomials:
            coefficients[monomial] = 0

        tuple_of_constants = tuple(
            [int(i) for i in np.zeros(self.no_variables)]
        )
        coefficients[tuple_of_constants] = 1

        return coefficients

    def get_coefficients_objective(self, instance_data):
        """
        Get the coefficients of the objective function.

        The coefficients are given by:
        --------------------------------------------------
        num_weight_i degree 1 = 2/n * sum_{k=1}^{n} x_ik * y_k
        num_weight_i degree 2 = 1/n * sum_{k=1}^{n} x_ik^2
        --------------------------------------------------
        cat_weight_i degree 1 = 2/n * sum_{k=1}^{n} B_ik * y_k
        cat_weight_i degree 2 = 1/n * sum_{k=1}^{n} B_ik^2
        --------------------------------------------------
        intercept degree 1 = 2/n * sum_{k=1}^{n} 1 * y_k
        intercept degree 2 = 1/n
        --------------------------------------------------
        num_i times num_j = 1/n * sum_{k=1}^{n} x_ik * x_jk
        num_i times cat_j = 2/n * sum_{k=1}^{n} x_ik * B_jk
        num_i times intercept = 2/n * sum_{k=1}^{n} x_ik
        cat_i times intercept = 2/n * sum_{k=1}^{n} B_ik

        Parameters
        ----------
        instance_data : instance_data
            The instance data.

        Returns
        -------
        coefficients : dict
            The coefficients of the objective function.

        """

        # We want to make a vector with all variables, (x, B, w_num, w_cat, c, z),
        # and then associate a coefficient to each monomial up to degree 2.
        variables_vector = [int(i) for i in np.zeros(self.no_variables)]
        coefficients = {}

        for monomial in self.distinct_monomials:
            coefficients[monomial] = 0

        # Populate the coefficients of the weights
        # weights linear
        for j in range(self.no_numfeatures + self.no_catfeatures):
            monomial = variables_vector.copy()
            monomial[
                self.no_poison_samples
                * (self.no_numfeatures + self.no_catfeatures)
                + j
            ] = 1
            coefficients[tuple(monomial)] = (
                -2
                / instance_data.no_train_samples
                * np.sum(
                    instance_data.train_dataframe.iloc[:, j]
                    * instance_data.train_dataframe.iloc[:, -1]
                )
            )

        # weights quadratic
        for j in range(self.no_numfeatures + self.no_catfeatures):
            monomial = variables_vector.copy()
            monomial[
                self.no_poison_samples
                * (self.no_numfeatures + self.no_catfeatures)
                + j
            ] = 2
            coefficients[tuple(monomial)] = (
                1
                / instance_data.no_train_samples
                * np.sum(
                    instance_data.train_dataframe.iloc[:, j] ** 2
                    + instance_data.regularization
                )
            )

        # combination of two different weights
        for i in range(self.no_numfeatures + self.no_catfeatures):
            for j in range(self.no_numfeatures + self.no_catfeatures):
                if i != j:
                    monomial = variables_vector.copy()
                    monomial[
                        self.no_poison_samples
                        * (self.no_numfeatures + self.no_catfeatures)
                        + i
                    ] = 1
                    monomial[
                        self.no_poison_samples
                        * (self.no_numfeatures + self.no_catfeatures)
                        + j
                    ] = 1
                    coefficients[tuple(monomial)] = (
                        2
                        / instance_data.no_train_samples
                        * np.sum(
                            instance_data.train_dataframe.iloc[:, i]
                            * instance_data.train_dataframe.iloc[:, j]
                        )
                    )

        # linear intercept
        monomial = variables_vector.copy()
        monomial[
            (self.no_poison_samples + 1)
            * (self.no_numfeatures + self.no_catfeatures)
        ] = 1
        coefficients[tuple(monomial)] = (
            -2
            / instance_data.no_train_samples
            * np.sum(instance_data.train_dataframe.iloc[:, -1])
        )

        # quadratic intercept
        monomial = variables_vector.copy()
        monomial[
            (self.no_poison_samples + 1)
            * (self.no_numfeatures + self.no_catfeatures)
        ] = 2
        coefficients[tuple(monomial)] = 1 / instance_data.no_train_samples

        # weight-intercept
        for j in range(self.no_numfeatures + self.no_catfeatures):
            monomial = variables_vector.copy()
            monomial[
                self.no_poison_samples
                * (self.no_numfeatures + self.no_catfeatures)
                + j
            ] = 1
            monomial[
                (self.no_poison_samples + 1)
                * (self.no_numfeatures + self.no_catfeatures)
            ] = 1
            coefficients[tuple(monomial)] = (
                2
                / instance_data.no_train_samples
                * np.sum(instance_data.train_dataframe.iloc[:, j])
            )

        # constant term
        monomial = variables_vector.copy()
        coefficients[tuple(monomial)] = (
            1
            / instance_data.no_train_samples
            * np.sum(instance_data.train_dataframe.iloc[:, -1] ** 2)
        )

        return coefficients

    def get_SOS_dict(self, level=1):
        """
        Get the A parameter for the SOS matrix for the given level.

        Parameters
        ----------
        level : int
            The level of the SOS matrix.

        Returns
        -------
        A : dict
            The dictionary of the SOS matrix.

        """

        matrix = generate_monomials_matrix(self.no_variables, self.degree)

        self.A = {}
        for i, monomial in enumerate(self.distinct_monomials):
            self.A[monomial] = pick_specific_monomial(
                matrix,
                monomial,
            )

        return self.A

    def get_num_bound_dict(self):
        """
        Get the picking dict for the numerical bound constraints.

        The inequality constraints are given by:
        x_i ( 1 - x_i) and we have one per numerical poison sample.

        The dictionary is of the form:
        A_bound = {tuple of monomial : vector of coefficients for each constraint}

        Then, say we are writing the SDP constraint for a specific monomial,
        we will have a vector that tells us the coefficient of the monomial in
        each constraint of the original problem (one per numerical poison sample).

        For example, for the monomial associated with the linear term of the
        first numerical poison sample, we will have a vector of zeros except for
        the first element, which will be 1. For the quadratic term, we will have
        a vector of zeros except for the first element, which will be -1.

        Returns
        -------
        A_bounds : dict
            The dictionary of the numerical bound constraints.

        """

        num_features_vector = [
            int(i)
            for i in np.zeros(self.no_poison_samples * self.no_numfeatures)
        ]

        self.A_bounds = {}
        for monomial in self.distinct_monomials:
            self.A_bounds[monomial] = num_features_vector

        for p in range(self.no_poison_samples):
            for i in range(self.no_numfeatures):
                # linear term
                monomial = np.zeros(self.no_variables)
                vector = num_features_vector.copy()
                vector[p * self.no_numfeatures + i] = 1
                monomial[
                    p * (self.no_numfeatures + self.no_catfeatures) + i
                ] = 1
                self.A_bounds[tuple(monomial)] = vector
                # quadratic term
                monomial = np.zeros(self.no_variables)
                vector = num_features_vector.copy()
                vector[p * self.no_numfeatures + i] = -1
                monomial[
                    p * (self.no_numfeatures + self.no_catfeatures) + i
                ] = 2
                self.A_bounds[tuple(monomial)] = vector

        return self.A_bounds

    def get_cat_binary_dict(self):
        """
        Get the picking dict for the categorical binary constraints.

        The inequality constraints are given by:
        B_i ( B_i - 1) and we have one per categorical poison sample.

        The dictionary is of the form:
        A_binary = {tuple of monomial : vector of coefficients for each constraint}

        Then, say we are writing the SDP constraint for a specific monomial,
        we will have a vector that tells us the coefficient of the monomial in
        each constraint of the original problem (one per categorical poison sample).

        For example, for the monomial associated with the linear term of the
        first categorical poison sample (0, ..., 0, ..., 1, 0, ..., 0), we will
        have a vector of zeros except for the first element, which will be -1.
        For the quadratic term (0, ..., 0, ..., 2, 0, ..., 0), we will have a vector
        of zeros except for the first element, which will be 1.

        Returns
        -------
        A_binary : dict
            The dictionary of the categorical binary constraints.

        """

        cat_features_vector = [
            int(i)
            for i in np.zeros(self.no_poison_samples * self.no_catfeatures)
        ]

        self.A_binary = {}
        for monomial in self.distinct_monomials:
            self.A_binary[monomial] = cat_features_vector

        for p in range(self.no_poison_samples):
            for i in range(self.no_catfeatures):
                # linear term
                monomial = np.zeros(self.no_variables)
                vector = cat_features_vector.copy()
                vector[p * self.no_catfeatures + i] = -1
                monomial[
                    p
                    * (self.no_numfeatures + self.no_catfeatures)
                    + self.no_numfeatures
                    + i
                ] = 1
                self.A_binary[tuple(monomial)] = vector
                # quadratic term
                monomial = np.zeros(self.no_variables)
                vector = cat_features_vector.copy()
                vector[p * self.no_catfeatures + i] = 1
                monomial[
                    self.no_poison_samples * (self.no_numfeatures + i)
                ] = 2
                self.A_binary[tuple(monomial)] = vector

        return self.A_binary

    def get_weights_dict(self):
        """
        Get the picking dict for the constraints of the derivative of the weights.

        The weights constraints for each weight m are given by:
        2 / n + q * ( sum_{i=1}^{n} f(x_i^tr, B_i^tr,W, c) - y_i * x_im^tr + sum_{k=1}^{q} t_k x_km^tr ) + lambda * W_m

        We have coefficients for several variables:
        - the weights:
        --------------------------------------------------
        for weight j, the coefficient is 2 / n + q sum_{i=1}^{n} x_ij^tr x_im^tr where
        m is the position in the vector of all the constraints. When m matches j, we
        have to add the regularization term, i.e.
        monomial for weight j:
        [2 / n + q sum_{i=1}^{n} x_ij^tr x_i1^tr,
         2 / n + q sum_{i=1}^{n} x_ij^tr x_i2^tr,
            ...
         2 / n + q sum_{i=1}^{n} x_ij^tr x_ij^tr + lambda, when m = j,
            ...
         2 / n + q sum_{i=1}^{n} x_ij^tr x_i(no_weights)^tr]

        - the intercept:
        --------------------------------------------------
        the coefficient is 2 / n + q sum_{i=1}^{n} x_im^tr where m is the index in
        the vector of all the constraints.
        monomial for intercept:
        [2 / n + q sum_{i=1}^{n} x_i1^tr,
         2 / n + q sum_{i=1}^{n} x_i2^tr,
            ...
         2 / n + q sum_{i=1}^{n} x_i(no_weights)^tr]

        - the bilinear terms and the poison sample:
        --------------------------------------------------
        for each bilinear term k and poison sample
        x_km^p, the coefficient is 2 / n + q
        monoimal for bilinear term k and poison sample x_km^p:
        [2 / n + q if in position m, 0 otherwise,
            ...,
         2 / n + q if in position m, 0 otherwise]

        - constant term:
        --------------------------------------------------
        the coefficient is 2 / n + q sum_{i=1}^{n} y_i x_im^tr where m is the index in
        the vector of all the constraints.
        monomial for constant term:
        [2 / n + q sum_{i=1}^{n} y_i x_i1^tr,
         2 / n + q sum_{i=1}^{n} y_i x_i2^tr,
            ...
         2 / n + q sum_{i=1}^{n} y_i x_i(no_weights)^tr]

        Returns
        -------
        weights_dict : dict
            The dictionary of the weights.

        """

        weights_vector = [
            int(i) for i in np.zeros(self.no_numfeatures + self.no_catfeatures)
        ]
        monomial_zeros = [int(i) for i in np.zeros(self.no_variables)]

        self.weights_dict = {}
        for monomial in self.distinct_monomials:
            self.weights_dict[monomial] = weights_vector

        # weights term
        for i in range(self.no_numfeatures + self.no_catfeatures):
            monomial = monomial_zeros.copy()
            vector = weights_vector.copy()
            monomial[
                self.no_poison_samples
                * (self.no_numfeatures + self.no_catfeatures)
                + i
            ] = 1
            for j in range(self.no_numfeatures + self.no_catfeatures):
                vector[j] = (
                    2
                    / (self.no_train_samples + self.no_poison_samples)
                    * np.sum(
                        self.instance_data.train_dataframe.iloc[:, i]
                        * self.instance_data.train_dataframe.iloc[:, j]
                    )
                )

            vector[i] += self.instance_data.regularization
            
            self.weights_dict[tuple(monomial)] = vector

        # intercept
        monomial = monomial_zeros.copy()
        monomial[
            (self.no_poison_samples + 1)
            * (self.no_numfeatures + self.no_catfeatures)
        ] = 1
        vector = weights_vector.copy()
        for i in range(self.no_numfeatures + self.no_catfeatures):
            vector[i] = (
                2
                / (self.no_train_samples + self.no_poison_samples)
                * np.sum(self.instance_data.train_dataframe.iloc[:, i])
            )
        self.weights_dict[tuple(monomial)] = vector

        # bilear and sample term
        for p in range(self.no_poison_samples):
            monomial = monomial_zeros.copy()
            monomial[(self.no_poison_samples + 1) * (self.no_numfeatures + self.no_catfeatures) + 1 + p] = 1
            for i in range(self.no_numfeatures + self.no_catfeatures):
                vector = weights_vector.copy()
                monomial[
                    p * (self.no_numfeatures + self.no_catfeatures) + i
                ] = 1
                vector[i] = 2 / (
                    self.no_train_samples + self.no_poison_samples
                )
                self.weights_dict[tuple(monomial)] = vector
                monomial[
                    p * (self.no_numfeatures + self.no_catfeatures) + i
                ] = 0

        # constant term
        tuple_of_constants = tuple(
            [int(i) for i in np.zeros(self.no_variables)]
        )
        vector = weights_vector.copy()
        for i in range(self.no_numfeatures + self.no_catfeatures):
            vector[i] = (
                2
                / (self.no_train_samples + self.no_poison_samples)
                * - np.sum(
                    self.instance_data.train_dataframe.iloc[:, -1]
                    * self.instance_data.train_dataframe.iloc[:, i]
                )
            )

        self.weights_dict[tuple_of_constants] = vector

        return self.weights_dict

    def get_intercept_dict(self):
        """
        Get the picking dict for the constraints of the derivative of the intercept.

        The intercept constraint is given by:
        2 / n + q sum_{i=1}^{n} f(x_i^tr, B_i^tr,W, c) - y_i + sum_{k=1}^{q} t_k

        We have coefficients for several variables:
        - the weights:
        --------------------------------------------------
        for weight j, the coefficient is 2 / n + q sum_{i=1}^{n} x_ij^tr.
        monomial for weight j: 2 / n + q sum_{i=1}^{n} x_ij^tr

        - the intercept:
        --------------------------------------------------
        the coefficient is 2 / n + q sum_{i=1}^{n} 1.

        - the bilinear terms:
        --------------------------------------------------
        for each bilinear term k, the coefficient is 2 / n + q.
        monomial for bilinear term k: 2 / n + q

        - constant term:
        --------------------------------------------------
        the coefficient is 2 / n + q sum_{i=1}^{n} y_i.
        monomial for constant term: 2 / n + q sum_{i=1}^{n} y_i

        Returns
        -------
        intercept_dict : dict
            The dictionary of the intercept.

        """

        monomial_vector = [int(i) for i in np.zeros(self.no_variables)]

        self.intercept_dict = {}
        for monomial in self.distinct_monomials:
            self.intercept_dict[monomial] = 0

        # constant term
        tuple_of_constants = tuple(
            [int(i) for i in np.zeros(self.no_variables)]
        )
        self.intercept_dict[tuple_of_constants] = (
            2
            / (self.no_train_samples + self.no_poison_samples)
            * - np.sum(self.instance_data.train_dataframe.iloc[:, -1])
        )

        # intercept
        monomial = monomial_vector.copy()
        monomial[
            (self.no_poison_samples + 1)
            * (self.no_numfeatures + self.no_catfeatures)
        ] = 1
        self.intercept_dict[tuple(monomial)] = (
            2
            / (self.no_train_samples + self.no_poison_samples)
            * self.no_train_samples
        )

        # bilinear term
        for p in range(self.no_poison_samples):
            monomial = monomial_vector.copy()
            monomial[
                (self.no_poison_samples + 1)
                * (self.no_numfeatures + self.no_catfeatures)
                + 1
                + p
            ] = 1
            self.intercept_dict[tuple(monomial)] = 2 / (
                self.no_train_samples + self.no_poison_samples
            )

        # weights
        for i in range(self.no_numfeatures + self.no_catfeatures):
            monomial = monomial_vector.copy()
            monomial[
                self.no_poison_samples
                * (self.no_numfeatures + self.no_catfeatures)
                + i
            ] = 1
            self.intercept_dict[tuple(monomial)] = (
                2
                / (self.no_train_samples + self.no_poison_samples)
                * np.sum(self.instance_data.train_dataframe.iloc[:, i])
            )

        return self.intercept_dict

    def get_bilinear_dict(self):
        """
        Get the picking dict for the constraints of the derivative of the bilinear terms.

        The bilinear term constraint is given by:
        f(x_i^p, B_i^p, W, c) - y_i - t_i

        We have coefficients for several variables:
        - weight-sample:
        --------------------------------------------------
        for each weight-sample pair, the coefficient is 1.
        monomial for weight-sample pair:
        [1, if in position m, 0 otherwise]

        - the intercept:
        --------------------------------------------------
        the coefficient is 1.
        monomial for intercept:
        [1,
        1,
            ...
        1]

        - constant term:
        --------------------------------------------------
        the coefficient is -y_i.
        monomial for constant term:
        [-y_1,
         -y_2,
         ...
         -y_n]

        Returns
        -------
        bilinear_dict : dict
            The dictionary of the bilinear terms.

        """

        bilinear_vector = [int(i) for i in np.zeros(self.no_poison_samples)]
        monomial_zeros = [int(i) for i in np.zeros(self.no_variables)]

        self.bilinear_dict = {}
        for monomial in self.distinct_monomials:
            self.bilinear_dict[tuple(monomial)] = bilinear_vector

        # weight and sample
        for i in range(self.no_numfeatures + self.no_catfeatures):
            monomial = monomial_zeros.copy()
            monomial[
                self.no_poison_samples * (self.no_numfeatures + self.no_catfeatures) + i
            ] = 1 
            for p in range(self.no_poison_samples):
                monomial[
                    p * (self.no_numfeatures + self.no_catfeatures)
                    + i
                ] = 1
                vector = bilinear_vector.copy()
                vector[p] = 1
            self.bilinear_dict[tuple(monomial)] = vector

        # bilinear term
        vector = bilinear_vector.copy()
        monomial = monomial_zeros.copy()
        for i in range(self.no_poison_samples):
            monomial[
                (self.no_poison_samples + 1)
                * (self.no_numfeatures + self.no_catfeatures)
                + 1
                + i
            ] = 1
            vector[i] = -1
        self.bilinear_dict[tuple(monomial)] = vector
            
        # intercept      
        vector = bilinear_vector.copy()
        for p in range(self.no_poison_samples):
            vector[p] = 1
        monomial = monomial_zeros.copy()
        monomial[
            (self.no_poison_samples + 1)
            * (self.no_numfeatures + self.no_catfeatures)
        ] = 1
        self.bilinear_dict[tuple(monomial)] = vector

        # constant term
        tuple_of_constants = tuple([i for i in np.zeros(self.no_variables)])
        vector = bilinear_vector.copy()
        for i in range(self.no_poison_samples):
            vector[i] = -self.instance_data.poison_dataframe.iloc[i, -1]
        self.bilinear_dict[tuple_of_constants] = vector

        return self.bilinear_dict

    def get_SOS1_dic(self):
        """
        Get the picking dict for the SOS1 constraints.

        The SOS1 constraints are given by:
        sum_{i=1}^{n(j)} b_kij - 1 for all j

        Since the degree of this variable is 1, we can multiply it by
        a polynomial of degree 1, instead of constant. This means that
        all monomials of degree 1 will appear, and they will all be
        multiplied by all the variables.

        The rows of the matrix are the number of categorical features,
        and the columns are the number of monomials of degree 1.
        [polynomial of cat feature 1
         polynomial of cat feature 2,
         ...
         polynomial of cat feature n]

        We have coefficients for several variables:
        - the weights:
        for weight j, the matrix is of the form:
        [[-1 if constant term, 0 otherwise]]
        - the weights-cateorical features:
        for weight j, and categorical feature kij, the matrix is of the form:
        [[1 if weight j position, 0 otherwise] repeat for all k
        ]
        - the intercept:
        [[-1 if constant term, 0 otherwise]]
        - the intercept-categorical features:
        - the bilinear terms:
        [[-1 if constant term, 0 otherwise]]
        - the bilinear terms-categorical features:
        - the constant term:
        [[-1 if constant term, 0 otherwise]]

        Parameters
        ----------
        categorical_feature : int
            The categorical feature.

        Returns
        -------
        A_SOS1 : dict
            The dictionary of the SOS1 matrix.

        """

        raise NotImplementedError(
            "The SOS1 constraints are not implemented yet."
        )


def solve_sdp_relaxation(instance_data: instance_data_class, degree: int):
    """
    Solves the SDP relaxation of the poisoning attack problem.

    Parameters
    ----------
    instance_data : instance_data
        The instance data.
    degree : int
        The degree of the SDP relaxation.

    Returns
    -------
    solution : dict
        The solution of the SDP relaxation.

    """

    # Initialize the SDP data
    parameters = SDP_data(instance_data, degree)

    # Create a model
    with mf.Model("SDP") as M:
        # Create variables
        X = M.variable(
            "X", mf.Domain.inPSDCone(int(parameters.size_psd_variable))
        )
        a = M.variable("a")
        c_bound = M.variable(
            "c_bound",
            parameters.no_poison_samples * parameters.no_numfeatures,
            mf.Domain.greaterThan(0.0),
        )
        c_binary = M.variable(
            "c_binary",
            parameters.no_poison_samples * parameters.no_catfeatures,
        )
        c_weights = M.variable(
            "c_weights", parameters.no_numfeatures + parameters.no_catfeatures
        )
        c_intercept = M.variable("c_intercept")
        c_bilinear = M.variable("c_bilinear", parameters.no_poison_samples)
        c_SOS1 = M.variable(
            "c_SOS1",
            [
                parameters.no_poison_samples,
                len(instance_data.categorical_feature_names),
            ],
        )  # Make matrix variable

        # Objective function
        M.objective(mf.ObjectiveSense.Minimize, a)

        # Constraints
        for monomial in parameters.distinct_monomials:

            # print(f"The constraint for monomial {monomial} is: ")
            # print(f"<{parameters.A[monomial]}, X> + <{parameters.A_bounds[monomial]}, c_bound> + <{parameters.A_binary[monomial]}, c_binary> + <{parameters.weights_dict[monomial]}, c_weights> + <{parameters.intercept_dict[monomial]}, c_intercept> + <{parameters.bilinear_dict[monomial]}, c_bilinear> + {parameters.objective_bound[monomial]}a = {parameters.coefficients[monomial]}")
            
            M.constraint(
                mf.Expr.sub( 
                    mf.Expr.mul(parameters.objective_bound[monomial], a),
                    mf.Expr.add(
                        mf.Expr.add(
                            mf.Expr.add(
                                mf.Expr.add(
                                    mf.Expr.add(
                                        mf.Expr.dot(parameters.A[monomial], X),
                                        mf.Expr.dot(
                                            parameters.A_bounds[monomial],
                                            c_bound,
                                        ),
                                    ),
                                    mf.Expr.dot(
                                        parameters.A_binary[monomial], c_binary
                                    ),
                                ),
                                mf.Expr.dot(
                                    parameters.weights_dict[monomial],
                                    c_weights,
                                ),
                            ),
                            mf.Expr.mul(
                                parameters.intercept_dict[monomial],
                                c_intercept,
                            ),
                        ),
                        mf.Expr.dot(
                            parameters.bilinear_dict[monomial], c_bilinear
                        ),
                    )
                ),
                # mf.Expr.dot(parameters.A_SOS1[monomial], c_SOS1) # Make matrix variable
                mf.Domain.lessThan(parameters.coefficients[monomial]),
            )
        
        print("Training dataframe: ")
        print(parameters.instance_data.train_dataframe)
        print("Poison dataframe: ")
        print(parameters.instance_data.poison_dataframe)

        # Solve the model
        start_time = time.time()
        M.solve()
        end_time = time.time()


        solution = {
            "a": a.level(),
            "objective": M.primalObjValue(),
            "size_psd_variable": parameters.size_psd_variable,
            "no_constraints": parameters.no_constraints,
            "computation_time": end_time - start_time,
        }

    return solution


if __name__ == "__main__":

    with open("programs/minlp/config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    config["regularization"] = 0.1
    config["training_samples"] = 1
    config["dataset_name"] = "test"
    config["poison_rate"] = 100

    time_start = time.time()
    instance_data = instance_data_class.InstanceData(config=config)
    time_end = time.time()
    print("Time to create instance data: ", time_end - time_start)

    degree = 2

    time_start = time.time()
    solution = solve_sdp_relaxation(instance_data, degree)
    time_end = time.time()
    print("Time to solve SDP relaxation: ", time_end - time_start)
    print("Objective value: ", solution["objective"])

    # Flipping attack strategy
    numerical_model = None
    _, _, instance_data, regression_parameters = flipping_attack.run(
        config, instance_data, numerical_model,
    )
    print("Flipping objective: ", regression_parameters["flipping_mse"])
