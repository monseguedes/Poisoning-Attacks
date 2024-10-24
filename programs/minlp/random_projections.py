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
            for permutation in generate_monomials_exact_degree(n - 1, d - value):
                yield (value,) + permutation


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

    if stable_set:
        monomials_vector = [
            tuple(
                1 if x in list(range(1, len(monomials_vector[0]))) else x
                for x in monomial
            )
            for monomial in monomials_vector
        ]
        set_monomials_vector = OrderedSet(monomials_vector)
        monomials_vector = list(set_monomials_vector)

    monomials_matrix = []
    for i in range(len(monomials_vector)):
        monomials_matrix.append([])
        for j in range(len(monomials_vector)):
            if stable_set:
                monomials_matrix[i].append(
                    tuple(
                        [
                            int(monomials_vector[i][k] + monomials_vector[j][k] != 0)
                            for k in range(len(monomials_vector[i]))
                        ]
                    )
                )
            else:
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
    monomials_matrix = outer_product_monomials(monomials_vector, stable_set=stable_set)

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
        monomial_matrix = np.zeros((len(monomials_matrix), len(monomials_matrix)))
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
        
        # No. of features
        self.no_numfeatures = instance_data.no_numfeatures
        self.no_catfeatures = instance_data.no_catfeatures

        # No. of samples
        self.no_poison_samples = instance_data.no_poison_samples
        self.no_train_samples = instance_data.no_train_samples

        # Problem dimensions
        self.no_variables = (
            self.no_poison_samples * (self.no_numfeatures + self.no_catfeatures)
            + self.no_numfeatures
            + self.no_catfeatures
            + 1
            + self.no_poison_samples
        )
        self.size_psd_variable = sp.binom(self.no_variables + degree / 2, degree / 2)
        self.no_constraints = sp.binom(self.no_variables + degree, degree) 
        if self.size_psd_variable > 10000 or self.no_constraints > 10000:
            raise ValueError("The size of the PSD matrices or the number of constraints is too large: ", self.size_psd_variable, self.no_constraints)
        
        # Monomials vector
        self.distinct_monomials = generate_monomials_up_to_degree(self.no_variables, degree) 

        # Coefficients dict
        self.coefficients = self.get_coefficients_objective(instance_data)

        # Picking dictionaries
        self.A = self.get_SOS_dict()
        self.A_bounds = self.get_num_bound_dict()
        self.A_binary = self.get_cat_binary_dict()
        self.num_weights_dict = self.get_num_weights_dict()
        self.cat_weights_dict = self.get_cat_weights_dict()
        self.intercept_dict = self.get_intercept_dict()
        self.bilinear_dict = self.get_bilinear_dict()
        self.A_SOS1 = []
        for categorical_feature in range(self.no_catfeatures):
            self.A_SOS1.append(self.get_SOS1_dict(categorical_feature))
    
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
        variables_vector = np.zeros(no_variables)
        coefficients = {}

        # Populate the coefficients of the weights
        # num weights
        for i in range(no_numfeatures):
            index = variables_vector
            index[i] = 1
            coefficients[index] = (
                2
                / instance_data.no_train_samples
                * np.sum(instance_data.train_dataframe[:, i])
            )

        # cat weights
        for i in range(no_catfeatures):
            index = variables_vector
            index[no_numfeatures + i] = 1
            coefficients[index] = (
                2
                / instance_data.no_train_samples
                * np.sum(instance_data.train_dataframe[:, no_numfeatures + i])
            )

        # intercept
        index = variables_vector
        index[no_numfeatures + no_catfeatures] = 2 * instance_data.no_train_samples

        # num-num weights
        for i in range(no_numfeatures):
            for j in range(no_numfeatures):
                index = variables_vector
                index[i] += 1
                index[j] += 1
                coefficients[index] = (
                    1
                    / instance_data.no_train_samples
                    * np.sum(
                        instance_data.train_dataframe[:, i]
                        * instance_data.train_dataframe[:, j]
                    )
                )

        # num-cat weights
        for i in range(no_numfeatures):
            for j in range(no_catfeatures):
                index = variables_vector
                index[i] = 1
                index[no_numfeatures + j] = 1
                coefficients[index] = (
                    2
                    / instance_data.no_train_samples
                    * np.sum(
                        instance_data.train_dataframe[:, i]
                        * instance_data.train_dataframe[:, no_numfeatures + j]
                    )
                )

        # num-intercept
        for i in range(no_numfeatures):
            index = variables_vector
            index[i] = 1
            index[no_numfeatures + no_catfeatures] = 1
            coefficients[index] = (
                2
                / instance_data.no_train_samples
                * np.sum(instance_data.train_dataframe[:, i])
            )

        # cat-intercept
        for i in range(no_catfeatures):
            index = variables_vector
            index[no_numfeatures + i] = 1
            index[no_numfeatures + no_catfeatures] = 1
            coefficients[index] = (
                2
                / instance_data.no_train_samples
                * np.sum(instance_data.train_dataframe[:, no_numfeatures + i])
            )


        raise NotImplementedError

         
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

        A = {}
        for i, monomial in enumerate(self.distinct_monomials):
            A[monomial] = pick_specific_monomial(generate_monomials_matrix(self.no_variables, self.degree), monomial)

        raise NotImplementedError
    


