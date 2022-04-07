"""
    Small set of functions to construct transition matrices
"""

import numpy as np
from wagedyn.primitives import Parameters
import scipy.linalg

def createPoissonTransitionMatrix(n,delta):
    """
    Creates a transition matrix which stays constant with probability delta and
    draws uniformly with probability (1-delta)
    :param n:
    :param delta:
    :return:
    """
    return delta * np.eye(n) + (1-delta) * 1 / n * np.ones((n, n))

def createBlockPoissonTransitionMatrix(n0,n1,delta):
    """
    Creates a transition matrix which stays constant with probability delta and
    draws uniformly with probability (1-delta)
    :param n0: number of blocks
    :param n1: number of elements per block
    :param delta: probability to stay on diagonal
    :return:
    """

    blocks = [ createPoissonTransitionMatrix(n1,delta) for i in range(int(n0)) ]
    return scipy.linalg.block_diag(*blocks)

class TransitionMatrices:
    """
        Class that computes and returns transition matrices based on the distribution/copula specified.
    """

    def __init__(self, input_param=None):
        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
        if input_param is None:
            self.p = Parameters()
        else:
            self.p = input_param

    def gaussian_mat(self,corr):
        """
            Computes the Gaussian-based transition matrix corresponding to the transitions associated with worker
            productivity.
        :return: Gaussian transition matrix of dimension (num_x, num_x)
        """

        def compute_poisson_mat(delta, n):
            return (1 - delta) * np.eye(n) + delta * 1 / n * np.ones((n, n))

        return compute_poisson_mat(self.p.x_corr, self.p.num_x)

    def poisson_mat(self):
        """
            Computes the Poisson-based transition matrix corresponding to the transitions associated with match
            productivity.
        :return: Poisson transition matrix of dimension (num_z, num_z)
        """

        def compute_poisson_mat(delta, n):
            return (1 - delta) * np.eye(n) + delta * 1 / n * np.ones((n, n))

        return compute_poisson_mat(self.p.z_corr, self.p.num_z)
