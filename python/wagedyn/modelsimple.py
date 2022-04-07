"""
    Solves the model without the incentive constraint. 
    This is a simpler model that we use as starting value for the full model.
"""

import numpy as np
import logging
from scipy.stats import lognorm as lnorm
import matplotlib.pyplot as plt

import opt_einsum as oe

from wagedyn.primitives import Preferences
from wagedyn.probabilities import createPoissonTransitionMatrix,createBlockPoissonTransitionMatrix

ax = np.newaxis

def impose_decreasing(M):
    nv = M.shape[1]
    for v in reversed(range(nv-1)):
        M[:,v,:] = np.maximum(M[:,v,:],M[:,v+1,:])
    return M

class SimpleModel:
    """
        This solves a version of the model without on-the-job search and with fixed wages.
        We will use this as a starting value for the main model.
    """

    def __init__(self, input_param=None):
        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
        self.log = logging.getLogger('SimpleModel')
        self.log.setLevel(logging.INFO)

        self.p = input_param

        num_x1 = int(self.p.num_x / self.p.num_np)

        # Model preferences initialized by the same parameter object.
        self.pref = Preferences(input_param=self.p)

        # Worker and Match Productivity Heterogeneity in the Model
        self.X_grid = self.construct_x_grid()   # Create worker productivity grid
        self.Z_grid = self.construct_z_grid()   # Create match productivity grid

        # Production Function in the Model
        self.fun_prod = self.p.prod_a * np.power(self.X_grid * self.Z_grid, self.p.prod_rho)

        # Unemployment Benefits across Worker Productivities
        self.unemp_bf = np.ones(self.p.num_x) * self.p.u_bf_m

        # Transition matrices
        self.X_trans_mat = createBlockPoissonTransitionMatrix(self.p.num_x/self.p.num_np,self.p.num_np, self.p.x_corr)
        self.Z_trans_mat = createPoissonTransitionMatrix(self.p.num_z, self.p.z_corr)

        # Value Function Setup
        self.V_grid   = self.construct_v_array()
        self.sup_wage = self.pref.inv_utility((1 - self.p.beta) * self.V_grid)
        self.J_grid   = -10 * np.ones((self.p.num_z, self.p.num_v, self.p.num_x))
        self.X1_w     = np.kron(np.linspace(0, 1, num_x1), np.ones(self.p.num_np))

        # Unemployment value function (initial condition)
        self.value_unemp = self.pref.utility(self.unemp_bf) / self.p.int_rate

        # Probability of finding a job of quality v for a worker of type x (initial condition)
        self.prob_find_vx = np.zeros((self.p.num_v, self.p.num_x))

        # Probability of quitting for worker type x seeing a value v
        self.prob_quit_vx = np.zeros((self.p.num_v, self.p.num_x))

        # Equilibrium wage function
        self.wage_eqm = np.tile(self.pref.inv_utility(self.V_grid * self.p.int_rate)[ax, :, ax],
                                (self.p.num_z, 1, self.p.num_x))

        self.Vf_U = np.zeros(self.p.num_x)
        self.w_grid = np.linspace(self.unemp_bf.min(), self.fun_prod.max(), self.p.num_v )

    def solve_with_effort(self):
        """
        Solves the case with a choice of the quit probability.
        """

        w_grid = self.w_grid

        # Setting up the initial values for the VFI
        Ji =  self.J_grid
        ite_prob_vx  =  self.prob_find_vx

        W1i = np.zeros(Ji.shape)
        Ui  = np.zeros(self.p.num_x )

        # prepare expectation call
        Exz = oe.contract_expression('avb,az,bx->zvx', W1i.shape, self.Z_trans_mat.shape, self.X_trans_mat.shape)
        Ex  = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)

        for ite_num in range(2*self.p.max_iter):
            Ji2 = Ji
            Ui2 = Ui
            W1i2 = W1i

            # we compute the expected value next period by applying the transition rules
            EW1i = Exz(W1i,self.Z_trans_mat,self.X_trans_mat)
            EJ1i = Exz(Ji ,self.Z_trans_mat,self.X_trans_mat)
            EUi  = Ex(Ui, self.X_trans_mat)

            # we compute quit decision of the worker
            qi = self.pref.inv_effort_cost_1d( - self.p.beta * (EW1i - EUi))

            # update Ji
            Ji  = self.fun_prod[:,ax,:] - w_grid[ax,:,ax] + self.p.beta * (1 - qi) * EJ1i
            Ji = impose_decreasing(Ji)

            # Update worker value function
            W1i = self.pref.utility(w_grid)[ax,:,ax] - self.pref.effort_cost(qi) + \
                       self.p.beta * qi * EUi + self.p.beta * (1-qi) * EW1i

            # Apply the matching function
            ite_prob_vx = self.p.alpha * np.power(1 - np.power(
                np.divide(self.p.kappa, np.maximum(Ji[self.p.z_0 - 1, :, :], 1.0)), self.p.sigma), 1/self.p.sigma)

            # Update the guess for U(x) given p
            Ui = np.max( self.pref.utility_gross(self.unemp_bf[ax, :]) + self.p.beta * ite_prob_vx *
                               (W1i[self.p.z_0 - 1, :, :] - EUi[ax, :]) + self.p.beta * EUi[ax, :], axis=0)

            # Compute the norm-inf between the two iterations of U(x)
            error_u  = np.max(abs(Ui - Ui2))
            error_j  = np.max(abs(Ji - Ji2))
            error_w1 = np.max(abs(W1i - W1i2))

            if np.array([error_u, error_w1, error_j]).max() < self.p.tol_simple_model and ite_num>10:
                break

            if (ite_num % 25 ==0):
                self.log.debug('[{}] Error_U = {:2.4e}, Error_J = {:2.4e}, Error_W1 = {:2.4e}'.format(ite_num, error_u, error_j,error_w1))

        self.log.info('[{}] Error_U = {:2.4e}, Error_J = {:2.4e}, Error_W1 = {:2.4e}'.format(ite_num, error_u, error_j, error_w1))

        # extract U2E probability
        usearch = np.argmax( self.pref.utility(self.unemp_bf[ax, :]) + self.p.beta * ite_prob_vx *
                     (W1i[self.p.z_0 - 1, :, :] - EUi[ax, :]) + self.p.beta * EUi[ax, :], axis=0)
        Pr_u2e = [ ite_prob_vx[usearch[ix],ix] for ix in range(self.p.num_x) ]

        self.Vf_J    = Ji
        self.Vf_W1   = W1i
        self.Fl_ce   = self.pref.effort_cost(qi)
        self.Pr_e2u  = qi
        self.Fl_wage = w_grid
        self.Vf_U    = Ui
        self.Pr_u2e  = Pr_u2e
        self.prob_find_vx = ite_prob_vx

    def plot(self):
        nrows = 3
        ncols = 3

        plt.figure(figsize=(16, 12))

        plt.subplot(nrows, ncols, 1)
        plt.plot(np.log(self.Fl_wage), self.Vf_W1[self.p.z_0 - 1, :, :])
        plt.title('W1')

        plt.subplot(nrows, ncols, 2)
        plt.plot(np.log(self.Fl_wage), self.Vf_J[0, :, :])
        plt.plot(np.log(self.Fl_wage), self.Vf_J[self.p.num_z-1, :, :])
        plt.title('J1')

        plt.subplot(nrows, ncols, 3)
        plt.plot(self.Vf_U)
        plt.title('U(X)')

        plt.subplot(nrows, ncols, 4)
        plt.plot(np.log(self.Fl_wage), self.Pr_e2u[2, :, :])
        plt.title('E2U')

        plt.subplot(nrows, ncols, 5)
        plt.plot(np.log(self.Fl_wage), self.Fl_ce[2, :, :])
        plt.title('Cost of effort')

        plt.show()

    def construct_x_grid(self):
        """
            Construct a grid for worker productivity heterogeneity.
        """
        num_x0 = int(self.p.num_x / self.p.num_np)

        # the fixed heterogeneity component
        x0 = lnorm.ppf(q=np.linspace(0, 1, num_x0 + 2)[1:-1],
                       s=self.p.prod_var_x)
        # the time varyinng heterogeneity component
        xt = lnorm.ppf(q=np.linspace(0, 1, self.p.num_np + 2)[1:-1],
                       s=self.p.prod_var_x2)

        xx = np.kron(x0,xt) # permanent is slow moving
        xx = xx[ax,:] + np.zeros((self.p.num_z,self.p.num_x))
        return xx

    def construct_z_grid(self):
        """
            Construct a grid for match productivity heterogeneity.
        """

        exp_z = np.tile(np.linspace(0, 1, self.p.num_z + 2)[1:-1][:, ax],
                        (1, self.p.num_x))

        return lnorm.ppf(q=exp_z, s=self.p.prod_var_z)

    def construct_v_array(self):
        """
            Construct a grid for the value function using the production function to determine the min and max values.
            :return: An array of values corresponding to the value function realizations.
        """
        v_min = self.pref.utility(np.min(self.unemp_bf)) / self.p.int_rate
        v_max = self.pref.utility(1.0 * np.max(self.fun_prod)) / (1 - self.p.beta)
        return np.linspace(v_min, v_max, self.p.num_v)
