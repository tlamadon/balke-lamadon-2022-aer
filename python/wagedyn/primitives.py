"""
    Set of functions and classes that help dealing with creating, modifying and saving primitives, such as parameters or the utility functions.
"""

import numpy as np
import logging
import json


class Parameters:

    """
        Class whose objects store the parameters used in the model.
    """

    def __init__(self, overwrite={}):
        """
            We can either initialize the object using a pre-defined set of parameters, or pass in a user-defined set
             of parameters.
            :param model_input: Dict or None
        """

        # Points in the Model
        self.num_l  = 101     # Number of points of evaluation
        self.num_v  = 200     # Number of points in the grid for V
        self.num_x  = 15      # Number of points of support for worker productivity
        self.num_np = 5       # Number of non-permanent levels
        self.num_z  = 7       # Number of points for match productivity
        self.num_s  = 50      # Number of points of support for piece rate contract

        # Time periods in the Model
        self.dt     = 0.25    # Time as a Fraction of Year

        # Utility Function Parameters
        self.u_rho = 1.5       # Risk aversion coefficient
        self.u_a   = 1.0
        self.u_b   = 1.0

        # Search Environment
        self.z_0      = 4          # Slice of value function of firms (index starts at 1)
        self.s_job    = 1.0        # Relative Efficiency of Search on the Job
        self.alpha    = 0.1        # Parameter for probability of finding a job
        self.sigma    = 1.0        # Parameter for probability of finding a job
        self.kappa    = 1.0        # Vacancy cost parameter

        # effort function that control separation
        self.efcost_sep = 0.005 * self.dt
        self.efcost_ce  = 0.3

        # Productivity shocks
        self.x_corr = 0.95  # Correlation in worker productivity
        self.z_corr = 0.95  # Correlation in match productivity

        # Productivity Function Parameters
        self.prod_var_x  = 1.0           # Variance of X (permanent)
        self.prod_var_x2 = 1.0           # Variance of X (non-permanent)
        self.prod_var_z  = 1.0           # Variance of Z
        self.prod_z      = 0.5           # Production function parameter
        self.prod_rho    = 1.0           # Production function parameter
        self.prod_mu     = 0.2           # Worker contribution
        self.prod_px     = 1.0           # Worker power (non linear in type)
        self.prod_py     = 1.0           # Firm power (nonlinear in type)
        self.prod_a      = 16 * self.dt  # Factor for output function
        self.prod_err_w  = 0.0           # Measurement error on wages
        self.prod_err_y  = 0.0           # Measurement error on wages

        # Discounting Rates
        self.beta     = 1 - (1 - 0.95) * self.dt  # Impatience
        self.int_rate = 1 / self.beta - 1         # Period interest rate

        # Unemployment Parameters
        self.u_bf_m = 0.05       # Intercept of benefit function for unemployed(x)
        self.u_bf_c = 0.5        # Slope of benefit function for unemployed(x) not used

        # Unemployment Parameters w_net = tau * w ^ lambda
        self.tax_lambda = 1.0       # curvature of the tax system 
        self.tax_tau    = 1.0       # proportion of take home
        self.tax_expost_lambda = 1.0  # this is for counterfactuals, allows to only apply taxes expost
        self.tax_expost_tau = 1.0     # this is for counterfactuals, allows to only apply taxes expost

        # Computational Parameters
        self.chain            = 1         # Chain id when running in parallel
        self.max_iter         = 5000
        self.max_iter_fb      = 5000
        self.verbose          = 5
        self.iter_display     = 25
        self.tol_simple_model = 1e-9
        self.tol_full_model   = 1e-8
        self.eq_relax_power   = 0.4       #  we relax the equilibrium constrain using an update rule based
        self.eq_relax_margin  = 500       #  on mumber of iterations
        self.eq_weighting_at0 = 0.01      # fitting J function with weight around 0

        # simulation parameters
        self.sim_ni      = 20000  # number of workers
        self.sim_nt      = 30     # time periods on top of nt_burn
        self.sim_nt_burn = 10     # periods to discard at begining
        self.sim_nh      = 200    # length of the firm history
        self.sim_nrep    = 20     # number of replication samples
        self.sim_net_earnings = False # whether to use net or gross earnings in the simulation

        for key, val in overwrite.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = val
            else:
                logging.warning("this key does not exists:" + key)

    @staticmethod
    def load(filename) -> 'Parameters':
        with open(filename, "r") as infile:
            pdict =  json.load(infile)
        p = Parameters(pdict)    
        return p

    def save(self,filename, append_dict = {}):
        temp_dict = self.__dict__.copy()
        temp_dict.update(append_dict)
        with open(filename, "w") as fp:
            json.dump(temp_dict, fp)

    def __getstate__(self):
        """ defines how the model is pickled """
        return self.__dict__     # get attribute dictionary
    
    def __setstate__(self, dict):
        """ defines how the model is unpickled """
        self.__dict__ = dict     # make dict our attribute dictionary

    def to_dict(self):
        return self.__dict__.copy()
    
    def get_x_components(self):
        """ give the values of x0 and x1 for each of the values of x """
        num_x0 = int(self.num_x / self.num_np)

        # the fixed heterogeneity component
        x0 = np.arange(num_x0)
        xt = np.arange(self.num_np)
        x0 = np.kron(x0,np.ones(self.num_np)) # permanent is slow moving
        xt = np.kron(np.ones(num_x0),xt) # permanent is slow moving
        return x0,xt

class Preferences:

    """
        Class whose methods represent the preferences and their derivatives, taking a Parameters object as input.
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

    def utility(self, wage):
        """
            Computes the utility function at a particular wage.
            :param wage: Argument of the function.
            :return: Output of the function.
        """
        aa = self.p.u_a * np.power(self.p.tax_tau, 1 - self.p.u_rho) 
        return np.divide(aa * np.power( wage, self.p.tax_lambda * (1.0 - self.p.u_rho)) - self.p.u_b,
                         1 - self.p.u_rho)
        # return np.divide(self.p.u_a * np.power(wage, 1 - self.p.u_rho) - self.p.u_b,
        #                  1 - self.p.u_rho)

    def utility_gross(self, wage):
        """
            Computes the utility function at a particular wage, not applying the tax function
            :param wage: Argument of the function.
            :return: Output of the function.
        """
        return np.divide(self.p.u_a * np.power(wage, 1 - self.p.u_rho) - self.p.u_b,
                         1 - self.p.u_rho)

    def inv_utility(self, value):
        """
            Computes the inverse utility function at a particular value.
            :param value: Argument of the function.
            :return: Output of the function.
        """
        aa = self.p.u_a * np.power(self.p.tax_tau, 1.0 - self.p.u_rho) 
        return np.power(np.divide((1.0 - self.p.u_rho) * value + self.p.u_b, aa),
                        (np.divide(1.0, self.p.tax_lambda * (1.0 - self.p.u_rho))))

    def utility_1d(self, wage):
        """
            Computes the first derivative of the utility function at a particular wage.
            :param wage: Argument of the function.
            :return: Output of the function.
        """
        #return self.p.u_a * np.power(wage, - self.p.u_rho)
        aa = self.p.u_a * np.power(self.p.tax_tau, 1.0 - self.p.u_rho) 
        return aa * self.p.tax_lambda * np.power(wage, self.p.tax_lambda * ( 1.0 - self.p.u_rho) - 1.0)

    def inv_utility_1d(self, value):
        """
            Computes the first derivative of the inverse utility function at a particular value.
            :param value: Argument of the function.
            :return: Output of the function.
        """
        aa = self.p.u_a * np.power(self.p.tax_tau, 1 - self.p.u_rho) 
        pow_arg = ( (1 - self.p.u_rho) * value + self.p.u_b   ) / aa
        return np.power( pow_arg, 1.0/( self.p.tax_lambda * (1 - self.p.u_rho) ) - 1.0) / ( self.p.tax_lambda * aa )

    def effort_cost(self, q_value):
        """
            Computes the effort cost function given the level of effort 'q'.
            :param q_value: Argument of the function.
            :return: Output of the function.
        """
        gam = self.p.efcost_ce ** -1
        return self.p.efcost_sep * q_value + \
            np.divide(self.p.efcost_sep,  self.p.efcost_ce - 1) - \
            np.divide(self.p.efcost_sep, 1 - gam) * (q_value ** (1 - gam))

    def inv_effort_cost_1d(self, V):
        """
            Returns the quit probability by computing the inverse of the first derivative of the
            effort cost function.
            :param eff: Argument of the function.
            :return: Output of the function.
        """
        #return np.power(1 + np.divide(np.maximum(eff, 0), self.p.efcost_sep), -self.p.efcost_ce)
        return np.power(1 + np.divide(np.maximum(-V, 0), self.p.efcost_sep), -self.p.efcost_ce)


    def log_consumption_eq(self, V):
        """
            Returns the log wage/consumption equivalent associated with a present value of the worker.
        """
        return(np.log(self.inv_utility( (1-self.p.beta) * V )))

    def log_profit_eq(self,J):
        """
            Returns the log profit equivalent associated with the firm present value
        """
        return( np.log( (1-self.p.beta) * J))


    def consumption_eq(self, V):
        """
            Returns the log wage/consumption equivalent associated with a present value of the worker.
        """
        return((self.inv_utility( (1-self.p.beta) * V )))

    def profit_eq(self,J):
        """
            Returns the log profit equivalent associated with the firm present value
        """
        return(( (1-self.p.beta) * J))
