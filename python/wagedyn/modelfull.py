"""
    Contains the code that solves the full model (ie the optimal contract in equilibrium)
"""

import opt_einsum as oe
import numpy as np
import logging
import matplotlib.pyplot as plt
import pickle

from datetime import datetime

import wagedyn.modelsimple as smodel
from wagedyn.search import JobSearchArray
from wagedyn.valuefunction import PowerFunctionGrid
from wagedyn.primitives import Parameters

ax = np.newaxis

def impose_decreasing(M):
    nv = M.shape[1]
    for v in reversed(range(nv-1)):
        M[:,v,:] = np.maximum(M[:,v,:],M[:,v+1,:])
    return M


def impose_increasing(A0):
    A = np.copy(A0)
    nv = len(A)
    for v in range(1,nv):
        A[v] = np.maximum(A[v],A[v-1])
    return A


def array_exp_dist(A,B,h):
    """ 
        computes sqrt( (A-B)^2 ) / sqrt(B^2) weighted by exp(- (B/h)^2 ) 
    """
    # log_weight = - 0.5*np.power(B/h,2) 
    # # handling underflow gracefully
    # log_weight = log_weight - log_weight.max()
    # weight = np.exp( np.maximum( log_weight, -100))
    # return  (np.power( A-B,2) * weight ).mean() / ( np.power(B,2) * weight ).mean() 
    weight = np.exp( - 0.5*np.power(B/h,2))
    return  (np.power( A-B,2) * weight ).mean() / ( np.power(B,2) * weight ).mean() 



def array_dist(A,B):
    """ 
        computes sqrt( (A-B)^2 ) / sqrt(B^2) weighted by exp(- (B/h)^2 ) 
    """
    return  (np.power( A-B,2) ).mean() / ( np.power(B,2) ).mean() 



class FullModel:
    """
        Class that solves for the full model. It uses the simple version
        as a starting value.
    """

    def __init__(self, input_param: Parameters, tensorboard=False):
        """
            Initialize with a parameter object.
            :param input_param: Input parameter object, can be None
        """
        self.log = logging.getLogger('FullModel')
        self.log.setLevel(logging.INFO)
        
        self.p = input_param
        self.deriv_eps = 1e-3 # step size for derivative

        # ----------------------------------------------------
        # solve the simple model, use it as starting values
        simple_model = smodel.SimpleModel(self.p)
        simple_model.solve_with_effort()
        self.simple_model = simple_model

        # create the equilibrium search representation from using the simple model
        self.js = JobSearchArray(self.p.num_x)
        self.js.update(simple_model.Vf_W1[self.p.z_0 - 1, :, :], simple_model.prob_find_vx)

        # Extracting primitives from the simple model
        self.pref        = simple_model.pref
        self.X_grid      = simple_model.X_grid
        self.Z_grid      = simple_model.Z_grid
        self.Z_trans_mat = simple_model.Z_trans_mat
        self.X_trans_mat = simple_model.X_trans_mat

        self.fun_prod = simple_model.fun_prod
        self.unemp_bf = simple_model.unemp_bf

        # Value Function Setup
        self.w_grid = simple_model.w_grid
        self.rho_grid = 1 / self.pref.utility_1d(self.w_grid)
        self.Vf_W1 = simple_model.Vf_W1  # simple model was evaluated at the same grid
        self.Vf_J  = simple_model.Vf_J   # simple model was evaluated at the same grid
        self.Vf_U  = simple_model.Vf_U

        # Model of the value function
        self.J1p = None

        # Unemployment flow value
        self.Fl_uf = self.pref.utility(self.unemp_bf)
        self.Vf_U = simple_model.Vf_U
        self.Pr_u2e = simple_model.Pr_u2e

        # policies
        self.rho_j2j  = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))
        self.rho_u2e  = np.zeros((self.p.num_x))  # rho that worker gets when coming out of unemployment
        self.rho_star = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x)) #
        self.qe_star  = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x)) # quiting policy
        self.pe_star  = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x)) # job search policy
        self.ve_star  = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x)) # job search policy (value applied to)

        # errors
        self.error_w1  = 0
        self.error_j   = 0
        self.error_j1p = 0
        self.error_js  = 0
        self.niter     = 0

    @staticmethod
    def load(filename) -> 'FullModel':
        model= None
        with open(filename, "rb") as infile:
            model =  pickle.load(infile)
        return model

    def __getstate__(self):
        """ defines how the model is pickled """
        odict = self.__dict__     # get attribute dictionary
        if 'simple_model' in odict.keys():
            del odict['simple_model'] # remove simple model
        del odict['log']          # remove logger

        return odict

    def get_parameters(self) -> 'Parameters':
        return(self.p)

    def save(self,filename):
        with open(filename, "wb") as output_file:
            pickle.dump(self, output_file)

    def __setstate__(self, dict):
        """ defines how the model is unpickled """

        # need to recreate the simple model and the seasrch representation
        self.__dict__ = dict     # make dict our attribute dictionary
        self.log = logging.getLogger('FullModel')


    def matching_function(self,J1):
        return self.p.alpha * np.power(1 - np.power(
            np.divide(self.p.kappa, np.maximum(J1, self.p.kappa)), self.p.sigma),
                                1 / self.p.sigma)

    def getWorkerDecisions(self, EW1, EU, employed=True):
        """
        :param EW1: Expected value of employment
        :param EU:  Expected value of unemployment
        :param employed: whether the worker is employed (in which case we multiply by efficiency
        :return: pe,re,qi search decision and associated return, as well as quit decision.
        """
        pe, re = self.js.solve_search_choice(EW1)
        assert (~np.isnan(pe)).all(), "pe is not NaN"
        assert (pe <= 1).all(), "pe is not less than 1"
        assert (pe >= -1e-10).all(), "pe is not larger than 0"

        if employed:
            pe = pe * self.p.s_job
            re = re * self.p.s_job

        # solve quit effort
        # qi = self.pref.inv_effort_cost_1d(self.p.beta * (re + (W1i - Ui) + 1/rho_grid[ax,:,ax] * Ji))
        qi = self.pref.inv_effort_cost_1d( - self.p.beta * (re + EW1 - EU))
        assert (qi <= 1).all(), "pe is not less than 1"
        assert (qi >= 0).all(), "pe is not larger than 0"

        # construct the continuation probability
        pc = (1 - pe) * (1 - qi)

        return pe, re, qi, pc

    def evaluateLagrangian(self):
        pass
        # R1 = - self.pref.effort_cost(qi) + \
        #      self.p.beta * qi * EUi + self.p.beta * (1 - qi) * (re + EW1i)
        # L1 = self.fun_prod[:, ax, :] - w_grid[v0] + \
        #      rho_grid[v0] * (self.pref.utility(w_grid[v0]) + R1) - \
        #      rho_grid[ax, :, ax] * self.p.beta * (1 - pe) * (1 - qi) * EW1i + \
        #      self.p.beta * (1 - pe) * (1 - qi) * (EJpi + rho_grid[ax, :, ax] * EW1i)

    def solve(self, update_eq = True, plot = False):
        """
            Solves for a fixed point in U(x) and J(z, v, x) given the set of problems for the worker and firm in the
            presence of the free-entry condition.
        :return: Updates the unemployment distribution and probability of finding a job of quality v for a worker of
                    type x
        """

        # Setting up the initial values for the VFI
        Ui = self.Vf_U
        Ji = self.Vf_J
        W1i = self.Vf_W1

        # create representation for J1p
        J1p = PowerFunctionGrid(W1i, Ji)

        EW1_star = np.copy(self.Vf_J)
        EJ1_star = np.copy(self.Vf_J)

        rho_bar = np.zeros((self.p.num_z, self.p.num_x))
        rho_star = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))

        # we create a wage grid
        w_grid = self.w_grid
        rho_grid = self.rho_grid

        # prepare expectation call
        Exz = oe.contract_expression('avb,az,bx->zvx', W1i.shape, self.Z_trans_mat.shape, self.X_trans_mat.shape)
        Ex = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)
        log_diff = np.zeros_like(EW1_star)

        ite_num = 0
        error_js = 1

        for ite_num in range(self.p.max_iter):
            # Store temporary value of J
            Ji2 = Ji
            W1i2 = W1i
            Ui2 = Ui

            # evaluate J1 tomorrow using our approximation
            Jpi = J1p.eval_at_W1(W1i)

            # we compute the expected value next period by applying the transition rules
            EW1i = Exz(W1i, self.Z_trans_mat, self.X_trans_mat)
            EJpi = Exz(Jpi, self.Z_trans_mat, self.X_trans_mat)
            EUi = Ex(Ui, self.X_trans_mat)

            # get worker decisions
            _, _, _, pc = self.getWorkerDecisions(EW1i, EUi)
            # get worker decisions at EW1i + epsilon
            _, _, _, pc_d = self.getWorkerDecisions(EW1i + self.deriv_eps, EUi)

            # compute derivative where continuation probability is >0
            log_diff[:] = np.nan
            log_diff[pc > 0] = np.log(pc_d[pc > 0]) - np.log(pc[pc > 0])
            foc = rho_grid[ax, :, ax] - EJpi * log_diff / self.deriv_eps
            assert (np.isnan(foc) & (pc > 0)).sum() == 0, "foc has NaN values where p>0"

            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):

                    assert np.all(EW1i[iz, 1:, ix] > EW1i[iz, :-1, ix])
                    # find highest V with J2J search
                    rho_bar[iz, ix] = np.interp(self.js.jsa[ix].e0, EW1i[iz, :, ix], rho_grid)
                    rho_min = rho_grid[pc[iz, :, ix] > 0].min()  # lowest promised rho with continuation > 0

                    # look for FOC below  rho_0
                    Isearch = (rho_grid <= rho_bar[iz, ix]) & (pc[iz, :, ix] > 0)
                    if Isearch.sum() > 0:
                        rho_star[iz, Isearch, ix] = np.interp(rho_grid[Isearch],
                                                              impose_increasing(foc[iz, Isearch, ix]),
                                                              rho_grid[Isearch], right=rho_bar[iz, ix])

                    # look for FOC above rho_0
                    Ieffort = (rho_grid > rho_bar[iz, ix]) & (pc[iz, :, ix] > 0)
                    if Ieffort.sum() > 0:
                        #assert np.all(foc[iz, Ieffort, ix][1:] > foc[iz, Ieffort, ix][:-1])
                        rho_star[iz, Ieffort, ix] = np.interp(rho_grid[Ieffort],
                                                              foc[iz, Ieffort, ix], rho_grid[Ieffort])

                    # set rho for quits to the lowest value
                    Iquit = ~(pc[iz, :, ix] > 0)
                    if Iquit.sum() > 0:
                        rho_star[iz, Iquit, ix] = rho_min

                    # get EW1_Star and EJ1_star
                    EW1_star[iz, :, ix] = np.interp(rho_star[iz, :, ix], rho_grid, EW1i[iz, :, ix])
                    EJ1_star[iz, :, ix] = np.interp(rho_star[iz, :, ix], rho_grid, EJpi[iz, :, ix])

            assert np.isnan(EW1_star).sum() == 0, "EW1_star has NaN values"

            # get pstar, qstar
            pe_star, re_star, qi_star, _ = self.getWorkerDecisions(EW1_star, EUi)

            # Update firm value function
            Ji = self.fun_prod[:, ax, :] - w_grid[ax, :, ax] + self.p.beta * (1 - pe_star) * (1 - qi_star) * EJ1_star

            # Update worker value function
            W1i = self.pref.utility(w_grid)[ax, :, ax] - self.pref.effort_cost(qi_star) + \
                self.p.beta * qi_star * EUi + self.p.beta * (1-qi_star) * (re_star + EW1_star)
            W1i = .2*W1i + .8*W1i2

            # Update present value of unemployment
            if update_eq:
                _, ru, _, _ = self.getWorkerDecisions(EUi, EUi, employed=False)
                Ui = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EUi)
                Ui = 0.2*Ui + 0.8*Ui2

            # Updating J1 representation
            error_j1p_chg, rsq_j1p = J1p.update_cst_ls(W1i, Ji)

            # Compute convergence criteria
            error_j1i = array_exp_dist(Ji,Ji2,100) #np.power(Ji - Ji2, 2).mean() / np.power(Ji2, 2).mean()  
            error_j1g = array_exp_dist(Jpi,J1p.eval_at_W1(W1i), 100)
            error_w1 = array_dist(W1i, W1i2)
            error_u = array_dist(Ui,Ui2)

            # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------
                    if (np.array([error_w1, error_js, error_j1p_chg]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            P_xv = self.matching_function(J1p.eval_at_W1(W1i)[self.p.z_0 - 1, :, :])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            error_js = self.js.update(W1i[self.p.z_0 - 1, :, :], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    if (np.array([error_w1, error_j1g]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            if (ite_num % 25) == 0:
                self.log.debug('[{}] W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e} U= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, error_u, self.js.rsq(), rsq_j1p ))

        self.log.info('[{}][final]  W1= {:2.4e} Ji= {:2.4e} Jg= {:2.4e} Jp= {:2.4e} Js= {:2.4e} U= {:2.4e}  rsq_p= {:2.4e} rsq_j= {:2.4e}'.format(
                                     ite_num, error_w1, error_j1i, error_j1g, error_j1p_chg, error_js, error_u, self.js.rsq(), rsq_j1p ))

        # --------- wrapping up the model ---------

        # find rho_j2j
        rho_j2j = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))
        ve_star = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))
        for ix in range(self.p.num_x):
            for iz in range(self.p.num_z):
                ve_star[iz, :, ix] = self.js.ve(ix,  EW1_star[iz,:,ix])
                rho_j2j[iz, :, ix] = np.interp(ve_star[iz, :, ix], W1i[self.p.z_0, :, ix], rho_grid)

        # find rho_u2e
        rho_u2e = np.zeros(self.p.num_x)
        Pr_u2e  = np.zeros(self.p.num_x)
        for ix in range(self.p.num_x):
            ve = self.js.ve(ix, EUi[ix])
            rho_u2e[ix] = np.interp(ve, W1i[self.p.z_0, :, ix], rho_grid)
            Pr_u2e[ix] = self.js.pe(ix, EUi[ix]) # this does not include the inefficiency of search for employed

        # find the target wages for each x,z
        target_w   = np.zeros((self.p.num_z, self.p.num_x))
        target_rho = np.zeros((self.p.num_z, self.p.num_x))
        for ix in range(self.p.num_x):
            for iz in range(self.p.num_z):
                rho1 = rho_grid.max()
                rho0 = rho_grid.min()
                for _ in range(20):
                    rhog = (rho0 + rho1) / 2
                    val = np.interp(rhog, rho_grid, EJpi[iz, :, ix])
                    if val > 0:
                        rho0 = rhog
                    else:
                        rho1 = rhog
                target_rho[iz, ix] = rhog
                target_w[iz, ix] = np.interp(rhog, rho_grid, w_grid)

        # value functions
        self.Vf_J = Ji
        self.Vf_W1 = W1i
        self.Vf_U = Ui
        self.J1p = J1p

        # policies
        self.rho_j2j = rho_j2j
        self.rho_u2e = rho_u2e
        self.rho_star = rho_star
        self.qe_star = qi_star
        self.pe_star = pe_star
        self.ve_star = ve_star
        self.Pr_u2e = Pr_u2e
        self.target_w = target_w
        self.target_rho = target_rho

        self.Fl_ce = self.pref.effort_cost(qi_star)
        self.Vf_Vbar = np.array([self.js[x].e0 for x in range(self.p.num_x)])

        if plot:
            self.plot()
            plt.show()

        self.error_w1 = error_w1
        self.error_j = error_j1i
        self.error_j1p = error_j1g
        self.error_js = error_js
        self.niter = ite_num

        return self

    def solve_fb(self, plot = False, upper_env=False):
        """
            Should be ran after the main equilibrium has been solved, 
            this will solve for the first best solution of the firm problem, keeping the
            equilibrium quantities fixed.
        """

        # we create a wage grid
        w_grid = self.w_grid
        rho_grid = self.rho_grid

        Ui = self.Vf_U
        Ji = self.Vf_J
        W1i = self.Vf_W1
        J1p = self.J1p

        Ji2 = np.zeros_like(Ji)
        W1i2 = np.zeros_like(W1i)

        # prepare expectation call
        Exz = oe.contract_expression('avb,az,bx->zvx', W1i.shape, self.Z_trans_mat.shape, self.X_trans_mat.shape)
        Ex = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)
        EUi = Ex(Ui, self.X_trans_mat)

        ite_num = 0

        for ite_num in range(self.p.max_iter):
            # Store temporary value of J
            Ji2[:] = Ji
            W1i2[:] = W1i

            # we compute the expected value next period by applying the transition rules
            EW1i = Exz(W1i, self.Z_trans_mat, self.X_trans_mat)
            EJpi = Exz(Ji, self.Z_trans_mat, self.X_trans_mat)

            # get worker decisions
            pe_star, re_start_pareto, qi_star, _ = self.getWorkerDecisions( 1/rho_grid[ax,:,ax]*EJpi + EW1i, EUi)

            # Update firm value function
            Ji = self.fun_prod[:, ax, :] - w_grid[ax, :, ax] + self.p.beta * (1 - pe_star) * (1 - qi_star) * EJpi
            Ji = .2*Ji + .8*Ji2

            if upper_env:
                Ji = np.maximum(Ji , J1p.eval_at_W1(W1i))

            # Update worker value function
            re_star = re_start_pareto + pe_star * 1/rho_grid[ax,:,ax] * EJpi 
            W1i = self.pref.utility(w_grid)[ax, :, ax] - self.pref.effort_cost(qi_star) + \
                self.p.beta * qi_star * EUi + self.p.beta * (1-qi_star) * ( re_star + EW1i )
            W1i = .2*W1i + .8*W1i2

            # Compute convergence criteria
            error_j = np.power(Ji - Ji2, 2).mean()
            error_w1 = np.power(W1i - W1i2, 2).mean()

            if (ite_num % 25) == 0:
                self.log.debug('[{}][FB] W1= {:2.4e} J= {:2.4e}'.format(ite_num, error_w1, error_j))

            if (np.array([error_w1, error_j]).max() < self.p.tol_full_model
                    and ite_num > 50):
                break

        self.log.debug('[{}][FB][final] W1= {:2.4e} J= {:2.4e}'.format(ite_num, error_w1, error_j))

        # --------- wrapping up the model ---------

        # find rho_j2j
        rho_j2j = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))
        ve_star = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))
        for ix in range(self.p.num_x):
            for iz in range(self.p.num_z):
                ve_star[iz, :, ix] = self.js.ve(ix,  1/rho_grid*EJpi[iz,:,ix] + EW1i[iz,:,ix] )
                rho_j2j[iz, :, ix] = np.interp(ve_star[iz, :, ix], W1i[self.p.z_0, :, ix], rho_grid)

        # find rho_u2e
        rho_u2e = np.zeros(self.p.num_x)
        Pr_u2e  = np.zeros(self.p.num_x)
        for ix in range(self.p.num_x):
            ve = self.js.ve(ix, EUi[ix])
            rho_u2e[ix] = np.interp(ve, W1i[self.p.z_0, :, ix], rho_grid)
            Pr_u2e[ix] = self.js.pe(ix, EUi[ix]) # this does not include the inefficiency of search for employed

        self.Vf_J = Ji
        self.Vf_W1 = W1i
        self.Vf_U = Ui
        self.J1p = J1p

        # policies
        self.rho_j2j = rho_j2j
        self.rho_u2e = rho_u2e
        self.qe_star = qi_star
        self.pe_star = pe_star
        self.ve_star = ve_star

        if plot:
            self.plot()
            plt.show()

        return self

    def solve_fb_eq(self, plot = False):
        """
            Should be ran after the main equilibrium has been solved, 
            this will solve for the first best solution of the firm problem
        """
        update_eq = True

        # we create a wage grid
        w_grid = self.w_grid
        rho_grid = self.rho_grid

        Ui = self.Vf_U
        Ji = self.Vf_J
        W1i = self.Vf_W1
        J1p = self.J1p

        Ji2 = np.zeros_like(Ji)
        W1i2 = np.zeros_like(W1i)
        Ui2 = np.zeros_like(Ui)

        # prepare expectation call
        Exz = oe.contract_expression('avb,az,bx->zvx', W1i.shape, self.Z_trans_mat.shape, self.X_trans_mat.shape)
        Ex = oe.contract_expression('b,bx->x', Ui.shape, self.X_trans_mat.shape)

        ite_num = 0
        error_js = 10

        for ite_num in range(self.p.max_iter):
            # Store temporary value of J
            Ji2[:] = Ji
            Ui2[:] = Ui
            W1i2[:] = W1i

            # we compute the expected value next period by applying the transition rules
            EW1i = Exz(W1i, self.Z_trans_mat, self.X_trans_mat)
            EJpi = Exz(Ji, self.Z_trans_mat, self.X_trans_mat)
            EUi = Ex(Ui, self.X_trans_mat)

            # get worker decisions
            pe_star, re_start_pareto, qi_star, _ = self.getWorkerDecisions( 1/rho_grid[ax,:,ax]*EJpi + EW1i, EUi)

            # Update firm value function
            Ji = self.fun_prod[:, ax, :] - w_grid[ax, :, ax] + self.p.beta * (1 - pe_star) * (1 - qi_star) * EJpi
            Ji = .2*Ji + .8*Ji2

            # Update worker value function
            re_star = re_start_pareto + pe_star * 1/rho_grid[ax,:,ax] * EJpi 
            W1i = self.pref.utility(w_grid)[ax, :, ax] - self.pref.effort_cost(qi_star) + \
                self.p.beta * qi_star * EUi + self.p.beta * (1-qi_star) * ( re_star + EW1i )
            W1i = .2*W1i + .8*W1i2

            # Update present value of unemployment
            if update_eq:
                pu, ru, _, _ = self.getWorkerDecisions(EUi, EUi, employed=False)
                Ui = self.pref.utility_gross(self.unemp_bf) + self.p.beta * (ru + EUi)
                Ui = 0.2*Ui + 0.8*Ui2


            # Compute convergence criteria
            error_j = np.power(Ji - Ji2, 2).mean()
            error_w1 = np.power(W1i - W1i2, 2).mean()
            error_u = np.power(Ui - Ui2, 2).mean()

           # update worker search decisions
            if (ite_num % 10) == 0:
                if update_eq:
                    # -----  check for termination ------
                    if (np.array([error_w1, error_js, error_j,error_u]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break
                    # ------ or update search function parameter using relaxation ------
                    else:
                            P_xv = self.matching_function(Ji[self.p.z_0 - 1, :, :])
                            relax = 1 - np.power(1/(1+np.maximum(0,ite_num-self.p.eq_relax_margin)), self.p.eq_relax_power)
                            error_js = self.js.update(EW1i[self.p.z_0 - 1, :, :], P_xv, type=1, relax=relax)
                else:
                    # -----  check for termination ------
                    if (np.array([error_w1,error_j]).max() < self.p.tol_full_model
                            and ite_num > 50):
                        break

            if (ite_num % 25) == 0:
                self.log.debug('[{}][FB] W1= {:2.4e} J= {:2.4e} U= {:2.4e} S= {:2.4e}'.format(ite_num, error_w1, error_j, error_u, error_js))

        self.log.debug('[{}][FB][final] W1= {:2.4e} J= {:2.4e}'.format(ite_num, error_w1, error_j))

        # --------- wrapping up the model ---------

        # find rho_j2j
        rho_j2j = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))
        ve_star = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))
        for ix in range(self.p.num_x):
            for iz in range(self.p.num_z):
                ve_star[iz, :, ix] = self.js.ve(ix,  EW1i[iz,:,ix])
                rho_j2j[iz, :, ix] = np.interp(ve_star[iz, :, ix], W1i[self.p.z_0, :, ix], rho_grid)

        # find rho_u2e
        rho_u2e = np.zeros(self.p.num_x)
        Pr_u2e  = np.zeros(self.p.num_x)
        for ix in range(self.p.num_x):
            ve = self.js.ve(ix, EUi[ix])
            rho_u2e[ix] = np.interp(ve, W1i[self.p.z_0, :, ix], rho_grid)
            Pr_u2e[ix] = self.js.pe(ix, EUi[ix]) # this does not include the inefficiency of search for employed

        self.Vf_J = Ji
        self.Vf_W1 = W1i
        self.Vf_U = Ui
        self.J1p = J1p
        self.Fl_ce = self.pref.effort_cost(qi_star)
        self.Vf_Vbar = np.array([self.js[x].e0 for x in range(self.p.num_x)])

        # policies
        self.rho_j2j = rho_j2j
        self.rho_u2e = rho_u2e
        self.qe_star = qi_star
        self.pe_star = pe_star
        self.ve_star = ve_star

        if plot:
            self.plot()
            plt.show()

        return self


    def solve_simple_only(self):
        """
        emulates the solve method but only uses the results from the simple model. This requires
        deriving the policies in terms of rhos. On the job though the value is constant over time
        :return:
        """

        Exz = oe.contract_expression('avb,az,bx->zvx', self.Vf_W1.shape, self.Z_trans_mat.shape, self.X_trans_mat.shape)
        Ex  = oe.contract_expression('b,bx->x', self.Vf_U.shape, self.X_trans_mat.shape)

        EW1i = Exz(self.Vf_W1, self.Z_trans_mat, self.X_trans_mat)
        EUi  = Ex(self.Vf_U, self.X_trans_mat)

        # find rho_j2j
        rho_j2j = np.zeros((self.p.num_z, self.p.num_v, self.p.num_x))
        for ix in range(self.p.num_x):
            for iz in range(self.p.num_z):
                ve = self.js.ve(ix,  EW1i[iz,:,ix])
                rho_j2j[iz, :, ix] = np.interp(ve, self.Vf_W1[ self.p.z_0, :, ix], self.rho_grid)

        # find rho_u2e
        rho_u2e = np.zeros((self.p.num_x))
        Pr_u2e  = np.zeros((self.p.num_x))
        for ix in range(self.p.num_x):
            ve = self.js.ve(ix, EUi[ix])
            rho_u2e[ix] = np.interp(ve, self.Vf_W1[self.p.z_0, :, ix], self.rho_grid)
            #Pr_u2e[ix]  = self.js.pe(ix, EUi[ix]) # this does not include the inefficiency of search for employed


        self.rho_j2j    = rho_j2j
        self.rho_u2e    = rho_u2e

        # set a fixed wage policy contract, 0 search and optimal quite policy
        for ix in range(self.p.num_x):
            for iz in range(self.p.num_z):
                self.rho_star[iz,:,ix] = self.rho_grid
        self.qe_star = self.simple_model.Pr_e2u

        return self

    def find_wstar(self):
        """
        finds the V associated with EJ equal to 0
        :return:
        """

        # this requires finding the rho that makes the EJ equal to 0.
        pass


    def plot(self):
        nrows = 3
        ncols = 4
        fig = plt.figure(figsize=(16, 12))

        plt.subplot(nrows, ncols, 2)
        for x in range(0,self.p.num_x,3):
            plt.plot(np.log(self.w_grid), self.Vf_W1[0, :, x])
            plt.plot(np.log(self.w_grid), self.Vf_W1[self.p.num_z - 1, :, x])
        plt.title('W1')

        plt.subplot(nrows, ncols, 3)
        for x in range(0,self.p.num_x,3):
            plt.plot(np.log(self.w_grid), self.Vf_J[0, :, x])
            plt.plot(np.log(self.w_grid), self.Vf_J[self.p.num_z - 1, :, x])
        plt.title('J1')

        plt.subplot(nrows, ncols, 4)
        for x in range(0,self.p.num_x,3):
            plt.plot(np.log(self.w_grid), self.Vf_J[0, :, x])
            plt.plot(np.log(self.w_grid), self.Vf_J[self.p.num_z - 1, :, x])
            plt.ylim(bottom=-50)
        plt.title('J1 - with floor')

        # plt.subplot(nrows, ncols, 4)
        # plt.plot(self.Pr_u2e)
        # plt.title('U2E')

        plt.subplot(nrows, ncols, 5)
        for ix0 in range(int(self.p.num_x/self.p.num_np)):
            plt.plot(self.Vf_U[ ix0*self.p.num_np : (ix0+1)*self.p.num_np])
            plt.plot(self.Vf_U[ix0 * self.p.num_np: (ix0 + 1) * self.p.num_np],'o')
        plt.title('U(X)')

        plt.subplot(nrows, ncols, 6)
        for x in range(0,self.p.num_x,3):
            plt.plot(np.log(self.w_grid),self.qe_star[self.p.num_z - 1,:,x])
            plt.plot(np.log(self.w_grid),self.qe_star[0, :, x])
        plt.title('E2U')

        plt.subplot(nrows, ncols, 7)
        for ix0 in range(int(self.p.num_x/self.p.num_np)):
            plt.plot(self.Pr_u2e[ ix0*self.p.num_np : (ix0+1)*self.p.num_np])
            plt.plot(self.Pr_u2e[ix0 * self.p.num_np: (ix0 + 1) * self.p.num_np ],'o')
        plt.title('U2E')

        plt.subplot(nrows, ncols, 8)
        for x in range(0,self.p.num_x,3):
            plt.plot(np.log(self.w_grid),self.Fl_ce[self.p.num_z - 1, :, x])
            plt.plot(np.log(self.w_grid),self.Fl_ce[0, :, x])
        plt.title('Cost of effort')

        # plt.subplot(nrows, ncols, 7)
        # for x in range(0,self.p.num_x,3):
        #     plt.plot(np.log(self.Gr_wage),self.Vf_ve[self.p.num_z - 1,:,x])
        #     plt.plot(np.log(self.Gr_wage),self.Vf_ve[0, :, x])
        # plt.title('Ve (v applied to by employed')

        plt.subplot(nrows, ncols, 9)
        for x in range(0,self.p.num_x,3):
            plt.plot(np.log(self.w_grid), self.pe_star[0, :, x])
            plt.plot(np.log(self.w_grid), self.pe_star[0, :, x],'o')
            plt.plot(np.log(self.w_grid), self.pe_star[self.p.num_z - 1, :, x])
            plt.plot(np.log(self.w_grid), self.pe_star[self.p.num_z - 1, :, x],'o')
        plt.title('J2J')

        plt.subplot(nrows, ncols, 10)
        for x in range(0,self.p.num_x,3):
            plt.plot( self.Vf_W1[0, :, x], self.pe_star[0, :, x])
            plt.plot( self.Vf_W1[self.p.num_z - 1, :, x], self.pe_star[self.p.num_z - 1, :, x])
        plt.title('J2J - x:W1')

        plt.subplot(nrows, ncols, 11)
        plt.plot(self.Vf_Vbar,"r")
        plt.plot([self.Vf_W1[:,:,x].max() for x in range(self.p.num_x)],"g")
        plt.plot([self.Vf_W1[:,:,x].min() for x in range(self.p.num_x)],"g")
        plt.plot(self.Vf_U,"b")
        plt.title('All values')

        plt.subplot(nrows, ncols, 12)
        for x in range(0,self.p.num_x,3):
            plt.plot(self.Vf_W1[0, :, x],self.Vf_J[0, :, x])
            plt.plot(self.Vf_W1[self.p.num_z - 1, :, x],self.Vf_J[self.p.num_z - 1, :, x])
            plt.ylim(bottom=-200)
        plt.title('Pareto frontier')

        return(fig)

    def plot_fit_J(self, J1p, Ji, W1i):
        Ji_hat = J1p.eval_at_W1(W1i)
        plt.plot(Ji.flatten(), Ji_hat.flatten(), 'o')
        plt.show()

    def get_prod_epv(self):
        """
        compute the sum of discounted output associated with a each z,x
        """

        p = self.p
        # computing present value, put probability 1 on (x,z)
        EPV = np.zeros((p.num_z,p.num_x))
        EPV[:] = self.fun_prod / self.p.beta

        # apply transition probabilities
        Ezx = oe.contract_expression('zx,az,bx->ab', EPV.shape, self.Z_trans_mat.shape, self.X_trans_mat.shape)

        for t in range(2000):
            EPV2 = self.fun_prod + p.beta * Ezx(EPV, self.Z_trans_mat, self.X_trans_mat)
            dist = np.sum( (EPV - EPV2)**2 )
            EPV[:] = EPV2
            if dist<1e-15:
                break

        return(EPV) 

    def get_transfer_epv(self, type="wage", force_ee=False):
        """
        compute the sum of discounted transfers with a each z,x
        can use wage or output
        """

        p = self.p        # computing present value, put probability 1 on (x,z)

        Wn0  = np.zeros_like(self.Vf_U)
        Wn1  = np.zeros_like(self.Vf_W1)
        Wn1_next = np.zeros_like(self.Vf_W1)
        Wn1_u2e = np.zeros_like(self.Vf_U)
        Wn1_j2j = np.zeros_like(self.Vf_W1)
        ax = np.newaxis

        transfer  = np.zeros_like(self.Vf_W1)
        if type=="wage":
            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):
                    transfer[iz,:,ix] = self.w_grid
        elif type=="output":
            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):
                    transfer[iz,:,ix] = self.fun_prod[iz, ix]
        else:
            raise ValueError("type {} not implemented".format(type))

        allow_move = 1.0 - 1.0*force_ee

        Exz = oe.contract_expression('avb,az,bx->zvx', Wn1.shape, self.Z_trans_mat.shape, self.X_trans_mat.shape)
        Ex = oe.contract_expression('b,bx->x', Wn0.shape, self.X_trans_mat.shape)

        for r in range(1000):

            EWn0 = Ex(Wn0, self.X_trans_mat)
            # compute expected values at rho_star
            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):
                    Wn1_next[iz,:,ix] = np.interp( self.rho_star[iz,:,ix], self.rho_grid, Wn1[iz,:,ix])

            EWn1 = Exz(Wn1_next, self.Z_trans_mat, self.X_trans_mat)

            # compute values on transitions
            for ix in range(self.p.num_x):
                Wn1_u2e[ix] = np.interp( self.rho_u2e[ix], self.rho_grid, Wn1[self.p.z_0-1, :, ix ])

            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):
                    Wn1_j2j[iz,:,ix] = np.interp( self.rho_j2j[iz,:,ix], self.rho_grid, Wn1[self.p.z_0-1, :, ix ])

            Wn0_bis = p.u_bf_m + \
                    p.beta * allow_move * self.Pr_u2e * Wn1_u2e + \
                    p.beta * (1 - allow_move * self.Pr_u2e) * EWn0

            Wn1_bis = transfer + \
                    p.beta * allow_move * self.qe_star * EWn0 + \
                    p.beta * (1-allow_move * self.qe_star) * allow_move * self.pe_star * Wn1_j2j + \
                    p.beta * (1-allow_move * self.qe_star) * (1-allow_move * self.pe_star) * EWn1 \

            dist = np.sum( (Wn1_bis - Wn1)**2 )
            Wn1[:] = Wn1_bis
            Wn0[:] = Wn0_bis            
            
            if dist<1e-15:
                break

        return(Wn1) 
