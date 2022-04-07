"""
    Classes to represent search functions when solving the model
"""

import numpy as np
from scipy.optimize import minimize
import logging
import matplotlib.pyplot as plt
#from numba import jit,njit

def curve_fit_search(x, vb , V , P):
    return np.power(x[0] + x[1] * np.power(np.abs(V - vb), x[2]) - P, 2).mean()


def curve_fit_search_and_grad(x, vb , V , P):
    xi_pow = np.power(np.abs(V - vb), x[2])
    val    = np.power(x[0] + x[1] * xi_pow - P, 2).mean()

    g1     = 2*(  x[0] + x[1] * xi_pow - P         ).mean()
    g2     = 2*( (x[0] + x[1] * xi_pow - P)*xi_pow ).mean()
    g3     = 2*( (x[0] + x[1] * xi_pow - P) * np.log(np.abs(V - vb)) * xi_pow * x[1] ).mean()

    return val, np.array([g1,g2,g3])


#@njit(debug=False)
def curve_fit_search2(x, vb , V , P):
    return np.power(x[0] + x[1] * np.power(np.abs(V - vb - np.exp(x[3])), x[2]) - P, 2).mean()


#@njit(debug=False)
def curve_fit_search3(x, vb , V , P, pw):
    """ fixes the power in the function """
    return np.power(x[0] + x[1] * np.power(np.abs(V - vb - np.exp(x[2])), pw) - P, 2).mean()


def curve_fit_search3_and_grad(x, vb , V , P, pw):
    Xi_arg = np.abs(V - vb - np.exp(x[2]))
    Xi_pow = np.power(Xi_arg ,pw)
    Ri     = x[0] + x[1] *  Xi_pow - P
    val    = np.power(Ri, 2).mean()

    g1     = 2 * Ri.mean()
    g2     = 2 * ( Ri * Xi_pow ).mean()
    g3     = 2 * ( Ri * x[1] * x[2] * np.exp(x[2]) * np.power( Xi_arg , pw - 1 ) ).mean()

    return val, np.array([g1,g2,g3])


def other_function(gamma, Xi, Yi, Xmax):
    Xi_arg = Xmax + np.exp(gamma[3]) - Xi
    Xi_pow = np.power( Xi_arg , gamma[2])
    Ri     = gamma[0] + gamma[1] * Xi_pow - Yi
    val    = np.power(Ri, 2).mean()

    g1     = 2 * Ri.mean()
    g2     = 2 * ( Ri * Xi_pow ).mean()
    g3     = 2 * ( Ri * np.log( Xi_arg ) * Xi_pow * gamma[1] ).mean()
    g4     = 2 * ( Ri * gamma[1] * gamma[2] * np.exp(gamma[3]) * np.power( Xi_arg , gamma[2] - 1 ) ).mean()

    return val, np.array([g1,g2,g3,g4])


def curve_fit_search2_and_grad(x, vb , V , P):
    xi_pow = np.power(np.abs(V - vb), x[2])
    val    = np.power(x[0] + x[1] * xi_pow - P, 2).mean()

    g1     = 2*(  x[0] + x[1] * xi_pow - P         ).mean()
    g2     = 2*( (x[0] + x[1] * xi_pow - P)*xi_pow ).mean()
    g3     = 2*( (x[0] + x[1] * xi_pow - P) * np.log(np.abs(V - vb)) * xi_pow * x[1] ).mean()

    return val, np.array([g1,g2,g3])


class JobSearch:
    """
    Class that captures the optimal search decision. Update it using a P,V combination. Then ask for the value
    of the job where to apply and the associated probability for any expected value e.
    """
    def __init__(self):
        self.x = [0,-1,-1]  # parameters of the approximation
        self.e0 = 0         # e value where p crosses 0
        self.e_asy = 0      # asymptotic point in function
        self.input_P = 0
        self.input_V = 0
        self.re_cst = 0
        self.log = logging.getLogger('JobSearch')
        self.log.setLevel(logging.INFO)
        self.update_last = np.zeros(3)
        self.update_cur  = np.zeros(3)
        self.mse = 0.0
        self.rsq = 0.0

    def update(self,V,P,disp=False,type=0,relax=0):

        # we first compute the maximization problem
        nv = P.size
        self.input_Ps = np.zeros(nv)
        self.input_Vs = np.zeros(nv)
        self.input_V  = V
        self.input_P  = P

        for v in range(nv):
            imax = np.argmax(P * (V - V[v]))
            self.input_Ps[v] = P[imax]
            self.input_Vs[v] = V[imax]

        I = self.input_Ps > 0

        self.input_V2 = self.input_V[I]
        self.input_Vs2 = self.input_Vs[I]
        self.input_Ps2 = self.input_Ps[I]

        #self.e_asy =self.input_V2.max() + 10 # we overshoot the intercept to not have to deal with -infinity
        #res = minimize(curve_fit_search_and_grad,  self.x,  jac=True, options={'gtol': 1e-8, 'disp': disp},args= (self.e_asy, self.input_V2, self.input_Ps2))
        #res = minimize(curve_fit_search, [0, -1, -1], options={'gtol': 1e-8, 'disp': disp},args= (self.e_asy, self.input_V2, self.input_Ps2))
        #self.x = res.x

        # to try: implement gradient
        if (type==0):
            res2 = minimize(curve_fit_search2, [0, -1, -1,np.log(5)], options={'gtol': 1e-8, 'disp': disp},
                           args=(self.input_V2.max(), self.input_V2, self.input_Ps2))
            self.x     = res2.x[0:3]
            self.e_asy = self.input_V2.max() + np.exp(res2.x[3])
        elif (type==1):

            if (I.sum()==0):
                self.update_cur[0:2] = 0.0
                self.update_cur[2]   = self.input_V.max() - self.e_asy
            else:
                res2 = minimize(curve_fit_search3, [0, -1,np.log(5)], options={'gtol': 1e-8, 'disp': disp},
                               args=(self.input_V2.max(), self.input_V2, self.input_Ps2,self.x[2]))

                self.update_cur[0:2] = res2.x[0:2]                             - self.x[0:2]
                self.update_cur[2]   = self.input_V2.max() + np.exp(res2.x[2]) - self.e_asy

            self.x[0:2] = self.x[0:2] + (1-relax) * (self.update_cur[0:2] + 0.0 * self.update_last[0:2])
            self.e_asy  = self.e_asy  + (1-relax) * (self.update_cur[2]   + 0.0 * self.update_last[2])

            # save last update for adaptive learning rates
            self.update_last = self.update_cur

        else:
            pass

        # evaluate fit
        if (I.sum() == 0):
            self.mse = 0.0
            self.rsq = 1.0
        else:
            self.mse = np.power(self.input_Ps2 - self.pe(self.input_V2), 2).mean()
            self.rsq = 1.0 - self.mse / np.power(self.input_Ps2, 2).mean()
        self.log.debug("done with optimization")

        self.e0 = -np.power(-self.x[0]/self.x[1], 1/self.x[2]) + self.e_asy
        self.re_cst = 0
        self.re_cst = -self.re(self.e0)

        assert self.e_asy>self.e0, "asymptote point is not larger than 0 crossing"
        assert (self.x[1]<0) & (self.x[2]<0), "parameter of the functional form are decreasing"

    def pe(self,e):
        """
        returns the probability at the optimal choice given a continuation value e
        :param e:
        :return:
        """

        e = np.minimum(e, self.e0)
        pe = self.x[0] + self.x[1] * np.power(np.abs(e - self.e_asy), self.x[2])
        return np.maximum(pe,0.0)

    def pe_deriv(self,e):
        """
        returns the derivative of probability at the optimal choice given a continuation value e
        :param e:
        :return:
        """

        e = np.minimum(e, self.e0)
        pe = self.x[1] *  self.x[2] * np.power(np.abs(e - self.e_asy), self.x[2] -1.0)
        return pe

    def re(self,e):
        """
        returns the extra value due to search p(v)(v-e) at optimal v for given e
        :param e:
        :return:
        """
        e = np.minimum(e, self.e0)
        return self.re_cst - self.x[0]*e + self.x[1]/(1+self.x[2])*np.power( np.abs( e - self.e_asy), 1 + self.x[2])

    def ve(self,e):
        """
        returns the  optimal seasrch choice given a continuation value e
        :param e:
        :return:
        """

        ve = self.re(e)/(1e-5+self.pe(e)) + np.minimum(e,self.e0)
        return np.minimum(ve,self.e0)

    def get_params(self):
        return(np.array([ self.re_cst,  self.x[0], self.x[1] , self.x[2], self.e_asy, self.e0 ]))

class JobSearchArray:
    """
    stores a vector of JobSearch, one for each x
    can solve for the value of being unemployed.
    """

    def __init__(self,nx):
        self.log = logging.getLogger('JobSearch')
        self.log.setLevel(logging.INFO)

        self.nx  = nx
        self.jsa = [ JobSearch() for x in range(nx)]

        self.Vf_U   = np.zeros(nx)   # flow value of unemployment
        self.Vf_vu  = np.zeros(nx)   # value unemployed applies to
        self.Pr_u2e = np.zeros(nx)  # u2e probablity

    def update(self,V,P,type=0,relax=0):
        # store the old parameters
        pp0 = self.get_params()
        if len(V.shape)==1:
            for x in range(self.nx):
                self.jsa[x].update(V,P[:,x],type=type,relax=relax)
        else:
            for x in range(self.nx):
                self.jsa[x].update(V[:,x],P[:,x],type=type,relax=relax)
        pp = self.get_params()
        error = (np.power(pp0 - pp, 2).mean(axis=0) / (1e-6 + np.power(pp0, 2).mean(axis=0)) ).mean()
        return error

    def re(self,x,e):
        return self.jsa[x].re(e)

    def ve(self,x,e):
        return self.jsa[x].ve(e)

    def pe(self,x,e):
        return self.jsa[x].pe(e)

    def pe_deriv(self,x,e):
        return self.jsa[x].pe_deriv(e)

    def plot(self):
        for x in range(self.nx):
            plt.plot(self.jsa[x].input_V2,self.jsa[x].pe(self.jsa[x].input_V2))
        plt.show()

    def solve_search_choice(self,e):
        pe = np.zeros(e.shape)
        re = np.zeros(e.shape)
        if len(e.shape)==3:
            for x in range(self.nx):
                pe[:, :, x] = self.jsa[x].pe(e[:, :, x])
                re[:, :, x] = self.jsa[x].re(e[:, :, x])
        if len(e.shape)==1:
            for x in range(self.nx):
                pe[x] = self.jsa[x].pe(e[x])
                re[x] = self.jsa[x].re(e[x])
        return pe,re

    def solve_U(self, uf, beta, Ex, TX):

        Ui = self.Vf_U
        pu = self.Pr_u2e
        vu = self.Vf_vu

        for iter in range(1000):
            # Update the guess for U(x) given p
            EUi = Ex(Ui, TX)
            for x in range(self.nx):
                pu[x] = self.pe(x, EUi[x])
                vu[x] = self.ve(x, EUi[x])
            assert (pu <= 1).all(), "pu is not less than 1"
            assert (pu >= -1e-10).all(), "pu is not larger than 0"
            Ui2 = uf + beta * pu * vu + beta * (1 - pu) * EUi
            error_u = np.max(abs(Ui - Ui2))
            Ui = Ui2
            if (error_u < 1e-6):
                break

            # self.log.debug('[{}] Error_U = {:2.4e}'.format(iter,error_u))
        self.log.info('[{}] Error_U = {:2.4e}'.format(iter, error_u))

        self.Vf_U = Ui2
        self.Vf_vu = vu
        self.Pr_u2e = pu

    def get_all_U(self):
        return  self.Vf_U, self.Vf_vu, self.Pr_u2e

    def get_params(self):
        params = np.zeros([self.nx,6])
        for x in range(self.nx):
            params[x,:] = self.jsa[x].get_params()
        return(params)

    def __getitem__(self, key):
        return self.jsa[key]

    def mse(self):
        mse = np.array([self.jsa[x].mse for x in range(self.nx)])
        return mse.mean()

    def rsq(self):
        rsq = np.array([self.jsa[x].rsq for x in range(self.nx)])
        return rsq.mean()
