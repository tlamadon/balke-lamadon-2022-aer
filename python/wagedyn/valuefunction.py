"""
    We implement a power represenation for the value function,
    together with methods to initialize and update it
"""

import numpy as np
from scipy.optimize import minimize,nnls
import matplotlib.pyplot as plt

def curve_fit_search_and_grad(gamma, Xi, Yi, Xmax):
    Xi_arg = (Xmax + np.exp(gamma[3]) - Xi)/ 100.0
    Xi_pow = np.power( Xi_arg , gamma[2])
    Ri     = gamma[0] + gamma[1] * Xi_pow - Yi
    val    = np.power(Ri, 2).mean()

    # the optimizer can handle invalid returns for gradient
    # with np.errstate(divide='ignore'):
    #     with np.errstate(invalid='ignore'):
    g1     = 2 * Ri.mean()
    g2     = 2 * ( Ri * Xi_pow ).mean()
    g3     = 2 * ( Ri * np.log( Xi_arg ) * Xi_pow * gamma[1] ).mean()
    g4     = 2 * ( Ri * gamma[1] * gamma[2] * np.exp(gamma[3]) * np.power( Xi_arg , gamma[2] - 1 ) ).mean()

    return val, np.array([g1,g2,g3,g4])

def curve_fit_search_terms(gamma, Xi, Yi, Xmax):
    Xi_arg = (Xmax + np.exp(gamma[3]) - Xi)/100.0
    Xi_pow = np.power( Xi_arg , gamma[2])
    return Xi_pow,Yi

def curve_eval(gamma, Xi, Xmax):
    Xi_arg = (Xmax + np.exp(gamma[3]) - Xi)/100.0
    Xi_pow = np.power( Xi_arg , gamma[2])
    return gamma[0] + gamma[1] * Xi_pow

class PowerFunctionGrid:
    """ Class that represents the value function using a power function representation.

        The different parameters are stored in gamma_all
        Y =   g0 + g1*(g4 + exp(g3) - X)^g2

        note, instead of using Vmax here, it might be better to use a more stable value, ie the
        actual max value of promisable utility to the worker which is u(infty)/r
        we might then be better of linearly fitting functions of the sort g1 * ( Vmax - X)^ g2 for
        a list of g2.
    """

    def __init__(self,W1,J1,weight=0.01):
        self.num_z, _ , self.num_x = J1.shape
        self.gamma_all = np.zeros( (self.num_z,self.num_x,5) )
        self.rsqr  = np.zeros( (self.num_z,self.num_x))
        self.weight = weight

        # we fit for each (z,x)
        p0 = [0, -1, -1, np.log(0.1)]
        for ix in range(self.num_x):
            for iz in range(self.num_z):
                p0[0] = J1[iz, 0, ix]
                res2 = minimize(curve_fit_search_and_grad, p0, jac=True,
                                options={'gtol': 1e-8, 'disp': False, 'maxiter': 2000},
                                args=(W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max()))
                p0 = res2.x
                self.gamma_all[iz, ix, 0:4] = res2.x
                self.gamma_all[iz, ix, 4]   = W1[iz, :, ix].max()
                self.rsqr[iz, ix] = res2.fun / np.power(J1[iz, :, ix],2).mean()

    def eval_at_zxv(self,z,x,v):
        return curve_eval(self.gamma_all[z,x,0:4],v,self.gamma_all[z,x,4])

    def get_vmax(self,z,x):
        return self.gamma_all[z, x, 4] + np.exp(self.gamma_all[z, x, 3])

    def eval_at_W1(self,W1):
        J1_hat = np.zeros(W1.shape)
        for ix in range(self.num_x):
            for iz in range(self.num_z):
                J1_hat[iz,:,ix] = self.eval_at_zxv(iz,ix,W1[iz,:,ix])
        # make a for loop on x,z
        return(J1_hat)

    def mse(self,W1,J1):
        mse_val = 0

        for ix in range(self.num_x):
            for iz in range(self.num_z):
                val,_ = curve_fit_search_and_grad( self.gamma_all[iz,ix,0:4], W1[iz, :, ix], J1[iz, :, ix], self.gamma_all[iz, ix, 4] )
                mse_val = mse_val + val

        return(mse_val)

    def update(self,W1,J1,lr,nsteps):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        mean_update = 0

        for ix in range(self.num_x):
            for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,ix,0:4], W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max() )
                self.gamma_all[iz, ix, 0:4] = self.gamma_all[iz,ix,0:4] - lr * grad
                self.gamma_all[iz, ix, 4]   = W1[iz, :, ix].max()
                mean_update = mean_update + np.abs(lr * grad).mean()

        return(mean_update/(self.num_x*self.num_z))

    def update_cst(self,W1,J1,lr,nsteps):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0

        for ix in range(self.num_x):
            for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,ix,0:4], W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max() )
                self.gamma_all[iz, ix, 0:2] = self.gamma_all[iz,ix,0:2] - lr * grad[0:2]
                self.gamma_all[iz, ix, 4]   = W1[iz, :, ix].max()
                tot_update_chg += np.abs(lr * grad[0:2]).mean()

        return(tot_update_chg/(self.num_x*self.num_z))

    def update_cst_ls(self,W1,J1):
        """
        Updates the parameters intercept and slope parameters of the representative
        function using lease square. Also stores the highest value to g4.

        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0

        pj_last = np.copy(self.gamma_all)
        for ix in range(self.num_x):
            for iz in range(self.num_z):
                Xi,Yi = curve_fit_search_terms( self.gamma_all[iz,ix,0:4], W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max() )
                # W = np.exp(- self.weight * np.power(Yi,2))
                W = 1.0 * (Yi >= -50)
                W = W / W.sum()
                xbar       = ( Xi * W ).sum()
                ybar       = ( Yi * W ).sum()
                self.gamma_all[iz, ix, 1] = ( (Xi-xbar) * (Yi-ybar) * W ).sum() / (  (Xi-xbar) * (Xi-ybar) * W ).sum()
                self.gamma_all[iz, ix, 0] = ( (Yi - self.gamma_all[iz, ix, 1]* Xi) * W ).sum()
                self.gamma_all[iz, ix, 4]   = W1[iz, :, ix].max()

        rsq = 1 - self.mse(W1,J1)/ np.power(J1,2).sum()
        chg = (np.power(pj_last - self.gamma_all,2).mean(axis=(0,1)) / np.power(pj_last,2).mean(axis=(0,1))).mean()
        return(chg,rsq)

class PowerFunctionGrid2:
    """ Class that represents the value function using a power function representation.

        The different parameters are stored in gamma_all
        Y =   1 - sum_k g0k*(gm - X)^(-g1k)

    """

    def __init__(self,W1,J1,vmax, gpow= np.arange(0.0,20.0,1) ,weight=0.01):
        self.num_z, _ , self.num_x = J1.shape
        self.num_g = len(gpow)
        self.gpow = np.array(gpow) # the sequence of power to use
        self.gamma_all = np.zeros( (self.num_z,self.num_x,self.num_g) )
        self.rsqr = np.zeros( (self.num_z,self.num_x))
        self.weight = weight
        self.vmax = vmax

        # we fit for each (z,x)
        for ix in range(self.num_x):
            for iz in range(self.num_z):

                self.gpow = np.exp( np.arange(-4,4))
                Yi = J1[iz, :, ix]
                # compute the design matrix
                XX1 = - np.power(self.vmax - W1[iz, :, ix][:,np.newaxis] , - self.gpow[np.newaxis,:])

                # constant plus linear
                XX2 = - np.power(W1[iz, :, ix][:,np.newaxis] , np.arange(2)[np.newaxis,:])
                XX2[:,0] = - XX2[:,0]
                XX = np.concatenate([XX1,XX2],axis=1)

                # prepare weights
                W = np.sqrt(1.0 * (Yi >= -50))

                # fit parameters imposing non-negativity
                par,norm = nnls(XX * W[:,np.newaxis], Yi * W)
                rsq = np.power( W * np.matmul(XX,par) , 2).mean() / np.power( W * Yi, 2).mean()

                I = W>0
                plt.plot( W1[iz, I, ix], J1[iz, I, ix],'blue')
                #for k in range(1,len(self.gpow)):
                #    plt.plot( W1[iz, I, ix],  XX[I,k] * par[k],"--")
                plt.plot( W1[iz, I, ix], np.matmul(XX[I,:],par),'red')
                plt.show()
                p0 = 0

                res2 = minimize(curve_fit_search_and_grad, p0, jac=True,
                                options={'gtol': 1e-8, 'disp': False, 'maxiter': 2000},
                                args=(W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max()))
                self.gamma_all[iz, ix, 0:4] = res2.x
                self.rsqr[iz, ix] = 0

    def eval_at_zxv(self,z,x,v):
        return curve_eval(self.gamma_all[z,x,0:4],v,self.gamma_all[z,x,4])

    def eval_at_W1(self,W1):
        J1_hat = np.zeros(W1.shape)
        for ix in range(self.num_x):
            for iz in range(self.num_z):
                J1_hat[iz,:,ix] = self.eval_at_zxv(iz,ix,W1[iz,:,ix])
        # make a for loop on x,z
        return(J1_hat)

    def mse(self,W1,J1):
        mse_val = 0

        for ix in range(self.num_x):
            for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,ix,0:4], W1[iz, :, ix], J1[iz, :, ix], self.gamma_all[iz, ix, 4] )
                mse_val = mse_val + val

        return(mse_val)

    def update(self,W1,J1,lr,nsteps):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        mean_update = 0

        for ix in range(self.num_x):
            for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,ix,0:4], W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max() )
                self.gamma_all[iz, ix, 0:4] = self.gamma_all[iz,ix,0:4] - lr * grad
                self.gamma_all[iz, ix, 4]   = W1[iz, :, ix].max()
                mean_update = mean_update + np.abs(lr * grad).mean()

        return(mean_update/(self.num_x*self.num_z))

    def update_cst(self,W1,J1,lr,nsteps):
        """
        Updates the parameters gamma using nsteps newton steps and lr as the learning rate
        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0

        for ix in range(self.num_x):
            for iz in range(self.num_z):
                val,grad = curve_fit_search_and_grad( self.gamma_all[iz,ix,0:4], W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max() )
                self.gamma_all[iz, ix, 0:2] = self.gamma_all[iz,ix,0:2] - lr * grad[0:2]
                self.gamma_all[iz, ix, 4]   = W1[iz, :, ix].max()
                tot_update_chg += np.abs(lr * grad[0:2]).mean()

        return(tot_update_chg/(self.num_x*self.num_z))

    def update_cst_ls(self,W1,J1):
        """
        Updates the parameters intercept and slope parameters of the representative
        function using lease square. Also stores the highest value to g4.

        :param W1: W1 input values to fit
        :param J1: J1 input values to fit
        :param lr: learning rate
        :param nsteps: number of steps
        :return:
        """
        tot_update_chg = 0

        pj_last = np.copy(self.gamma_all)
        for ix in range(self.num_x):
            for iz in range(self.num_z):
                Xi,Yi = curve_fit_search_terms( self.gamma_all[iz,ix,0:4], W1[iz, :, ix], J1[iz, :, ix], W1[iz, :, ix].max() )
                # W = np.exp(- self.weight * np.power(Yi,2))
                W = 1.0 * (Yi >= -50)
                W = W / W.sum()
                xbar       = ( Xi * W ).sum()
                ybar       = ( Yi * W ).sum()
                self.gamma_all[iz, ix, 1] = ( (Xi-xbar) * (Yi-ybar) * W ).sum() / (  (Xi-xbar) * (Xi-ybar) * W ).sum()
                self.gamma_all[iz, ix, 0] = ( (Yi - self.gamma_all[iz, ix, 1]* Xi) * W ).sum()
                self.gamma_all[iz, ix, 4]   = W1[iz, :, ix].max()

        rsq = 1 - self.mse(W1,J1)/ np.power(J1,2).sum()
        chg = (np.power(pj_last - self.gamma_all,2).mean(axis=(0,1)) / np.power(pj_last,2).mean(axis=(0,1))).mean()
        return(chg,rsq)

