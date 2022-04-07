"""
Set of function to help with the computation of pass-throughs
"""

import opt_einsum as oe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import Counter
import shutil
import itertools

def interp_sort(xn,x,y):
    """ interpolate, but sort values first """
    x_sorted, y_sorted = zip(*sorted(zip(x,y)))
    return(np.interp(xn, x_sorted, y_sorted))


class Passthrough:
    """
    class that computes pass-throughs
    """

    def __init__(self,model):
        """ initialize from a model"""
        self.model = model
        self.p = model.p
        self.prod_epv = model.get_prod_epv()
        self.output_epv = model.get_transfer_epv(type="output")
        self.transfer_epv = model.get_transfer_epv(type="wage")
        self.output_epv_ee = model.get_transfer_epv(type="output", force_ee = True)
        self.transfer_epv_ee = model.get_transfer_epv(type="wage", force_ee = True)

    def get_pt(self,ix,iz,dx,dz,num="dv",den="df"):
        """
        computes pass-through given the model at ix,iz and all values of model_grid for a shock
        described by dz,dz
        """
                
        # change in log consumption equivalent of the value to the worker
        if num=="dv":
            dlc = ( self.model.pref.log_consumption_eq(self.model.Vf_W1[iz+dz,:,ix+dx]) - 
                     self.model.pref.log_consumption_eq(self.model.Vf_W1[iz,:,ix]))
        elif num=="dw-follow":
            dlc = ( np.log(self.transfer_epv[iz+dz,:,ix+dx]) - 
                     np.log(self.transfer_epv[iz,:,ix]) )
        elif num=="df-follow":
            dlc = ( np.log(self.output_epv[iz+dz,:,ix+dx]) - 
                     np.log(self.output_epv[iz,:,ix]) )
        elif num=="dw-ee":
            dlc = ( np.log(self.transfer_epv_ee[iz+dz,:,ix+dx]) - 
                     np.log(self.transfer_epv_ee[iz,:,ix]) )
        elif num=="df-ee":
            dlc = ( np.log(self.output_epv_ee[iz+dz,:,ix+dx]) - 
                     np.log(self.output_epv_ee[iz,:,ix]) )
        else:
            raise ValueError("num with {} not implemented".format(num))

        # measures for the denominator
        if den=="dvmax":
            # get W where the firm makes 0 expected profit
            W_J0  = interp_sort( 0.0, self.model.Vf_J[iz,:,ix], self.model.Vf_W1[iz,:,ix]) # W_star
            # get W where the firm makes 0 expected profit in (z+dz,x+dx)
            W_100 = interp_sort( self.model.Vf_J[iz,:,ix], self.model.Vf_J[iz+dz,:,ix+dx], self.model.Vf_W1[iz+dz,:,ix+dx])
            # extract change
            df = self.model.pref.consumption_eq(W_100) - self.model.pref.consumption_eq(self.model.Vf_W1[iz,:,ix])    
            # scale for the total level, consumption when worker gets everything to get elasticity
            dlf = df/self.model.pref.consumption_eq(W_J0)
        elif den=="df-ee":
            # compute denominator using present value of output
            # note that the (1-beta) would drop out in logs if we transformed it into a flow
            dlf = np.log(self.prod_epv[iz+dz,ix+dx]) - np.log(self.prod_epv[iz,ix])
        elif den=="df-follow":
            # compute denominator using present value of output
            # note that the (1-beta) would drop out in logs if we transformed it into a flow
            dlf = ( np.log(self.output_epv[iz+dz,:,ix+dx]) - 
                     np.log(self.output_epv[iz,:,ix]) )

        else:
            raise ValueError("den with {} not implemented".format(den))
            
        # return the ratio
        return(dlc/dlf, dlc, dlf)

    def get_pt_interp(self,ix,iz,shock_type,rho,num,den):
        """
        computes interpolated passthrough of a shock of type shock_type (dz+.dx+.dz-,dx-) 
        at values ix,iz, and an array of rho values.
        """
        ix2,iz2,dx2,dz2 = self.get_pt_shock(ix,iz,shock_type)        
        lpt,_,_ = self.get_pt(ix2, iz2, dx2, dz2, num=num, den=den)    
        return(interp_sort(rho,self.model.rho_grid,lpt))
        

    def get_pt_shock(self,ix,iz,shock_type):
        """
        constructs different type of shocks: dz+,dz-,dx,+dx-
        importantly this makes sure not to change x0 when computnig x shocks and not
        to get outside of boundaries for both x and z.
        """

        p = self.p

        if shock_type=="dz+":
            if iz == p.num_z-1:
                return(ix,iz-1,0,1) # for the last one, we use a positive shock from below
            return(ix,iz,0,1)
        if shock_type=="dz-":
            if iz == 0:
                return(ix,iz+1,0,-1) # for the last one, we use a positive shock from below
            return(ix,iz,0,-1)
        
        if shock_type=="dx+":
            # we get the decomposition of x into x0 and x1
            _,x1 = self.model.p.get_x_components()
            
            if x1[ix] == x1.max():
                return(ix-1,iz,1,0) # for the last one, we use a positive shock from below
            return(ix,iz,1,0)

        if shock_type=="dx-":
            # we get the decomposition of x into x0 and x1
            _,x1 = self.model.p.get_x_components()
            
            if x1[ix] == x1.min():
                return(ix+1,iz,-1,0) # for the last one, we use a positive shock from below
            return(ix,iz,-1,0)
    
    def get_pt_vec(self,Z,X,R,shock_type,pt_num,pt_den):
        """ function that appends pass-through value to each row in a data set"""
        Pt_vec = np.zeros_like(R)
        p = self.model.p

        for ix in range(p.num_x):
            for iz in range(p.num_z):
                I = (X==ix) & (Z==iz)
                Pt_vec[I] = self.get_pt_interp(ix,iz,shock_type,R[I],num=pt_num,den=pt_den)
                
                #new_x, new_y = zip(*sorted(zip(R[I], Pt_vec[I])))
                #plt.plot(new_x,new_y,label="z={}".format(iz))
        return(Pt_vec)

   
