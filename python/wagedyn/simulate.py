""" 
    Simulates panel data from the model    
"""

import numpy as np
import logging
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from tqdm import tqdm
import gc

def bool_index_combine(I,B):
    """ returns an index where elements of I have been updated using B
    I,B are boolean, and len(B)==I.sum() """
    I2 = np.copy(I)
    I2[I]=B
    return I2


class Event:
    uu  = 0
    ee  = 1
    u2e = 2
    e2u = 3
    j2j = 4


def create_year_lag(df,colnames,lag):
    """ the table should be index by i,year
    """
    # prepare names
    if lag>0:
        s = "_l" + str(lag)
    else:
        s = "_f" + str(-lag)

    values = [n + s for n in colnames]
    rename = dict(zip(colnames, values))

    # create lags
    dlag = df.reset_index() \
             .assign(year=lambda d: d['year'] + lag) \
             .rename(columns=rename)[['i','year'] + values] \
             .set_index(['i','year'])

    # join and return
    return(df.join(dlag))


def create_lag_i(df,time_col,colnames,lag):
    """ the table should be index by i,year
    """
    # prepare names
    if lag>0:
        s = "_l" + str(lag)
    else:
        s = "_f" + str(-lag)

    values = [n + s for n in colnames]
    rename = dict(zip(colnames, values))

    # create lags
    dlag = df.reset_index() \
             .assign(t=lambda d: d[time_col] + lag) \
             .rename(columns=rename)[['i',time_col] + values] \
             .set_index(['i',time_col])

    # join and return
    return(df.join(dlag))


def create_lag(df,time_col,colnames,lag):
    """ the table should be index by i,year
    """
    # prepare names
    if lag>0:
        s = "_l" + str(lag)
    else:
        s = "_f" + str(-lag)

    values = [n + s for n in colnames]
    rename = dict(zip(colnames, values))
    assign_arg = {time_col : lambda d: d[time_col] + lag}

    # create lags
    dlag = df.reset_index() \
             .assign(**assign_arg) \
             .rename(columns=rename)[[time_col] + values] \
             .set_index([time_col])

    # join and return
    return(df.join(dlag))


class Simulator:
    """
    Simulates data from the model and computes moments on simulated data.
    """

    def __init__(self,model,p):
        self.sdata = pd.DataFrame()
        self.model = model
        self.p = p
        self.moments = {}
        self.Zhist = np.zeros((p.num_z,p.sim_nh),dtype=int)
        self.log = logging.getLogger('Simulator')
        self.log.setLevel(logging.INFO)

    def simulate(self,redraw_zhist=True,ignore=[]):
        return(self.simulate_val(
            self.p.sim_ni, 
            self.p.sim_nt_burn + self.p.sim_nt, 
            self.p.sim_nt_burn,
            self.p.sim_nh,
            redraw_zhist=redraw_zhist,
            ignore=ignore))

    def simulate_val(self,ni=int(1e4),nt=40,burn=20,nl=100,redraw_zhist=True,ignore=[]):
        """ we simulate a panel using a solved model

            ni (1e4) : number of individuals
            nt (20)  : number of time period
            nl (100) : length of the firm shock history

            returns a data.frame available at self.sdata with the following columns:
                i: worker id
                t: time
                e: employment status
                h: firm history (where in the common history of shocks)
                x: worker productivity
                z: match productivity
                r: value of rho
                d: even associated with current period
                w: wage
                y: firm present value
                s: tenure at current firm

             the timing is such that the event is what leads to the current state, so the current wage reflects
             the current productivity, current event realized, and rho/wage has been updated. In other words, the event
             happens at the begining of the period, hence U2E are associated with a wage, but E2U are not.

                1) even_t realizes with new firm if necessary
                2) X,Z are drawn
                3) wage is evaluated
        """

        model = self.model
        p     = self.p

        # prepare the ignore shocks
        INCLUDE_E2U = not ('e2u' in ignore)
        INCLUDE_J2J = not ('j2j' in ignore)
        INCLUDE_XCHG = not ('xshock' in ignore)
        INCLUDE_ZCHG = not ('zshock' in ignore)
        INCLUDE_WERR = not ('werr' in ignore)

        # we store the current state into an array
        X  = np.zeros(ni,dtype=int)  # current value of the X shock
        Z  = np.zeros(ni,dtype=int)  # current value of the Z shock
        R  = np.zeros(ni)            # current value of rho
        E  = np.zeros(ni,dtype=int)  # employment status (0 for unemployed, 1 for employed)
        H  = np.zeros(ni,dtype=int)  # location in the firm shock history (so that workers share common histories)
        D  = np.zeros(ni,dtype=int)  # event
        W  = np.zeros(ni)            # log-wage
        P  = np.zeros(ni)            # firm profit
        S  = np.zeros(ni,dtype=int)  # number of periods in current spell
        pr = np.zeros(ni)            # probability, either u2e or e2u

        # we create a long sequence of firm innovation shocks where we store
        # a sequence of realized Z, we store realized Z_t+1 | Z_t for each
        # value of Z_t.
        if (redraw_zhist):
            Zhist = np.zeros((p.num_z,nl),dtype=int)
            for i in range(1,nl):
                # at each time we draw a uniform shock
                u = np.random.uniform(0,1,1)
                # for each value of Z we find the draw given that shock
                for z in range(p.num_z):
                   Zhist[z,i] = np.argmax( model.Z_trans_mat[ z , : ].cumsum() >= u  )
            self.Zhist = Zhist

        # we initialize worker types
        X = np.random.choice(range(p.num_x),ni)

        df_all = pd.DataFrame()
        # looping over time
        for t in range(nt):

            # save the state when starting the period
            E0 = np.copy(E)
            Z0 = np.copy(Z)

            # first we look at the unemployed of a given type X
            for ix in range(p.num_x):
                Ix = (E0==0) & (X==ix)

                if Ix.sum() == 0: continue

                # get whether match a firm
                meet_u2e = np.random.binomial(1, model.Pr_u2e[ix], Ix.sum())==1
                pr[Ix] = model.Pr_u2e[ix]

                # workers finding a job
                Ix_u2e     = bool_index_combine(Ix,meet_u2e)
                H[Ix_u2e]  = np.random.choice(nl, Ix_u2e.sum()) # draw a random location in the shock history
                E[Ix_u2e]  = 1                                  # make the worker employed
                R[Ix_u2e]  = model.rho_u2e[ix]                  # find the firm and the initial rho
                Z[Ix_u2e]  = p.z_0-1                            # starting z_0 for new matches
                D[Ix_u2e]  = Event.u2e
                W[Ix_u2e]  = np.interp(R[Ix_u2e], model.rho_grid, np.log(model.w_grid))  # interpolate wage
                P[Ix_u2e]  = np.interp(R[Ix_u2e], model.rho_grid, model.Vf_J[p.z_0-1,:,ix])  # interpolate wage
                S[Ix_u2e]  = 1

                # workers not finding a job
                Ix_u2u     = bool_index_combine(Ix,~meet_u2e)
                E[Ix_u2u]  = 0           # make the worker unemployed
                W[Ix_u2u]  = 0           # no wage
                D[Ix_u2u]  = Event.uu
                H[Ix_u2u]  = -1
                S[Ix_u2u]  = S[Ix_u2u] + 1 # increase spell of unemployment
                R[Ix_u2u]  = 0
                S[Ix_u2u]  = 0

            # next we look at employed workers of type X,Z
            for ix in range(p.num_x):
                for iz in range(p.num_z):
                    Ixz = (E0 == 1) & (X == ix) & (Z0 == iz)

                    if Ixz.sum() == 0: continue

                    # we check the probability to separate
                    pr_sep  = np.interp( R[Ixz], model.rho_grid , model.qe_star[iz,:,ix])
                    sep     = INCLUDE_E2U * np.random.binomial(1, pr_sep, Ixz.sum() )==1
                    pr[Ixz] = pr_sep

                    # workers who quit
                    Ix_e2u      = bool_index_combine(Ixz,sep)
                    E[Ix_e2u]   = 0
                    D[Ix_e2u]   = Event.e2u
                    W[Ix_e2u]   = 0  # no wage
                    H[Ix_e2u]   = -1
                    S[Ix_e2u]   = 1
                    R[Ix_e2u]   = 0

                    # search decision for non-quiters
                    Ixz     = bool_index_combine(Ixz,~sep)
                    pr_meet = INCLUDE_J2J * np.interp( R[Ixz], model.rho_grid , model.pe_star[iz,:,ix])
                    meet    = np.random.binomial(1, pr_meet, Ixz.sum() )==1

                    # workers with j2j
                    Ixz_j2j      = bool_index_combine(Ixz,meet)
                    H[Ixz_j2j]   = np.random.choice(nl, Ixz_j2j.sum()) # draw a random location in the shock history
                    R[Ixz_j2j]   = np.interp(R[Ixz_j2j], model.rho_grid, model.rho_j2j[iz,:,ix]) # find the rho that delivers the v2 applied to
                    if INCLUDE_ZCHG:
                        Z[Ixz_j2j]   = p.z_0-1                        # starting z_0 for new matches
                    else:
                        Z[Ixz_j2j]   = np.random.choice(range(p.num_z),Ixz_j2j.sum()) # this is for counterfactual simulations
                    D[Ixz_j2j]   = Event.j2j
                    W[Ixz_j2j]   = np.interp(R[Ixz_j2j], model.rho_grid, np.log(model.w_grid)) # interpolate wage
                    P[Ixz_j2j]   = np.interp(R[Ixz_j2j], model.rho_grid, model.Vf_J[iz, :, ix])  # interpolate wage
                    S[Ixz_j2j]   = 1

                    # workers with ee
                    Ixz_ee      = bool_index_combine(Ixz,~meet)
                    R[Ixz_ee]   = np.interp(R[Ixz_ee], model.rho_grid, model.rho_star[iz,:,ix]) # find the rho using law of motion
                    if INCLUDE_ZCHG:
                        Z[Ixz_ee]   = Zhist[ (Z[Ixz_ee] , H[Ixz_ee]) ] # extract the next Z from the pre-computed histories
                    H[Ixz_ee]   = (H[Ixz_ee] + 1) % nl             # increment the history by 1
                    D[Ixz_ee]   = Event.ee
                    W[Ixz_ee]   = np.interp(R[Ixz_ee], model.rho_grid, np.log(model.w_grid))  # interpolate wage
                    P[Ixz_ee]   = np.interp(R[Ixz_ee], model.rho_grid, model.Vf_J[iz, :, ix])  # interpolate firm Expected profit @fixme this done at past X not new X
                    S[Ixz_ee]   = S[Ixz_ee] + 1

            # we shock the type of the worker
            for ix in range(p.num_x):
                Ix    = (X==ix)
                if INCLUDE_XCHG:
                    X[Ix] = np.random.choice(p.num_x, Ix.sum(), p=model.X_trans_mat[:,ix])

            # append to data
            if (t>burn):
                df     = pd.DataFrame({ 'i':range(ni),'t':np.ones(ni) * t, 'e':E, 's':S, 'h':H, 'x':X , 'z':Z, 'r':R, 'd':D, 'w':W , 'Pi':P, 'pr':pr} )
                df_all = pd.concat([df_all, df], axis =0)

        # append match output
        df_all['f'] = model.fun_prod[(df_all.z, df_all.x)]
        df_all.loc[df_all.e==0,'f'] = 0

        # construct a year variable called t4
        df_all['year'] = (df_all['t'] - (df_all['t'] % 4))//4

        # make earnings net of taxes (w is in logs here)
        df_all['w_gross'] = df_all['w']      
        df_all['w_net'] = np.log(self.p.tax_tau) + self.p.tax_lambda * df_all['w']  

        # apply expost tax transform
        df_all['w'] = np.log(self.p.tax_expost_tau) + self.p.tax_expost_lambda * df_all['w']  

        # add log wage measurement error
        # measurement error is outside the model, so we apply it after the taxes
        if INCLUDE_WERR:
            df_all['w'] = df_all['w'] + p.prod_err_w * np.random.normal(size=len(df_all['w']))

        # sort the data
        df_all = df_all.sort_values(['i', 't'])

        self.sdata = df_all
        return(self)

    def simulate_force_ee(self,X0,Z0,H0,R0,nt,update_x=True, update_z=True, pb=False):
        """
        init should give the vector of initial values of X,Z,rho
        we start from this initial value and simulate forward
        one can choose to update x, z using update_z and update_x
        one can choose to show a progress bar with pb=True
        """
        X  = X0.copy() # current value of the X shock
        R  = R0.copy() # current value of rho
        H  = H0.copy() # location in the firm shock history (so that workers share common histories)
        Z  = Z0.copy() # location in the firm shock history (so that workers share common histories)

        ni = len(X)
        W  = np.zeros(ni)     # log-wage
        W1 = np.zeros(ni)     # value to the worker
        Ef = np.zeros(ni)     # effort
        Vs = np.zeros(ni)     # search decision
        tw = np.zeros(ni)     # target wage

        Y = np.zeros(ni)  # log-output
        P = np.zeros(ni)      # firm profit
        pr_sep = np.zeros(ni)  # probability, either u2e or e2u
        pr_j2j = np.zeros(ni)  # probability, either u2e or e2u

        model = self.model
        nl = self.Zhist.shape[1]
        all_df = []

        if pb:
            rr = tqdm(range(nt))
        else:
            rr = range(nt) 

        for t in rr:

            # we store the outcomes at the current state
            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):
                    Ixz_ee = (X == ix) & (Z == iz)
                    if Ixz_ee.sum() == 0: continue

                    Y[Ixz_ee] = np.log(model.fun_prod[iz,ix])
                    pr_sep[Ixz_ee] = np.interp( R[Ixz_ee], model.rho_grid , model.qe_star[iz,:,ix])
                    pr_j2j[Ixz_ee] = np.interp( R[Ixz_ee], model.rho_grid , model.pe_star[iz,:,ix])
                    W[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, np.log(model.w_grid))  # interpolate wage
                    W1[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, model.Vf_W1[iz, :, ix] )  # value to the worker
                    P[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, model.Vf_J[iz, :, ix])  # interpolate firm Expected profit 
                    Vs[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, model.ve_star[iz, :, ix])  # interpolate firm Expected profit 
                    tw[Ixz_ee] = np.log(model.target_w[iz,ix])

            ef = np.log(model.pref.inv_utility(model.pref.effort_cost(pr_sep)))
            all_df.append(pd.DataFrame({ 'i':range(ni),'t':t, 'h':H, 
                'x':X , 'z':Z, 'r':R, 'w':W , 'Pi':P, 
                'pr_e2u':pr_sep, 'pr_j2j':pr_j2j , 'y':Y, 'W1':W1, 'vs':Vs, 
                'target_wage':tw, 'effort': ef }))

            # we update the different shocks
            for ix in range(self.p.num_x):
                for iz in range(self.p.num_z):
                    Ixz_ee = (X == ix) & (Z == iz)
                    if Ixz_ee.sum() == 0: continue
                    R[Ixz_ee] = np.interp(R[Ixz_ee], model.rho_grid, model.rho_star[iz,:,ix]) # find the rho using law of motion

            if update_x:
                for ix in range(self.p.num_x):
                    Ixz_ee = (X == ix) 
                    if Ixz_ee.sum() == 0: continue
                    X[Ixz_ee] = np.random.choice(self.p.num_x, Ixz_ee.sum(), p=model.X_trans_mat[:,ix])

            if update_z:
                for iz in range(self.p.num_z):
                    Ixz_ee = (Z == iz)
                    if Ixz_ee.sum() == 0: continue
                    Z[Ixz_ee] = self.Zhist[ (Z[Ixz_ee] , H[Ixz_ee]) ] # extract the next Z from the pre-computed histories
                    H[Ixz_ee] = (H[Ixz_ee] + 1) % nl                  # increment the history by 1
            
        return pd.concat(all_df).sort_values(['i','t'])

    def get_sdata(self):
        return(self.sdata)

    def get_yearly_data(self):

        sdata = self.sdata

        # compute firm output and sizes at year level
        hdata = (sdata.set_index(['i', 't'])
                      .pipe(create_lag_i, 't', ['d'], -1)
                      .reset_index()
                      .query('h>=0')
                      .assign(c_e2u=lambda d: d.d_f1 == Event.e2u,
                              c_j2j=lambda d: d.d_f1 == Event.j2j)
                      .groupby(['h'])
                      .agg( {'f': 'sum', 'i': "count", 'c_e2u': 'sum', 'c_j2j': 'sum'}))
        hdata['f_year'] = hdata.f + np.roll(hdata.f, -1) + np.roll(hdata.f, -2) + np.roll(hdata.f, -3)
        hdata['c_year'] = hdata.i + np.roll(hdata.i, -1) + np.roll(hdata.i, -2) + np.roll(hdata.i, -3)
        hdata['c_e2u_year'] = hdata.c_e2u + np.roll(hdata.c_e2u, -1) + np.roll(hdata.c_e2u, -2) + np.roll(hdata.c_e2u, -3)
        hdata['c_j2j_year'] = hdata.c_j2j + np.roll(hdata.c_j2j, -1) + np.roll(hdata.c_j2j, -2) + np.roll(hdata.c_j2j, -3)
        hdata['ypw'] = np.log(hdata.f_year/hdata.c_year)
        hdata['lsize'] = np.log(hdata.c_year/4) # log number of worker in the year

        # create year on year growth at the firm level
        hdata['le2u'] = np.log(hdata['c_e2u_year'] / hdata['c_year'])
        hdata['lj2j'] = np.log(hdata['c_j2j_year'] / hdata['c_year'])
        hdata['lsep'] = np.log((hdata['c_j2j_year'] + hdata['c_e2u_year']) / hdata['c_year'])
        hdata = hdata.drop(columns='i')

        # add measurement error to ypw
        hdata_sep = (hdata.assign(ypwe=lambda d: d.ypw + self.p.prod_err_y * np.random.normal(size=len(d.ypw)))
                          .pipe(create_lag, 'h', ['ypw', 'ypwe', 'le2u', 'lj2j', 'lsep'], 4)
                          .assign(dlypw=lambda d: d.ypw - d.ypw_l4,
                                  dlypwe=lambda d: d.ypwe - d.ypwe_l4,
                                  dle2u=lambda d: d.le2u - d.le2u_l4,
                                  dlsep=lambda d: d.lsep - d.lsep_l4,
                                  dlj2j=lambda d: d.lj2j - d.lj2j_l4)[['dlypw', 'dlypwe', 'dle2u', 'dlj2j', 'dlsep', 'c_year']])

        # compute wages at the yearly level, for stayers
        sdata['s2'] = sdata['s']
        sdata['es'] = sdata['e']
        sdata['w_exp'] = np.exp(sdata['w'])

        sdata_y = sdata.groupby(['i', 'year']).agg({'w_exp': 'sum', 'h': 'min', 's': 'min', 's2': 'max', 'e': 'min', 'es': 'sum'})
        sdata_y = sdata_y.pipe(create_year_lag, ['e', 's'], -1).pipe(create_year_lag, ['e', 'es'], 1)
        # make sure we stay in the same spell, and make sure it is employment
        sdata_y = sdata_y.query('h>=0').query('s+3==s2')
        sdata_y['w'] = np.log(sdata_y['w_exp'])

        # attach firm output, compute lags and growth
        sdata_y = (sdata_y.join(hdata.ypw, on="h")
                          .pipe(create_year_lag, ['ypw', 'w', 's', 'h'], 1)
                          .assign(dw=lambda d: d.w - d.w_l1,
                                  dypw=lambda d: d.ypw - d.ypw_l1))


        return(sdata_y)
        
    def computeMoments(self):
        """
        Computes the simulated moments using the simulated data
        :return:
        """
        sdata = self.sdata
        moms = {}
 
        # extract total output
        moms['total_output'] = sdata.query('h>0')['f'].sum()/len(sdata)
        moms['total_wage_gross'] = np.exp(sdata.query('h>0')['w_gross']).sum()/len(sdata)
        moms['total_wage_net'] = np.exp(sdata.query('h>0')['w_net']).sum()/len(sdata)
        moms['total_uben'] = self.p.u_bf_m * sdata.eval('h==0').sum()/len(sdata)

        # ------  transition rates   -------
        # compute unconditional transition probabilities
        moms['pr_u2e'] = sdata.eval('d==@Event.u2e').sum() / sdata.eval('d==@Event.u2e | d==@Event.uu').sum()
        moms['pr_j2j'] = sdata.eval('d==@Event.j2j').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()
        moms['pr_e2u'] = sdata.eval('d==@Event.e2u').sum() / sdata.eval('d==@Event.j2j | d==@Event.ee | d==@Event.e2u').sum()

        # ------  earnings and value added moments at yearly frequency  -------
        # compute firm output and sizes at year level
        hdata = (sdata.set_index(['i', 't'])
                      .pipe(create_lag_i, 't', ['d'], -1)
                      .reset_index()
                      .query('h>=0')
                      .assign(c_e2u=lambda d: d.d_f1 == Event.e2u,
                              c_j2j=lambda d: d.d_f1 == Event.j2j)
                      .groupby(['h'])
                      .agg( {'f': 'sum', 'i': "count", 'c_e2u': 'sum', 'c_j2j': 'sum'}))
        hdata['f_year'] = hdata.f + np.roll(hdata.f, -1) + np.roll(hdata.f, -2) + np.roll(hdata.f, -3)
        hdata['c_year'] = hdata.i + np.roll(hdata.i, -1) + np.roll(hdata.i, -2) + np.roll(hdata.i, -3)
        hdata['c_e2u_year'] = hdata.c_e2u + np.roll(hdata.c_e2u, -1) + np.roll(hdata.c_e2u, -2) + np.roll(hdata.c_e2u, -3)
        hdata['c_j2j_year'] = hdata.c_j2j + np.roll(hdata.c_j2j, -1) + np.roll(hdata.c_j2j, -2) + np.roll(hdata.c_j2j, -3)
        hdata['ypw'] = np.log(hdata.f_year/hdata.c_year)
        hdata['lsize'] = np.log(hdata.c_year/4) # log number of worker in the year

        # create year on year growth at the firm level
        hdata['le2u'] = np.log(hdata['c_e2u_year'] / hdata['c_year'])
        hdata['lj2j'] = np.log(hdata['c_j2j_year'] / hdata['c_year'])
        hdata['lsep'] = np.log((hdata['c_j2j_year'] + hdata['c_e2u_year']) / hdata['c_year'])
        hdata = hdata.drop(columns='i')

        # add measurement error to ypw
        hdata_sep = (hdata.assign(ypwe=lambda d: d.ypw + self.p.prod_err_y * np.random.normal(size=len(d.ypw)))
                          .pipe(create_lag, 'h', ['ypw', 'ypwe', 'le2u', 'lj2j', 'lsep'], 4)
                          .assign(dlypw=lambda d: d.ypw - d.ypw_l4,
                                  dlypwe=lambda d: d.ypwe - d.ypwe_l4,
                                  dle2u=lambda d: d.le2u - d.le2u_l4,
                                  dlsep=lambda d: d.lsep - d.lsep_l4,
                                  dlj2j=lambda d: d.lj2j - d.lj2j_l4)[['dlypw', 'dlypwe', 'dle2u', 'dlj2j', 'dlsep', 'c_year']])

        # covaraince between change in log separation and log value added per worker
        moms['cov_dydsep'] = hdata_sep.cov()['dlypw']['dlsep']

        # moments of the process of value added a the firm level
        cov = hdata_sep.pipe(create_lag, 'h', ['dlypwe'], 4)[['dlypwe', 'dlypwe_l4']].cov()
        moms['var_dy'] = cov['dlypwe']['dlypwe']
        moms['cov_dydy_l4'] = cov['dlypwe']['dlypwe_l4']

        # compute wages at the yearly level, for stayers
        sdata['s2'] = sdata['s']
        sdata['es'] = sdata['e']
        sdata['w_exp'] = np.exp(sdata['w'])

        sdata_y = sdata.groupby(['i', 'year']).agg({'w_exp': 'sum', 'h': 'min', 's': 'min', 's2': 'max', 'e': 'min', 'es': 'sum'})
        sdata_y = sdata_y.pipe(create_year_lag, ['e', 's'], -1).pipe(create_year_lag, ['e', 'es'], 1)
        # make sure we stay in the same spell, and make sure it is employment
        sdata_y = sdata_y.query('h>=0').query('s+3==s2')
        sdata_y['w'] = np.log(sdata_y['w_exp'])

        # attach firm output, compute lags and growth
        sdata_y = (sdata_y.join(hdata.ypw, on="h")
                          .pipe(create_year_lag, ['ypw', 'w', 's'], 1)
                          .assign(dw=lambda d: d.w - d.w_l1,
                                  dypw=lambda d: d.ypw - d.ypw_l1))

        # make sure that workers stays in same firm for 2 periods
        cov = sdata_y.query('s == s_l1 + 4')[['dw', 'dypw']].cov()
        moms['cov_dydw'] = cov['dypw']['dw']

        # Extract 2 U2E trnaistions within individual
        wid_2spells = (sdata_y.query('e_l1<1')
                            .assign(w1=lambda d: d.w, w2=lambda d: d.w, count=lambda d: d.h)
                            .groupby('i')
                            .agg({'count':'count','w1':'first','w2':'last'})
                            .query('count>1'))
        cov = wid_2spells[['w1','w2']].cov()
        moms['var_w_longac'] = cov['w1']['w2']

        cov = sdata_y.pipe(create_year_lag, ['w'], 4)[['w', 'w_l4']].cov()
        moms['var_w'] = sdata_y['w'].var()

        # lag wage growth auto-covariance
        cov = sdata_y.pipe(create_year_lag, ['dw'], 1).pipe(create_year_lag, ['dw'], 2)[['dw', 'dw_l1', 'dw_l2']].cov()
        moms['cov_dwdw_l4'] = cov['dw']['dw_l1']
        moms['cov_dwdw_l8'] = cov['dw']['dw_l2']
        moms['var_dw'] = cov['dw']['dw']

        # compute wage growth J2J and unconditionaly
        sdata_y.query('s == s_l1 + 4')['dw'].mean()
        moms['mean_dw'] = sdata_y['dw'].mean()
        sdata_y.pipe(create_year_lag, ['w'], 2).eval('w - w_l2').mean()

        # compute u2e, ee gap
        moms['w_u2e_ee_gap'] = sdata_y['w'].mean() - sdata_y.query('es_l1==0')['w'].mean()

        # compute wage growth given employer change
        moms['mean_dw_j2j_2'] = (sdata_y
                                    .pipe(create_year_lag, ['w', 'h', 'e'], 2)
                                    .query('e_l2 == 1').query('h_l2 + 8 != h')
                                    .assign(diff=lambda d: d.w - d.w_l2)['diff'].mean())

        del wid_2spells 
        del sdata_y 

        self.moments = moms
        return self

    def clean(self):
        del self.sdata
        gc.collect()

    def compute_growth_var_by_xz(self):
        """ 
        returns wage and match output growth variance for each (x,z) types.

        this function is useful for the coutnerfactual decomposition of wage
        and output growth """

        sdata = self.sdata
        sdata['w_exp'] = np.exp(sdata['w'])
        sdata['s2'] = sdata['s']

        sdata_y = sdata.groupby(['i', 'year']).agg({'w_exp': 'sum', 'f':'sum', 
                    'h':'min', 's': 'min', 's2': 'max', 
                    'e': 'min', 'x':'first', 'z':'first'})
        sdata_y = sdata_y.pipe(create_year_lag, ['e', 's', 'f'], 1)
        #sdata_y = sdata_y.pipe(create_year_lag, ['e', 's', 'f'], 2)
        sdata_y = sdata_y.query('h>=0').query('s+3==s2')
        sdata_y['w'] = np.log(sdata_y['w_exp'])
        sdata_y['lf'] = np.log(sdata_y['f'])
        sdata_y = sdata_y.pipe(create_year_lag, ['w', 'lf'], 1)

        dd = sdata_y.assign( dw = lambda d: d.w - d.w_l1,
                             df = lambda d: d.lf - d.lf_l1
                             ).groupby(['x','z']).agg(
                                 dw_m=('dw','mean'),
                                 dw_v=('dw','var'),
                                 df_m=('df','mean'),
                                 df_v=('df','var'),
                                 e_count=('e','count'))

        return dd 

    def get_moments(self):
        return self.moments

    def simulate_moments_rep(self, nrep):
        """
        simulates moments from the model, running it multiple times
        :param nrep: number of replications
        :return:
        """

        moms = pd.DataFrame()
        self.log.info("Simulating {} reps".format(nrep))
        for i in range(nrep):
            self.log.debug("Simulating rep {}/{}".format(i+1, nrep))
            mom = self.simulate().computeMoments().get_moments()
            moms = pd.concat([ moms, pd.DataFrame({ k:[v] for k,v in mom.items() })] , axis=0)
            self.clean()
        self.log.info("done simulating")
        moms_mean = moms.mean().rename('value_model')
        moms_var = moms.var().rename('value_model_var')

        return(moms_mean, moms_var)
