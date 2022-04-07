"""
Files that generates all results, figures and tables.

Many of the functions invovlve generating parameter files and calling main_model_eval_once.py from the terminal.

This file solely used from within scriptflow.py, please use the scriptflow interface to generate all the results.
"""

import os
from xml.etree.ElementPath import prepare_star

# limit to 1 core/thread - needs to be set before everything else
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os, shutil
import logging.config
import matplotlib.pyplot as plt
import matplotlib

import wagedyn as wd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import wagedyn.pytab as pt
import json
import subprocess

import wagedyn.passthrough as wdpt
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline, UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess as llr

import logging
from collections import Counter
import itertools
import toml

from pathlib import Path
#np.seterr(all='warn')

#logging.config.fileConfig('logging.conf')
logging.basicConfig(level=logging.INFO)

# loading global conf
JMP_CONF = toml.load('../balkelamadon.toml')
PATH_PAPER_FIG = Path("../figures")
PATH_RESULTS   = Path("../results")
PATH_PAPER_FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------
#    Utility functions 
# ---------------------------

def smooth_cross(x,y, lb=.001, ub=.1, verbose=False):
    """ function which smooths using cross-validation """

    bw_grid = np.linspace(lb, ub,100)
    n = len(x)
    I = np.arange(n)

    mse_lo = np.zeros(100)
    for i in I: 
        llr = UnivariateSpline(x[I != i], y[I != i])

        for k in range(100):
            llr.set_smoothing_factor(bw_grid[k])
            mse_lo[k] += np.power( y[i] - llr( x[i] ),2.0)

    bw_star = bw_grid[np.argmin(mse_lo)]
    llr = UnivariateSpline(x, y)
    llr.set_smoothing_factor(bw_star)

    if verbose:
        print("using bw={} out of [{},{}]".format(bw_star,bw_grid.min(),bw_grid.max()))
    else:
        logging.debug("using bw={} out of [{},{}]".format(bw_star,bw_grid.min(),bw_grid.max()))
    return llr 


from numpy.polynomial import Chebyshev as cheby

def policy_interp(v,x,y):
    c = cheby.fit(x, y, deg=1)
    return(c(v))

# ---------------------------------------
#    Model solutions at main parameters 
# ---------------------------------------

def save_model_solution(param_file):
    all_res = json.load(open(param_file))
    pdict = all_res

    p = wd.Parameters(pdict)
    # solve the model
    model = wd.FullModel(p).solve(plot=False)
    model.save("res_main_model.pkl")

def save_model_fit():

    model     = wd.FullModel.load("res_main_model.pkl")
    np.random.seed(JMP_CONF['seeds']['model_fit'])

    p = model.p
    sim = wd.Simulator(model, p)    
    moms_mean, moms_var = sim.simulate_moments_rep(p.sim_nrep)
    dmoms = wd.get_data_moments(filename = "../" + JMP_CONF['main']['moment_file'] ).join(moms_mean).join(moms_var)
    dmoms.to_csv('../results/res_main_fit.csv')

    # create the table
    table_model_fit()

def save_model_solution_fb():
    model_fb = wd.FullModel.load("res_main_model.pkl").solve_fb(upper_env=False)
    model_fb.save("res_main_model_fb.pkl")


# ---------------------------------------
#         BRINGING MODEL TO LIFE
# ---------------------------------------
def cf_model_to_life(first_best, update_prod=False, pr_cache=False):
    """
    We simulate the response of several variables to a shock to z and x.
    We fixed the cross-section distribution of (X,Z) and set rho to rho_start
    We apply a permanent shock to either X or Z, and fix the employment relationship, as well as (X,Z)
    We then simulate forward the Rho, and the wage, and report several different variable of interest.
    """

    nt = 20*4
    np.random.seed(JMP_CONF['seeds']['model_to_life'])

    # we load the model
    model = wd.FullModel.load("res_main_model.pkl")
    p = model.p
    p.tax_expost_tau = p.tax_tau
    p.tax_expost_lambda = p.tax_lambda

    # we simulate from the model to get a cross-section
    sim = wd.Simulator(model, p)
    sdata = sim.simulate().get_sdata()

    # we construct the different starting values
    tm = sdata['t'].max()
    d0 = sdata.query('e==1 & t==@tm')[['x','z','h','r']]

    # we start at target rho
    R0 = model.target_rho[ (d0['z'],d0['x']) ]

    # starting with X and Z shocks
    def get_z_pos(pr):
        Z1_pos = np.minimum(sdata['z'].max(), d0['z'] + 1)
        Z1_pos = np.where(np.random.uniform(size=len(Z1_pos)) > pr, Z1_pos, d0['z']  )
        return(Z1_pos)

    def get_z_neg(pr):
        Z1_neg = np.maximum(0, d0['z'] - 1)
        Z1_neg = np.where(np.random.uniform(size=len(Z1_neg)) > pr, Z1_neg, d0['z']  )
        return(Z1_neg)

    def get_x_pos(pr):    
        Xtrans_pos = np.array([1,2,3,4,4,6,7,8,9,9,11,12,13,14,14],int)
        X1_pos = Xtrans_pos[d0['x']]
        X1_pos = np.where(np.random.uniform(size=len(X1_pos)) > pr, X1_pos, d0['x']  )
        return(X1_pos)

    def get_x_neg(pr):    
        Xtrans_neg = np.array([0,0,1,2,3, 5,5,6,7,8, 10,10,11,12,13],int)
        X1_neg = Xtrans_neg[d0['x']]
        X1_neg = np.where( np.random.uniform(size=len(X1_neg)) > pr, X1_neg, d0['x']  )
        return(X1_neg)

    # simulate a control group
    var_name = {'x':r'worker productivity $x$', 
                'w':r'log earnings $\log w$', 
                'W1':'worker promised value $V$', 
                'lceq':'worker cons. eq.', 
                'Pi':r'firm present value $J(x,z,V)$', 
                'y':r'log match output $\log f(x,z)$', 
                'pr_j2j':'J2J probability', 
                'pr_e2u':'E2U probability',
                'target_wage':r'log of target wage $\log w^*(x,z)$',
                'vs':'worker search decision $v_1$',
                'effort':'effort cost $c(e)$'}
    var_list = { k:'mean' for k in var_name.keys()  }

    def sim_agg(dd):
        # compute consumption equivalent for W1
        dd['lceq'] = model.pref.log_consumption_eq(dd['W1'])
        dd['lpeq'] = model.pref.log_profit_eq(dd['W1'])
        return(dd.groupby('t').agg(var_list))

    if first_best:
        model_fb = wd.FullModel.load("res_main_model_fb.pkl")
        for iz in range(model_fb.p.num_z):
            for ix in range(model_fb.p.num_x):
                model_fb.rho_star[iz,:,ix] = model_fb.rho_grid
        sim.model = model_fb

        # let's find rho_star for the first best model
        I=range(p.num_v)[::-1]
        R0_fb = np.zeros((p.num_z,p.num_x))
        for ix in range(p.num_x):
            for iz in range(p.num_z):
                R0_fb[iz,ix] = np.interp( 0.0, 
                                    model_fb.Vf_J[iz,I,ix], 
                                    model_fb.rho_grid[I])

        R0 = R0_fb[ (d0['z'],d0['x']) ]    

    sdata0 = sim_agg(sim.simulate_force_ee(d0['x'],d0['z'],d0['h'],R0, nt, update_x=False, update_z=False, pb=True))

    # we run for a grid of probabilities
    if pr_cache:
        with open("res_cf_pr_fb{}.json".format(first_best)) as f:
            all = json.load(f)
    else:
        all = []
        vec = np.linspace(0,1,10)
        for i in range(len(vec)):
            logging.info("simulating {}/{}".format(i, len(vec)))
            res = {}
            res['pr'] = vec[i]
            pr = vec[i]
            res['x_pos'] = sim.simulate_force_ee(
                    get_x_pos(pr), d0['z'],d0['h'],R0, nt, 
                    update_x=False, update_z=False, pb=True)['y'].mean() 
            res['x_neg'] = sim.simulate_force_ee(
                    get_x_neg(pr), d0['z'],d0['h'],R0, nt, 
                    update_x=False, update_z=False, pb=True)['y'].mean() 
            res['z_pos'] = sim.simulate_force_ee(
                    d0['x'], get_z_pos(pr), d0['h'],R0, nt, 
                    update_x=False, update_z=False, pb=True)['y'].mean() 
            res['z_neg'] = sim.simulate_force_ee(
                    d0['x'], get_z_neg(pr), d0['h'],R0, nt, 
                    update_x=False, update_z=False, pb=True)['y'].mean() 
            all.append(res)

        # save to file!
        # with open("res_cf_pr_fb{}.json".format(first_best), 'w') as fp:
        #     json.dump(all, fp)
    
    df = pd.DataFrame(all)

    df = df.sort_values(['x_pos'])
    pr_x_pos = np.interp( sdata0['y'].mean() + 0.1, df['x_pos'] , df['pr'] )
    df = df.sort_values(['x_neg'])
    pr_x_neg = np.interp( sdata0['y'].mean() - 0.1, df['x_neg'] , df['pr'] )
    df = df.sort_values(['z_pos'])
    pr_z_pos = np.interp( sdata0['y'].mean() + 0.1, df['z_pos'] , df['pr'] )
    df = df.sort_values(['z_neg'])
    pr_z_neg = np.interp( sdata0['y'].mean() - 0.1, df['z_neg'] , df['pr'] )
    
    logging.info(" chosen probability x pos:{}".format(pr_x_pos))
    logging.info(" chosen probability x neg:{}".format(pr_x_neg))
    logging.info(" chosen probability z pos:{}".format(pr_z_pos))
    logging.info(" chosen probability z neg:{}".format(pr_z_neg))
 
    sdata0 = sim_agg(sim.simulate_force_ee(d0['x'],d0['z'],d0['h'],R0, nt, update_x=update_prod, update_z=update_prod, pb=True))

    # finaly we simulate at the probabilities that we have chosen.
    sdata_x_pos = sim_agg(sim.simulate_force_ee(
        get_x_pos(pr_x_pos),d0['z'],d0['h'],R0, nt, 
        update_x=update_prod, update_z=update_prod,pb=True))

    sdata_x_neg = sim_agg(sim.simulate_force_ee(
        get_x_neg(pr_x_neg),d0['z'],d0['h'],R0, nt, 
        update_x=update_prod, update_z=update_prod,pb=True))

    sdata_z_pos = sim_agg(sim.simulate_force_ee(
        d0['x'],get_z_pos(pr_z_pos),d0['h'],R0, nt, 
        update_x=update_prod, update_z=update_prod,pb=True))

    sdata_z_neg = sim_agg(sim.simulate_force_ee(
        d0['x'],get_z_neg(pr_z_neg),d0['h'],R0, nt, 
        update_x=update_prod, update_z=update_prod,pb=True))

    # preparing the lead and lag plots
    pp0 = lambda v : np.concatenate([ np.zeros(5), v ])
    ppt = lambda v : np.concatenate([ [-4,-3,-2,-1,0], v ])
    
    to_plot = {'w','pr_j2j','pr_e2u','vs','effort','Pi','y','W1','target_wage'}
    to_plot = {k:v for k,v in var_name.items() if k in to_plot}

    # Z shock response
    plt.clf()
    # plt.rcParams["figure.figsize"]=12,12
    plt.figure(figsize=(12, 12), dpi=80)
    for i,name in enumerate(to_plot.keys()):
        plt.subplot(3, 3, i+1)
        plt.plot( ppt (sdata0.index/4) , pp0(sdata_z_pos[name] - sdata0[name]) )
        plt.plot( ppt (sdata0.index/4) , pp0(sdata_z_neg[name] - sdata0[name]), linestyle='--')
        #plt.plot( ppt (sdata0.index/4) , pp0(sdata_z_pos_fb[name] - sdata0[name]), linestyle='dashdot')
        #plt.plot( ppt (dd0.index/4) , pp0(sdata_x_pos[name] - sdata0[name]) )
        #plt.plot( ppt (dd0.index/4) , pp0(sdata_x_neg[name] - sdata0[name]) )
        plt.axhline(0,linestyle=':',color="black")
        plt.xlabel(var_name[name])
        #plt.xlabel('years')
        plt.xticks(range(0,21,5))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-3,5))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if first_best:
        plt.savefig('../figures/figurew6-ir-zshock-fb.pdf', bbox_inches='tight')
    else:
        plt.savefig('../figures/figure4-ir-zshock.pdf', bbox_inches='tight')

    plt.clf()
    # plt.rcParams["figure.figsize"]=12,12
    plt.figure(figsize=(12, 12), dpi=80)
    for i,name in enumerate(to_plot.keys()):
        plt.subplot(3, 3, i+1)
        plt.plot( ppt (sdata0.index/4) , pp0(sdata_x_pos[name] - sdata0[name]) )
        plt.plot( ppt (sdata0.index/4) , pp0(sdata_x_neg[name] - sdata0[name]) ,ls='--')
        #plt.plot( ppt (dd0.index/4) , pp0(sdata_x_pos[name] - sdata0[name]) )
        #plt.plot( ppt (dd0.index/4) , pp0(sdata_x_neg[name] - sdata0[name]) )
        plt.axhline(0,linestyle=':',color="black")
        plt.xlabel(var_name[name])
        plt.xticks(range(0,21,5))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-3,5))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if first_best:
        plt.savefig('../figures/figurew5-ir-xshock-fb.pdf', bbox_inches='tight')
    else:
        plt.savefig('../figures/figure3-ir-xshock.pdf', bbox_inches='tight')


# ---------------------------------------
#         Variance decompositions
# ---------------------------------------

def cf_vardec_growth_rep(seed, filename,burn=200,ni=100000):
    """
    We compute the variance decomposition of growth in the cross-section
    Here we simulate the data forward however we shut down some updates sequentially

    1. transition in and out of unemployment
    2. transition between firms
    3. x1 shock and z shocks 

    In all cases we reweight by the original distribution of (x1,z)
    """

    np.random.seed(seed)

    def getdvar(dd,var):
        ww = dd['e_count']/dd['e_count'].sum() 
        wth = (ww * dd[var + '_v']).sum() 
        mm  = (ww * dd[var + '_m']).sum() 
        btw = (ww * (dd[var + '_m'] - mm)**2 ).sum()
        return btw+wth

    # we load the model
    model = wd.FullModel.load("res_main_model.pkl")
    p = model.p

    # we simulate a long time
    p.sim_nt_burn = burn
    p.sim_ni = ni

    # SIM1: all components are active
    sim = wd.Simulator(model, p)
    dd1 = sim.simulate(ignore=['werr']).compute_growth_var_by_xz()

    # SIM2: shut down E2U
    dd2 = sim.simulate(ignore=['e2u','werr']).compute_growth_var_by_xz()
    dd2 = dd2.join(dd1['e_count'],lsuffix = '_cf')

    # SIM3: shut down E2U
    dd3 = sim.simulate(ignore=['e2u','j2j','werr']).compute_growth_var_by_xz()
    dd3 = dd3.join(dd1['e_count'],lsuffix = '_cf')

    # SIM3: shut down E2U
    dd4 = sim.simulate(ignore=['e2u','j2j','xshock','werr']).compute_growth_var_by_xz()
    dd4 = dd4.join(dd1['e_count'],lsuffix = '_cf')

    # SIM3: shut down E2U
    dd5 = sim.simulate(ignore=['e2u','j2j','zshock','werr']).compute_growth_var_by_xz()
    dd5 = dd5.join(dd1['e_count'],lsuffix = '_cf')

    dd6 = sim.simulate(ignore=['e2u','j2j','zshock','xshock','werr']).compute_growth_var_by_xz()
    dd6 = dd6.join(dd1['e_count'],lsuffix = '_cf')

    res = {}
    res['dw1'] = getdvar(dd1,'dw')
    res['df1'] = getdvar(dd1,'df')
    res['dw2'] = getdvar(dd2,'dw')
    res['df2'] = getdvar(dd2,'df')
    res['dw3'] = getdvar(dd3,'dw')
    res['df3'] = getdvar(dd3,'df')
    res['dw4'] = getdvar(dd4,'dw')
    res['df4'] = getdvar(dd4,'df')
    res['dw5'] = getdvar(dd5,'dw')
    res['df5'] = getdvar(dd5,'df')
    res['dw6'] = getdvar(dd6,'dw')
    res['df6'] = getdvar(dd6,'df')

    with open(filename, 'w') as fp:
        json.dump(res, fp)


def cf_vardec_growth_rep_one(seed, filename,burn=200,ni=20000,reweight=True,include_noise=False):

    np.random.seed(seed)
    def getdvar(dd,var):
        ww = dd['e_count']/dd['e_count'].sum() 
        wth = (ww * dd[var + '_v']).sum() 
        mm  = (ww * dd[var + '_m']).sum() 
        btw = (ww * (dd[var + '_m'] - mm)**2 ).sum()
        return btw+wth

    # we load the model
    model = wd.FullModel.load("res_main_model.pkl")
    p = model.p

    # we simulate a long time
    p.sim_nt_burn = burn
    p.sim_ni = ni

    main_ignore = ['werr']
    if include_noise:
        main_ignore = []

    sim = wd.Simulator(model, p)
    dd0 = sim.simulate(ignore = main_ignore).compute_growth_var_by_xz()

    # we iterate over each possible sequence
    res = {}
    res['dw_all'] = getdvar(dd0,'dw')
    res['df_all'] = getdvar(dd0,'df')

    for var_name in ['e2u','j2j','zshock','xshock']:
        to_ignore = main_ignore + [var_name]
        sim1 = wd.Simulator(model, p)
        dd1 = sim1.simulate(ignore=to_ignore).compute_growth_var_by_xz()
        
        if reweight:
            dd1 = dd1.join(dd0['e_count'],lsuffix = '_cf')

        res['dw_%s' % var_name] = getdvar(dd1,'dw')
        res['df_%s' % var_name] = getdvar(dd1,'df')
        
    with open(filename, 'w') as fp:
        json.dump(res, fp)


def cf_vardec_growth_rep_new(seed, filename,burn=200,ni=20000):
    """
    We compute the variance decomposition of growth in the cross-section
    Here we simulate the data forward however we shut down some updates sequentially

    1. transition in and out of unemployment
    2. transition between firms
    3. x1 shock and z shocks 

    In all cases we reweight by the original distribution of (x1,z)
    """

    reweight = True
    np.random.seed(seed)

    def getdvar(dd,var):
        ww = dd['e_count']/dd['e_count'].sum() 
        wth = (ww * dd[var + '_v']).sum() 
        mm  = (ww * dd[var + '_m']).sum() 
        btw = (ww * (dd[var + '_m'] - mm)**2 ).sum()
        return btw+wth

    # we load the model
    model = wd.FullModel.load("res_main_model.pkl")
    p = model.p

    # we simulate a long time
    p.sim_nt_burn = burn
    p.sim_ni = ni

    sim = wd.Simulator(model, p)
    dd0 = sim.simulate(ignore=['werr']).compute_growth_var_by_xz()

    # we iterate over each possible sequence
    res_all = {}
    for seq in itertools.permutations(['e2u','j2j','zshock','xshock'], 4):
        # we add teh components one by one
        res = {}
        res['dw%i' % 0] = getdvar(dd0,'dw')
        res['df%i' % 0] = getdvar(dd0,'df')
        seq_name = "_".join(seq)
        for sub in range(4):
            to_ignore = list(seq[0:(sub+1)]) + ["werr"]
            sim1 = wd.Simulator(model, p)
            dd1 = sim1.simulate(ignore=to_ignore).compute_growth_var_by_xz()
            if reweight:
                dd1 = dd1.join(dd0['e_count'],lsuffix = '_cf')
            res['dw%i' % (sub+1)] = getdvar(dd1,'dw')
            res['df%i' % (sub+1)] = getdvar(dd1,'df')
            print("done with %s part %i" % (seq_name,sub))
        print("done with %s" % seq_name)
        res_all[seq_name] = res

    with open(filename, 'w') as fp:
        json.dump(res_all, fp)

def cf_simulate_level_var_dec(filename,seed,noise=True, model = None):

    # ---------------  we load the model and simulate data ------------------------------
    if (model==None):
        model = wd.FullModel.load("res_main_model.pkl")

    p = model.p
    np.random.seed(seed)

    if noise == False:
        p.prod_err_w = 0.0
        p.prod_err_y = 0.0
        suffix_str = "_nm"
    else:
        suffix_str = ""

    # we simulate from the model to get a cross-section
    sim = wd.Simulator(model, p)
    sdata = sim.simulate().get_sdata()
    tm = sdata['t'].max()
    sdata = sdata.query('e==1 & t>=(@tm-20)').copy()

    # attach target wage
    sdata['tw'] = np.log(model.target_w[sdata['z'],sdata['x']])

    # construct x0,x1 from x
    x0,x1 = p.get_x_components()

    sdata['x0'] = x0[sdata['x']]
    sdata['x1'] = x1[sdata['x']]

    def mode2(a):
        """ returns second most common value, if only 1 value, returns that"""
        ctr = Counter(a)
        best2 = ctr.most_common(2)
        if (len(best2)==2):
            second_most_common_value, its_frequency = best2[1]
            return(second_most_common_value)
        else:
            second_most_common_value, its_frequency = best2[0]
            return(second_most_common_value)

    # select fully employed at same employer
    stayers = sdata.groupby(['i','year']).agg(
        h_max=('h', 'max'),
        h_min=('h', 'min'),
        s=('h','count'),
        z = ('z',lambda x: pd.Series.mode(x)[0]),  
        x1 = ('x1',lambda x: pd.Series.mode(x)[0]),
        x0 = ('x0',lambda x: pd.Series.mode(x)[0]),
        w  = ('w',lambda x: np.log(np.sum(np.exp(x)))),
        tw  = ('tw',lambda x: np.log(np.sum(np.exp(x)))),
        f  = ('f','sum'),
    )

    # use first and last values
    stayers['z1']  = sdata.groupby(['i','year']).agg(z1=('z',lambda x:x.iloc[0]))
    stayers['z4']  = sdata.groupby(['i','year']).agg(z4=('z',lambda x:x.iloc[len(x)-1]))
    stayers['zh']  = stayers['z1']*p.num_z + stayers['z4']
    stayers['x11'] = sdata.groupby(['i','year']).agg(x11=('x1',lambda x:x.iloc[0]))
    stayers['x14'] = sdata.groupby(['i','year']).agg(x14=('x1',lambda x:x.iloc[len(x)-1]))
    stayers['x1h'] = stayers['x11']*p.num_x + stayers['x14']

    # use first one
    # stayers['zh']  = stayers['z'] 
    # stayers['x1h'] = stayers['x1']
    # # use mode and second most frequent
    # stayers['zb']  = sdata.groupby(['i','year']).agg(zb=('z',mode2))
    # stayers['zh']  = stayers['z']*p.num_z + stayers['zb']
    # stayers['x1b'] = sdata.groupby(['i','year']).agg(x1b=('x1',mode2))
    # stayers['x1h'] = stayers['x1']*p.num_z + stayers['x1b']

    sdata_y = stayers.query("h_max == h_min+3").copy()
    sdata_y['logf'] = np.log(sdata_y['f'])

    # ------------ Compute Outcomes in simulated data ----------------
    for k,var in {"w":'w', 'y':'logf', 'tw':'tw'}.items():
        # ----- WAGE ------
        fit_w = ols( '{} ~ C(x0) + C(x1h) + C(zh)'.format(var), data=sdata_y ).fit() 

        sdata_y['{}_hat_x1'.format(k)] = fit_w.predict(sdata_y.eval('x0=0.0').eval('zh=0.0'))
        sdata_y['{}_hat_z'.format(k)] = fit_w.predict(sdata_y.eval('x0=0.0').eval('x1h=0.0'))
        sdata_y['{}_hat_x0'.format(k)] = fit_w.predict(sdata_y.eval('zh=0.0').eval('x1h=0.0'))
        sdata_y['{}_hat_res'.format(k)] = (sdata_y[var] - sdata_y['{}_hat_x1'.format(k)] - 
                                            sdata_y['{}_hat_z'.format(k)] - sdata_y['{}_hat_x0'.format(k)])

        variable_mean = sdata_y.groupby(['x0','x1h','zh'])['{}_hat_res'.format(k)].agg('mean').rename("{}_hat_nl".format(k))
        sdata_y = pd.merge(sdata_y.reset_index(),variable_mean.reset_index(),on=['x0','x1h','zh'],how="left")

    # ------------ Compute the different decomposition terms ------------
    res = {}
    for k,var in {"w":'w', 'y':'logf', 'tw':'tw'}.items():
        for n in ['x0','x1','z','res','nl']:
            colname = '{}_hat_{}'.format(k,n)
            res['{}_{}_it'.format(k,n)] = sdata_y[colname].var()
            variable_mean = sdata_y.groupby(['i'])[colname].agg('mean').rename("tmp")
            res['{}_{}_i'.format(k,n)]  = pd.merge(sdata_y,variable_mean.reset_index(),on="i",how="left")['tmp'].var()

        res['{}_tot_it'.format(k)] =  sdata_y[var].var()
        variable_mean = sdata_y.groupby(['i'])[var].agg('mean').rename("tmp")
        res['{}_tot_i'.format(k)]  = pd.merge(sdata_y,variable_mean.reset_index(),on="i",how="left")['tmp'].var()

    # for k,v in res.items():    
        # print("{} = {:2.3f}".format(k,v))

    with open(filename, 'w') as fp:
        json.dump(res, fp)
    


def table_variance_decomposition_level():
    noise = 0
    p = Path('.')
    files = list(p.glob('./build/cf/vdec-level-noise%i-rep[0-9]*.json' % noise))
    res_all = []
    for fname in files:
        with open(fname) as f:
            data = json.load(f)
            res_all.append(data)
    res = pd.DataFrame(res_all).mean().to_dict()

    texfile = '../figures/table3-vdec-cs.tex'

    # finally we create the table
    tab = pt.Table().setHeaders(['l'] + ['r'] + 5*['r'])

    tab.append(pt.Row(['', r'total', r'$x_0$', r'$x_1$', r'$z$','other']).setEndSpace(4) )

    tab.append(pt.Row(['',r'\multicolumn{5}{c}{overall}']).setEndSpace(-3))
    tab.addRule([[2,6]])

    STR_PERC = r"${:0.0f}\%$" 
    STR_W_L =r"${:0.2f}$" 

    for s2,s1 in itertools.product(['it','i'],['y','w','tw']):
        res['%s_rem_%s' % (s1,s2)] = ( res['%s_tot_%s' % (s1,s2)] - res['%s_x0_%s' % (s1,s2)] 
                                    -res['%s_x1_%s' % (s1,s2)] -res['%s_z_%s'% (s1,s2)])

    # we start with overall
    tab.append(
        pt.Row([r'match output $f^a_{it}$'])
            .append([ STR_W_L.format(res['y_tot_it'])])
            .append([ 
                #STR_W_L.format(res[n]) + " " + 
                STR_PERC.format(100*res[n]/res['y_tot_it']) 
                    for n in ['y_x0_it','y_x1_it','y_z_it','y_rem_it'] ])
        )
    tab.append(
        pt.Row([r'target wage $w^{*a}_{it}$'])
            .append([ STR_W_L.format(res['tw_tot_it'])])
            .append([ 
                #STR_W_L.format(res[n]) + " " + 
                STR_PERC.format(100*res[n]/res['tw_tot_it']) 
                    for n in ['tw_x0_it','tw_x1_it','tw_z_it','tw_rem_it'] ])
        )
    tab.append(
        pt.Row([r'earnings $w^a_{it}$'])
            .append([ STR_W_L.format(res['w_tot_it'])])
            .append([ 
                #STR_W_L.format(res[n]) + " " + 
                STR_PERC.format(100*res[n]/res['w_tot_it']) 
                    for n in ['w_x0_it','w_x1_it','w_z_it','w_rem_it'] ])
        .setEndSpace(5)
        )

    tab.append(pt.Row(['',r'\multicolumn{5}{c}{within individual, over time}']).setEndSpace(-3))
    tab.addRule([[2,6]])

    tab.append(
        pt.Row([r'match output $f^a_{it}$'])
            .append([ STR_W_L.format(res['y_tot_it'] - res['y_tot_i'])])
            .append([ 
                #STR_W_L.format(res[n]) + " " + 
                STR_PERC.format(100*np.abs(res[n+"t"] - res[n])/res['y_tot_it']) 
                    for n in ['y_x0_i','y_x1_i','y_z_i','y_rem_i'] ])
        )
    tab.append(
        pt.Row([r'target wage $w^{*a}_{it}$'])
            .append([ STR_W_L.format(res['tw_tot_it'] - res['tw_tot_i'])])
            .append([ 
                #STR_W_L.format(res[n]) + " " + 
                STR_PERC.format(100*np.abs(res[n+"t"] - res[n])/res['tw_tot_it']) 
                    for n in ['tw_x0_i','tw_x1_i','tw_z_i','tw_rem_i'] ])
        )
    tab.append(
        pt.Row([r'earnings $w^a_{it}$'])
            .append([ STR_W_L.format(res['w_tot_it'] - res['w_tot_i'])])
            .append([ 
                #STR_W_L.format(res[n]) + " " + 
                STR_PERC.format(100*np.abs(res[n+"t"] - res[n])/res['tw_tot_i']) 
                        for n in ['w_x0_i','w_x1_i','w_z_i','w_rem_i'] ])
        )

    tab.save_to_Tex(texfile)

def table_variance_decomposition_growth():

    args = {"weight":0, "noise":0}

    # load results from files
    data = []
    for i in range(20):
        filename = "build/cf/vdec-growth-noise%i-weight%i-rep%i.json" % (args['weight'], args['noise'],i)
        if os.path.exists(filename):
            with open(filename) as f:
                data.append(json.load(f))
    df = pd.DataFrame(data)
    res = df.mean().to_dict()

    # prepare elements for the table
    tab_res = {}
    shocks_list = ['e2u','j2j','xshock','zshock']

    for var_name in shocks_list:
        tab_res['dw_%s' % var_name] = 100*(res["dw_all"] - res["dw_%s"%var_name])/res["dw_all"]
        tab_res['df_%s' % var_name] = 100*(res["df_all"] - res["df_%s"%var_name])/res["df_all"]

    # finally we create the table
    tab = (pt.Table().setHeaders(['l','r','r','r','r','r'])
        .append( pt.Row(['', 'Total', r'U2E/E2U', r'J2J', r'$x_1$', '$z$'])  )
        .addRule([[2,6]])
        .append( pt.Row([r'$\text{Var}(\Delta\log f^a_{it})$'])
                    .append(  [res["df_all"]] , format="{:0.3f}" )
                    .append(  [ tab_res["df_%s" % k] for k in shocks_list ], format=r"{:.01f}\%")  ) 
        .append( pt.Row([r'$\text{Var}(\Delta\log w^a_{it}) $'])
                    .append( [res["dw_all"]] , format="{:0.3f}" )
                    .append( [ tab_res["dw_%s" % k] for k in shocks_list ], format=r"{:.01f}\%")  ) )
    #print(tab.toTex())

    file_output = '../figures/table4-vdec-growth.tex'
    tab.save_to_Tex(file_output)
    
# ---------------------------------------
#         Policy evaluation
# ---------------------------------------

def cf_policy_gen_neutral(include_meas_error=False):
    """ we fix 2 values of lambda and solve the model 
        for many different values of tau. The goal
        is to extract a revenue neutral policy."""

    model = wd.FullModel.load("res_main_model.pkl")
    p = model.get_parameters()
    p.tol_full_model = 1e-9

    np.random.seed(JMP_CONF['seeds']['policy'])

    if include_meas_error == False:
        p.prod_err_w = 0.0
        p.prod_err_y = 0.0
        suffix_str = "_nm"
    else:
        suffix_str = ""


    out = Path('build') / Path('policy')
    out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # we first vary the structural parameter to
    # get the outcome at the new equilibrium

    # we start with low lambda value
    p.tax_lambda = 0.9
    tau_vec = np.linspace(0.9,1.3,25)
    for i in range(len(tau_vec)):
        p.tax_tau = tau_vec[i]
        p.tax_expost_lambda = 1.0
        p.tax_expost_tau = 1.0
        p.save( out / Path('param_llow_eq_gross_{}{}.json'.format(i,suffix_str)), 
            {'rseed':np.random.randint(1,1e5) })
        
        p.tax_expost_lambda = p.tax_lambda 
        p.tax_expost_tau = p.tax_tau
        p.save(out / Path('param_llow_eq_net_{}{}.json'.format(i,suffix_str)),
            {'rseed':np.random.randint(1,1e5) })

    # we next run the high lambda values 
    p.tax_lambda = 1.1
    tau_vec = np.linspace(0.7,1.1,25)
    for i in range(len(tau_vec)):
        p.tax_tau = tau_vec[i]
        p.tax_expost_lambda = 1.0
        p.tax_expost_tau = 1.0
        p.save(out / Path('param_lhigh_eq_gross_{}{}.json'.format(i,suffix_str)),
            {'rseed':np.random.randint(1,1e5) })

        p.tax_expost_lambda = p.tax_lambda 
        p.tax_expost_tau = p.tax_tau
        p.save(out / Path('param_lhigh_eq_net_{}{}.json'.format(i,suffix_str)),
            {'rseed':np.random.randint(1,1e5) })

    # -----------------------------------------
    # second we vary the ex-post parameters to 
    # get the direct effects

    # reset tax paramters
    p.tax_lambda=1.0
    p.tax_tau=1.0
    p.tax_expost_lambda = 1.0
    p.tax_expost_tau = 1.0

    # we start with low lambda value
    p.tax_expost_lambda = 0.9
    tau_vec = np.linspace(0.9,1.3,25)
    for i in range(len(tau_vec)):
        p.tax_expost_tau = tau_vec[i]
        p.save(out / Path('param_llow_direct_net{}{}.json'.format(i,suffix_str)),
            {'rseed':np.random.randint(1,1e5) })


    tau_vec = np.linspace(0.7,1.1,25) 
    p.tax_expost_lambda = 1.1
    for i in range(len(tau_vec)):
        p.tax_expost_tau = tau_vec[i]
        p.save(out / Path('param_lhigh_direct_net{}{}.json'.format(i,suffix_str)),
            {'rseed':np.random.randint(1,1e5) })

    tau_vec = np.linspace(0.8,1.2,25) 
    p.tax_expost_lambda = 1.0
    for i in range(len(tau_vec)):
        p.tax_expost_tau = tau_vec[i]
        p.save(out / Path('param_lorig_direct_net{}{}.json'.format(i,suffix_str)),
            {'rseed':np.random.randint(1,1e5) })


def cf_policy_collect():
    """ collect of the results from the policy folder"""

    # we collect all output files
    dir = Path("build/policy/")
    files = dir.glob('param_*_moments.pkl')


    all = []
    for f in files:
        rep = {}
        res = pd.read_pickle(f)
        rep['tax_lambda'] = res['parameters']['tax_lambda']
        rep['tax_tau'] = res['parameters']['tax_tau']
        rep['tax_expost_lambda'] = res['parameters']['tax_expost_lambda']
        rep['tax_expost_tau'] = res['parameters']['tax_expost_tau']
        rep['sim_net_earnings'] = res['parameters']['sim_net_earnings']
        rep['var_dw'] = res['moments']['var_dw']
        rep['var_w'] = res['moments']['var_w']
        rep['total_wage_gross'] = res['moments']['total_wage_gross']
        rep['total_wage_net'] = res['moments']['total_wage_net']
        rep['total_uben'] = res['moments']['total_uben']
        rep['var_dy'] = res['moments']['var_dy']
        rep['cov_dydw'] = res['moments']['cov_dydw']
        rep['cov_dwdw_l4'] = res['moments']['cov_dwdw_l4']
        rep['prod_err_y'] = res['parameters']['prod_err_y']
        rep['prod_err_w'] = res['parameters']['prod_err_w']
        all.append(rep)

    df = pd.DataFrame(all).sort_values('tax_tau')
    df['gvt_budget'] = df['total_wage_net'] - df['total_wage_gross'] - rep['total_uben'] 

    baseline_gvt_budget = (df.query("prod_err_w>0").query('tax_lambda==1.0')
            .query('tax_expost_lambda==1.0')
            .sort_values('tax_expost_tau')
            ['gvt_budget'].mean())

    # step 2: find the revenue neutral tau for each lambda
    df1 = df.query('tax_lambda==1.1').query('tax_expost_lambda==1.0')
    tau_neutral_high = np.interp(baseline_gvt_budget, df1['gvt_budget'],df1['tax_tau'])

    df1 = df.query('tax_lambda==0.9').query('tax_expost_lambda==1.0')
    tau_neutral_low = np.interp(baseline_gvt_budget, df1['gvt_budget'],df1['tax_tau'])

    print("tau_high = %f and tau_low = %f " %(tau_neutral_high,tau_neutral_low))


    res = {  }
    for wn in [True, False]:

        if wn:
            suffix_noise = ""
            noise_query = "prod_err_w>0"
        else:
            suffix_noise = "_won"
            noise_query = "prod_err_w<0.00001"

        # baseline
        name = "baseline_{}%s" % suffix_noise
        df1 = df.query('tax_lambda==1.0').query('tax_expost_lambda==1.0').query(noise_query)
        res[name.format("dw")]      = policy_interp(baseline_gvt_budget,  df1['tax_expost_tau'],df1['var_dw'])
        res[name.format("dwdwl1")]  = policy_interp(baseline_gvt_budget,  df1['tax_expost_tau'],df1['cov_dwdw_l4'])
        res[name.format("w")]       = policy_interp(baseline_gvt_budget,  df1['tax_expost_tau'],df1['var_w'])
        res[name.format("perm")]    = res[name.format("dw")] + 2 * res[name.format("dwdwl1")]
        if not wn:
            plt.scatter(df1['tax_expost_tau'],df1['var_dw'])
            plt.axhline([res[name.format("dw")]])

        # direct effect HIGH 
        name = "baseline_{}_direct_high%s" % suffix_noise
        df1 = df.query('tax_lambda==1.0').query('tax_expost_lambda==1.1').query(noise_query) 
        res[name.format("dw")]      = policy_interp(tau_neutral_high,  df1['tax_expost_tau'],df1['var_dw'])
        res[name.format("dwdwl1")]  = policy_interp(tau_neutral_high,  df1['tax_expost_tau'],df1['cov_dwdw_l4'])
        res[name.format("w")]       = policy_interp(tau_neutral_high,  df1['tax_expost_tau'],df1['var_w'])
        res[name.format("perm")]    = res[name.format("dw")] + 2 * res[name.format("dwdwl1")]
        if not wn:
            plt.scatter(df1['tax_expost_tau'],df1['var_dw'])
            plt.axhline([res[name.format("dw")]])
            
        # direct effect LOW 
        name = "baseline_{}_direct_low%s" % suffix_noise
        df1 = df.query('tax_lambda==1.0').query('tax_expost_lambda==0.9').query(noise_query) 
        res[name.format("dw")]      = policy_interp(tau_neutral_low,  df1['tax_expost_tau'],df1['var_dw'])
        res[name.format("dwdwl1")]  = policy_interp(tau_neutral_low,  df1['tax_expost_tau'],df1['cov_dwdw_l4'])
        res[name.format("w")]       = policy_interp(tau_neutral_low,  df1['tax_expost_tau'],df1['var_w'])
        res[name.format("perm")]    = res[name.format("dw")] + 2 * res[name.format("dwdwl1")]

        # new gross
        name = "eq_gross_{}_high%s" % suffix_noise
        df1 = df.query('tax_lambda==1.1').query('tax_expost_lambda==1.0').query(noise_query) 
        res[name.format("dw")]      = np.interp(tau_neutral_high,  df1['tax_expost_tau'],df1['var_dw'])
        res[name.format("dwdwl1")]  = np.interp(tau_neutral_high,  df1['tax_expost_tau'],df1['cov_dwdw_l4'])
        res[name.format("w")]       = np.interp(tau_neutral_high,  df1['tax_expost_tau'],df1['var_w'])
        res[name.format("perm")]    = res[name.format("dw")] + 2 * res[name.format("dwdwl1")]

        name = "eq_gross_{}_low%s" % suffix_noise
        df1 = df.query('tax_lambda==0.9').query('tax_expost_lambda==1.0').query(noise_query) 
        res[name.format("dw")]      = np.interp(tau_neutral_low,  df1['tax_expost_tau'],df1['var_dw'])
        res[name.format("dwdwl1")]  = np.interp(tau_neutral_low,  df1['tax_expost_tau'],df1['cov_dwdw_l4'])
        res[name.format("w")]       = np.interp(tau_neutral_low,  df1['tax_expost_tau'],df1['var_w'])
        res[name.format("perm")]    = res[name.format("dw")] + 2 * res[name.format("dwdwl1")]

        # new net
        name = "eq_net_{}_high%s" % suffix_noise
        df1 = df.query('tax_lambda==1.1').query('tax_expost_lambda==1.1').query(noise_query) 
        res[name.format("dw")]      = policy_interp(tau_neutral_high,  df1['tax_expost_tau'],df1['var_dw'])
        res[name.format("dwdwl1")]  = policy_interp(tau_neutral_high,  df1['tax_expost_tau'],df1['cov_dwdw_l4'])
        res[name.format("w")]       = policy_interp(tau_neutral_high,  df1['tax_expost_tau'],df1['var_w'])
        res[name.format("perm")]    = res[name.format("dw")] + 2 * res[name.format("dwdwl1")]
        if not wn:
            plt.scatter(df1['tax_expost_tau'],df1['var_dw'])
            plt.axhline([res[name.format("dw")]])
            
        name = "eq_net_{}_low%s" % suffix_noise
        df1 = df.query('tax_lambda==0.9').query('tax_expost_lambda==0.9').query(noise_query) 
        res[name.format("dw")]      = policy_interp(tau_neutral_low,  df1['tax_expost_tau'],df1['var_dw'])
        res[name.format("dwdwl1")]  = policy_interp(tau_neutral_low,  df1['tax_expost_tau'],df1['cov_dwdw_l4'])
        res[name.format("w")]       = policy_interp(tau_neutral_low,  df1['tax_expost_tau'],df1['var_w'])
        res[name.format("perm")]    = res[name.format("dw")] + 2 * res[name.format("dwdwl1")]
        if not wn:
            plt.scatter(df1['tax_expost_tau'],df1['var_dw'])
            plt.axhline([res[name.format("dw")]])


    STR_PERC = r"$({:+0.1f}\%)$" 
    STR_W_L =r"${:0.3f}$" 
    # create the table
    # finally we create the table
    tab = (pt.Table().setHeaders(['c','l','l','l','l','l'])
        .append( pt.Row(['', '', r'\multicolumn{2}{c}{baseline contracts}', r'\multicolumn{2}{c}{reoptimized contracts}'])  )
        .append( pt.Row(['','',r'\multicolumn{1}{c}{gross}',r'\multicolumn{1}{c}{net}',r'\multicolumn{1}{c}{gross}',r'\multicolumn{1}{c}{net}']))
        .addRule([[3,4],[5,6]])
        .append( pt.Row([r'More progressive', r'$Var(\log w^a)$'])
                .append( [
                    STR_W_L.format(res['baseline_w_won']),
                    STR_W_L.format(res['baseline_w_direct_low_won']) + " " +
                    STR_PERC.format(100*(res['baseline_w_direct_low_won']-res['baseline_w_won'])/res['baseline_w_won']),
                    STR_W_L.format(res['eq_gross_w_low_won']) + " " +
                    STR_PERC.format(100*(res['eq_gross_w_low_won']-res['baseline_w_won'])/res['baseline_w_won']),
                    STR_W_L.format(res['eq_net_w_low_won']) + " " +
                    STR_PERC.format(100*(res['eq_net_w_low_won']-res['baseline_w_won'])/res['baseline_w_won'])
                ]))
        .append( pt.Row([r'$(\tau_1=0.9,\tau_0={:0.2f})$'.format(tau_neutral_low), r'$Var(\Delta \log w^a)$'])
                    .append( [
                        r"${:0.3f}$".format(res['baseline_dw_won']),
                        r"${:0.3f}$".format(res['baseline_dw_direct_low_won']) + " " +
                        STR_PERC.format(100*(res['baseline_dw_direct_low_won']-res['baseline_dw_won'])/res['baseline_dw_won']),
                        "{:0.3f}".format(res['eq_gross_dw_low_won']) + " " +
                        STR_PERC.format(100*(res['eq_gross_dw_low_won']-res['baseline_dw_won'])/res['baseline_dw_won']),
                        "{:0.3f}".format(res['eq_net_dw_low_won']) + " " +
                        STR_PERC.format(100*(res['eq_net_dw_low_won']-res['baseline_dw_won'])/res['baseline_dw_won'])
                    ]).setEndSpace(8) )
        .append( pt.Row([r'Less progressive', r'$Var(\log w^a)$'])
                .append( [
                    STR_W_L.format(res['baseline_w_won']),
                    STR_W_L.format(res['baseline_w_direct_high_won']) + " " +
                    STR_PERC.format(100*(res['baseline_w_direct_high_won']-res['baseline_w_won'])/res['baseline_w_won']),
                    STR_W_L.format(res['eq_gross_w_high_won']) + " " +
                    STR_PERC.format(100*(res['eq_gross_w_high_won']-res['baseline_w_won'])/res['baseline_w_won']),
                    STR_W_L.format(res['eq_net_w_high_won']) + " " +
                    STR_PERC.format(100*(res['eq_net_w_high_won']-res['baseline_w_won'])/res['baseline_w_won'])
                ]))
        .append( pt.Row([r'$(\tau_1=1.1,\tau_0={:0.2f})$'.format(tau_neutral_high), r'$Var(\Delta \log w^a )$'])
                    .append( [
                        r"${:0.3f}$".format(res['baseline_dw_won']),
                        r"${:0.3f}$".format(res['baseline_dw_direct_high_won']) + " " +
                        STR_PERC.format(100*(res['baseline_dw_direct_high_won']-res['baseline_dw_won'])/res['baseline_dw_won']),
                        "{:0.3f}".format(res['eq_gross_dw_high_won']) + " " +
                        STR_PERC.format(100*(res['eq_gross_dw_high_won']-res['baseline_dw_won'])/res['baseline_dw_won']),
                        "{:0.3f}".format(res['eq_net_dw_high_won']) + " " +
                        STR_PERC.format(100*(res['eq_net_dw_high_won']-res['baseline_dw_won'])/res['baseline_dw_won'])
                    ]))
        )

    print(tab.toTex())
    tab.save_to_Tex('../figures/table6-policy.tex')


#  -------------------------------------- 
#               PASSTHROUGH
#  -------------------------------------- 

def passthrough_analysis():

    model = wd.FullModel.load("res_main_model.pkl")
    p = model.p
    np.random.seed(JMP_CONF['seeds']['passthrough'])

    # we simulate from the model to get a cross-section
    sim = wd.Simulator(model, p)
    sdata = sim.simulate().get_sdata()

    # we construct the different starting values
    tm = sdata['t'].max()
    d0 = sdata.query('e==1 & t==@tm')[['x','z','h','r']]

    # looking at a bunch of possibilities
    passthrough = wdpt.Passthrough(model)

    X = np.array(d0['x'])
    Z = np.array(d0['z'])
    R = np.array(d0['r'])
    res = []

    for shock in ['dz+','dx+','dz-','dx-']:
        for num in ['dv','dw-follow','df-follow','df-ee','dw-ee']:
            for den in ['df-ee','df-follow']:
                d0['pt'] = passthrough.get_pt_vec(Z,X,R,shock,num,den)
                res.append({
                    'val': d0['pt'].mean(),
                    'shock':shock,
                    'den':den,
                    'num':num})
                
    # create the table structure
    # array of cells and class names (that is it)
    df = pd.DataFrame(res)

    # next average positive and negative shocks @todo
    df2 = pd.pivot_table(df,values="val",columns='shock',index=['num','den']).reset_index()
    df2['dx'] = 0.5*(df2['dx+'] + df2['dx-'])
    df2['dz'] = 0.5*(df2['dz+'] + df2['dz-'])
    df2

    tab = pt.Table().setHeaders(['l','l'] + 2*['r'])
    tab.append(pt.Row(["","mobility",r"$x_1$ shock", "$z$ shock"]))
    tab.addRule([[2,4]])

    tab.append(pt.Row(["utility passthrough", "outcome only"])
        .append(df2.query(" num=='dv' & den=='df-ee'")['dx'].to_numpy(), format="{:2.2f}" )
        .append(df2.query(" num=='dv' & den=='df-ee'")['dz'].to_numpy(), format="{:2.2f}" ))

    tab.append(pt.Row(["","yes"])
        .append(df2.query(" num=='dv' & den=='df-follow'")['dx'].to_numpy(), format="{:2.2f}" )
        .append(df2.query(" num=='dv' & den=='df-follow'")['dz'].to_numpy(), format="{:2.2f}" ))

    tab.append(pt.Row(["wage passthrough","yes"])
        .append(df2.query(" num=='dw-follow' & den=='df-follow'")['dx'].to_numpy(), format="{:2.2f}" ) 
        .append(df2.query(" num=='dw-follow' & den=='df-follow'")['dz'].to_numpy(), format="{:2.2f}" ))

    tab.append(pt.Row(["","no"])
        .append( df2.query(" num=='dw-ee' & den=='df-ee'")['dx'].to_numpy(), format="{:2.2f}" )
        .append( df2.query(" num=='dw-ee' & den=='df-ee'")['dz'].to_numpy(), format="{:2.2f}" ))
        
    tab.save_to_Tex('../figures/table5-passthrough.tex')


#  -------------------------------------- 
#             SLICES 
#  -------------------------------------- 

def cf_slices_gen(grid_size):


    param_list = ['x_corr', 'z_corr', 'prod_var_x', 'prod_err_w', 'prod_err_y',
                        'prod_var_x2', 'prod_var_z', 'efcost_ce',
                        'u_bf_m', 'alpha', 's_job', 'efcost_sep'] 

    param_bounds = {
        'efcost_ce': [0.2, 0.5],
        's_job': [0.1, 0.8],
        'alpha': [0.0, 0.4],
        'x_corr': [0.6, 0.9999],
        'z_corr': [0.6, 0.9999],
        'prod_var_x': [0.01, 1.0],
        'prod_var_x2': [0.01, 1.0],
        'prod_var_z': [0.01, 1.0],
        'u_bf_m': [0.000, 0.5],
        'u_rho': [1.2, 1.7],
        'sigma': [0.3, 0.9],
        'prod_rho':   [0.5, 1.5],
        'efcost_sep': [0.000 * 0.25, 0.01 * 0.25],
        'sim_nt_burn': [1, 100],
        'prod_err_w': [0, 0.3],
        'prod_err_y': [0, 0.5]
    }

    # values that parameters that are not moved will take
    p_init = {
        'efcost_ce':  0.33,
        's_job':      0.4,
        'alpha':      0.21,
        'x_corr':     0.89,
        'z_corr':     0.95,
        'prod_var_x': 0.25,
        'prod_var_x2': 0.75,
        'prod_var_z': 0.5,
        'u_bf_m':     0.028,
        'u_rho':      1.5,  # Do not change
        'sigma':      0.8,  # Do not change
        'prod_rho':   1.0,
        'prod_err_w': 0.19,
        'prod_err_y': 0.2,
        'tax_lambda': 1.0,
        'tax_tau':    1.0,
        'max_iter':   5000,
        'num_v':      500,
        'tol_full_model': 1e-8,
        'efcost_sep': 0.002 * 0.25,
        'sim_net_earnings':True, # default is Fasle, this is to analyze tax policy
        'sim_nt_burn': 50,
        'sim_nh': 1000,
        'sim_ni': 25000,
        'sim_nt': 30}

    def create_slice_eval(params, param_name, bounds, n):
        """ create a slice in the parameter space using bounds and number of points """
        local_param_list = []
        for v in np.linspace(bounds[0],bounds[1],n):
            param_temp = params.copy()
            param_temp['moving_param'] = param_name
            param_temp[param_name] = v
            param_temp['rseed'] = np.random.randint(1,1e5)
            local_param_list.append(param_temp)
        return(local_param_list)

    p_init_file = json.load(open("../results/parameters_at_optimal.json"))
    p_init.update(p_init_file)

    np.random.seed(JMP_CONF['seeds']['slices'])
    
    # we add the optimal parameter
    p_init['rseed'] = np.random.randint(1,1e5)
    with open("build/slices/param_slice_init.json", 'w') as fp:
                json.dump(p_init, fp)

    # we add slices for each of the parameters
    for current_param in param_list :
        for i,lp in enumerate(create_slice_eval(p_init, current_param, param_bounds[current_param], grid_size)):
            with open("build/slices/param_slice_{}_{}.json".format(current_param.replace("_",""),i), 'w') as fp:
                json.dump(lp, fp)


def cf_slices_collect():

    d_moments = wd.get_data_moments(filename = "../" + JMP_CONF['main']['moment_file'] ) # should be able to get it from the results instead of pulling the file
    param_list = ['x_corr', 'z_corr', 'prod_var_x', 'prod_err_w', 'prod_err_y',
                            'prod_var_x2', 'prod_var_z', 'efcost_ce',
                            'u_bf_m', 'alpha', 's_job', 'efcost_sep'] 

    def collect_res(f):
        res = pd.read_pickle(f)
        par = json.load(open("{}.json".format(str(f).replace("_moments.pkl",""))))
        
        dd = d_moments.join(res['moments'] , how='left')
        objective = ( dd.eval(' (value_data -  value_model) *weight ').to_numpy() **2).mean()

        ret = {"p_{}".format(p):res['parameters'][p]  for p in param_list}
        ret.update({ "m_{}".format(k):l for k,l in res['moments'].items() })
        ret['objective'] = objective
        
        if 'moving_param' in par.keys():
            ret['moving_param'] = par['moving_param']
        else:
            ret['moving_param'] = 'init'
        
        return(ret)

    #collect_res(next(Path("../python/build/slices").glob("*.pkl")))
    all_res = [collect_res(f) for f in Path("../python/build/slices").glob("*.pkl")]
    all_res = pd.DataFrame(all_res).sort_values('moving_param')
    all_res.to_csv('build/slices.csv')

# ---------------------------------------
#           SURROGATE SLICE
# ---------------------------------------
def surrogate_gen(grid_size):

    model     = wd.FullModel.load("../python/res_main_model.pkl")
    p_init    = model.get_parameters().to_dict()
    param_bounds = JMP_CONF['param_bounds']
    np.random.seed(JMP_CONF['seeds']['slices'])

    Path('build/surrogate').mkdir(parents=True,exist_ok=True)
    param_list = ['efcost_ce'] 

    def create_slice_eval(params, param_name, bounds, n):
        """ create a slice in the parameter space using bounds and number of points """
        local_param_list = []
        for v in np.linspace(bounds[0],bounds[1],n):
            param_temp = params.copy()
            param_temp['moving_param'] = param_name
            param_temp[param_name] = v
            param_temp['rseed'] = np.random.randint(1,1e5)
            local_param_list.append(param_temp)
        return(local_param_list)
    
    # we add slices for each of the parameters
    for current_param in param_list :
        for i,lp in enumerate(create_slice_eval(p_init, current_param, param_bounds[current_param], grid_size)):
            with open("build/surrogate/param_slice_{}_{}.json".format(current_param.replace("_",""),i), 'w') as fp:
                json.dump(lp, fp)

def surrogate_collect():

    d_moments = wd.get_data_moments(filename = "../" + JMP_CONF['main']['moment_file'] ) # should be able to get it from the results instead of pulling the file
    param_list = ['x_corr', 'z_corr', 'prod_var_x', 'prod_err_w', 'prod_err_y',
                            'prod_var_x2', 'prod_var_z', 'efcost_ce',
                            'u_bf_m', 'alpha', 's_job', 'efcost_sep'] 

    model     = wd.FullModel.load("../python/res_main_model.pkl")
    pstar = model.get_parameters().to_dict()

    def collect_res(f):
        res = pd.read_pickle(f)
        par = json.load(open("{}.json".format(str(f).replace("_moments.pkl",""))))

        dd = d_moments.join(res['moments'] , how='left')
        objective = ( dd.eval(' (value_data -  value_model) *weight ').to_numpy() **2).mean()

        ret = {"p_{}".format(p):res['parameters'][p]  for p in param_list}
        ret.update({ "m_{}".format(k):l for k,l in res['moments'].items() })
        ret['objective'] = objective

        if 'moving_param' in par.keys():
            ret['moving_param'] = par['moving_param']
        else:
            ret['moving_param'] = 'init'

        return(ret)


    all_res = [collect_res(f) for f in Path("../python/build/surrogate").glob("*.pkl")]
    all_res = pd.DataFrame(all_res).sort_values('p_efcost_ce')

    X = all_res.p_efcost_ce.to_numpy()
    Y = all_res.objective.to_numpy()

    # param = "efcost_ce"
    plt.figure(figsize=(6, 4), dpi=80)
    ax = plt.gca()

    # set log scaale
    ax.set_yscale('log')

    # plotting the objective
    ax.axvline(x=pstar['efcost_ce'], linestyle=":", color="green")
    spline = smooth_cross(X, np.log(Y), 0.01,10)
    objective_spline = np.exp(spline(X))

    ax.plot(X, Y,'o',fillstyle='none')
    ax.plot(X, objective_spline)

    xlook = np.linspace(X[0],X[-1],1000)
    best_i = np.argmin(spline(xlook))

    ax.axvline(x= xlook[best_i],  color="red")
    ax.set_xlabel(r"$\gamma_1$",fontsize=12)

    plt.savefig("../figures/figurew1-surrogate.pdf",bbox_inches='tight')


# ---------------------------------------
#               BOOTSTRAP
# ---------------------------------------

def cf_bootstrap_gen():

    conf      = toml.load('../balkelamadon.toml')
    d_moments = wd.get_data_moments(filename = "../" + JMP_CONF['main']['moment_file'] ) # should be able to get it from the results instead of pulling the file
    model     = wd.FullModel.load("../python/res_main_model.pkl")
    res       = pd.read_csv('build/slices.csv')

    pstar = model.get_parameters().to_dict()
    params = conf['main']['parameters']
    moments = conf['main']['moments']
    param_bounds = conf['param_bounds']

    # Store derivative of moment wrt param
    moments_1d = np.zeros([len(params), len(moments)])  

    # iterate over moments and parameters
    for ip, param in enumerate(params):
        rp = res[res.moving_param == param].sort_values('p_' + param)
        for im, moment in enumerate(moments):
            # compute derivative at true param using spline
            spline = smooth_cross(rp['p_' + param].to_numpy(), rp['m_' + moment].to_numpy())            
            moments_1d[ip, im] = spline.derivative()( pstar[param] ) 
            
    d_moments = d_moments.reindex(moments, copy=True)
    std_err_W = np.diag(d_moments['weight'] ** 2)
    std_err_S = np.diag(d_moments['value_sd'] ** 2)
    std_err_J = moments_1d @ std_err_W @ np.transpose(moments_1d)
    std_err_omega = moments_1d @ std_err_W @ std_err_S @ \
                    std_err_W @ np.transpose(moments_1d)  

    param_var = np.linalg.inv(std_err_J) @ std_err_omega @ np.linalg.inv(std_err_J)
    param_std = np.sqrt(np.diag(param_var))  

    np.random.seed(JMP_CONF['seeds']['bootstrap'])
    npoints = 15

    for rep in range(100):

        d_moments_pbs = d_moments.copy()
        delta_moments = np.random.normal(size = len(d_moments.index)) * d_moments['value_sd']
        d_moments_pbs['value_data'] = d_moments['value_data'] + delta_moments

        delta =  np.linalg.inv(std_err_J) @ moments_1d @ std_err_W @ delta_moments
        delta = delta / np.sum(delta)

        glb =-np.inf
        gub = np.inf
        for (i,par) in enumerate(params):

            if delta[i]>0:
                lb = (param_bounds[par][0] - pstar[par]) / delta[i]
                ub = (param_bounds[par][1] - pstar[par])/ delta[i]
            else:
                lb = (param_bounds[par][1]  - pstar[par])/ delta[i]
                ub = (param_bounds[par][0]  - pstar[par])/ delta[i]

            glb = np.max([glb,lb])    
            gub = np.min([gub,ub])   

        # v = np.linspace(glb,gub,npoints+2)[1:(npoints+1)]
        glb_in = glb + 0.005 * (gub - glb)
        gub_in = gub - 0.005 * (gub - glb)
        v = np.linspace(glb_in,gub_in,npoints)
        Path('build/bootstrap').mkdir(parents=True,exist_ok=True)

        for k in range(npoints):
            pl = pstar.copy()
            pl['poorbs_rep'] = rep
            pl['rseed'] = np.random.randint(1,1e5)
            pl['bs_shift'] = v[k]
            for (i, par) in enumerate(params):
                pl[par] = pstar[par] + v[k] * delta[i]
            pl.update(d_moments_pbs['value_data'].to_dict())
            json.dump(pl, open(Path('build/bootstrap/param_pb_r{}_p{}.json'.format(rep,k)),'w'),indent=2)

def cf_bootstrap_collect():

    d_moments = wd.get_data_moments(filename = "../" + JMP_CONF['main']['moment_file'] ) 
    conf      = toml.load('../balkelamadon.toml')
    model     = wd.FullModel.load("../python/res_main_model.pkl")

    params = conf['main']['parameters']
    moments = conf['main']['moments']
    param_bounds = conf['param_bounds']    
    pstar = model.get_parameters().to_dict()

    def collect_res(f):
        res = pd.read_pickle(f)
        par = json.load(open("{}.json".format(str(f).replace("_moments.pkl",""))))

        # compute the objective using drawn moments
        model_moments     = res['moments'].to_dict()
        bootstrap_moments = { m:par[m] for m in moments}
        weights           = d_moments.weight.to_dict()
            
        dist = np.array([ (model_moments[m] - bootstrap_moments[m])*weights[m] for m in moments ])
        objective = (dist**2).mean()

        ret = {p:res['parameters'][p]  for p in params}
        ret.update( { m:par[m] for m in moments}) # we collect the value of the moments for this draw
        ret['objective'] = objective
        ret['poorbs_rep'] = par['poorbs_rep']
        ret['bs_shift'] = par['bs_shift']
        return(ret)

    #all_res = [collect_res(f) for f in Path("/home/tlamadon/git/bl-test/python/build/bootstrap").glob("*.pkl")]
    all_res = [collect_res(f) for f in Path("../python/build/bootstrap").glob("*.pkl")]
    all_res = pd.DataFrame(all_res).sort_values(['poorbs_rep','bs_shift'])

    # find the optimal in each of the replications
    A_PARAM = 'bs_shift'
    all_p = []
    for rep in np.unique(all_res['poorbs_rep']):

        dl = all_res.query('poorbs_rep == {}'.format(rep)).sort_values(A_PARAM)

        unit_supp = dl['bs_shift'].to_numpy() #np.linspace(0,1,len(dl))
        spline = smooth_cross(unit_supp, np.log(dl['objective'].to_numpy()), 0.01, 5.0) 

        supp_ext = np.linspace( unit_supp.min(), unit_supp.max(), 1000)
        pred = spline(supp_ext)    
        u_star = supp_ext[pred.argmin()]

        pp = {'u_star' : u_star}
        for p1 in params:
            psupp = dl[p1].to_numpy()
            pval = pstar[p1]  +  u_star * (psupp[-1] - psupp[0] ) / (unit_supp[-1] - unit_supp[0])
            pp[p1] = pval

        all_p.append(pp)

    rr = pd.DataFrame(all_p)
    rr.std()

    r2 = pd.DataFrame(rr.std(),columns=['sd']).drop('u_star')
    r2['value'] = [pstar[p] for p in r2.index]
    r2 = r2[['value','sd']].reset_index().rename(columns={"index": "Parameter", "value":"Value", 'sd':'StdErr'})
    r2.to_csv('../results/res_main_parameters.csv')

    # finally we generate the table
    table_parameters()

# ---------------------------------------
#         Generating pdfs
# ---------------------------------------

def generate_alone_pdf(latex_file):

    # we encapsulate the latex table and generat the pdf
    file = Path(latex_file)
    alone_file = file.with_name( file.stem + '_alone' + '.textmp')
    print(alone_file)

    latex_head = r"""
        \documentclass{article} 
        \usepackage{booktabs} 
        \usepackage{graphics} 
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \global\long\def\E{\mathbb{E}}
        \begin{document} 
        \resizebox{\columnwidth}{!}{%     
        """

    latex_bottom= r"""
        }
        \end{document}     
    """

    latex_source = latex_head + "".join(open(latex_file,'r').readlines()) + latex_bottom
    with alone_file.open('w') as fp:
        fp.write(latex_source)
    print(subprocess.check_output(['pdflatex', '-interaction=nonstopmode','-output-directory', '../figures', alone_file]))

    alone_file.with_suffix(".log").unlink(missing_ok=True)
    alone_file.with_suffix(".aux").unlink(missing_ok=True)
    alone_file.with_suffix(".fls").unlink(missing_ok=True)
    alone_file.with_suffix(".fdb_latexmk").unlink(missing_ok=True)
    alone_file.unlink()

    file.with_name( file.stem + '_alone' + '.pdf').rename( file.with_name( file.stem + '.pdf') )



# ---------------------------------------
#         Moments sensitivity
# ---------------------------------------

def cf_sensitivity_measure():

    parameter_list = {
        'x_corr': [r'$\lambda_x$','persistence for worker productivity'],
        'z_corr': [r'$\lambda_z$','persistence for match quality'],
        'prod_var_x': [r'$\sigma_{x_0}$','dispersion for worker permanent productivity'],
        'prod_var_x2': [r'$\sigma_{x_1}$','dispersion for worker transitory productivity'],
        'prod_var_z': [r'$\sigma_z$','dispersion for match quality'],
        'efcost_sep': [r'$\gamma_0$', 'effort cost parameter'],
        'efcost_ce': [r"$\gamma_1$",'effort cost curvature'],
        'u_bf_m': [r"$b$","flow payment while unemployed"],
        'alpha': [r'$\alpha$','efficiency of the matching function'],
        's_job': [r'$\kappa$','on-the-job search efficiency'],
        'prod_err_w': [r'$m_w$', 'measurement error on earnings'],
        'prod_err_y': [r'$m_y$', 'measurement error on value added per worker']
    }

    moment_list = {
        'pr_u2e': r'$Pr^\text{U2E}$',
        'pr_j2j': r'$Pr^\text{J2J}$',
        'pr_e2u': r'$Pr^\text{E2U}$',
        'var_w': r'$\text{Var}_{S^\text{E}} \big[ \log w_{it} \big]$',
        'mean_dw': r'${\E}_{S^\text{EE}} \big[ \Delta\log w_{it} \big]$',
        'var_dw': r'$\text{Var}_{S^\text{EE}} \big[ \Delta \log w_{it} \big]$',
        'cov_dwdw_l4': r'$\text{Cov}_{S^\text{EEE}} \big[ \Delta \log w_{it} ,  \Delta \log w_{it-1} \big]$',
        'mean_dw_j2j_2': r'${\E}_{S^\text{J2J}} \big[\log w_{it} - \log w_{it-2} \big]$',
        'w_u2e_ee_gap': r"${\E}_{S^\text{E}} \big[ \log w_{it} \big] - {\E}_{S^\text{U2E}} \big[ \log w_{it} \big]$",
        'var_w_longac': r"$\text{Cov}_{S^\text{UEUE}} \big[ \log w_{i,\tau_i(1)}, \log w_{i,\tau_i(2)} \big]$",
        'var_dy': r'$\text{Var}_{S^\text{S}} \big[ \Delta \log y_{it} \big]$',
        'cov_dydy_l4': r'$\text{Cov}_{S^\text{SS}} \big[ \Delta \log y_{it} ,  \Delta \log y_{it-1} \big]$',
        'cov_dydw': r'$\text{Cov}_{S^\text{S}} \big[ \Delta \log w_{it} ,  \Delta \log y_{it} \big]$',
        'cov_dydsep': r'$\text{Cov}_{S^\text{S}} \big[ \Delta \log(1-\tilde{p}_{it}) ,  \Delta \log y_{it} \big]$',
    }

    moment_list = {k:v.replace(r"\big","").replace(r"\E",r"\mathbb{E}").replace(r"\text","") for k,v in moment_list.items()}

    conf      = toml.load('../balkelamadon.toml')
    d_moments = wd.get_data_moments(filename = "../" + JMP_CONF['main']['moment_file'] ) # should be able to get it from the results instead of pulling the file
    model     = wd.FullModel.load("../python/res_main_model.pkl")
    res       = pd.read_csv('build/slices.csv')

    pstar = model.get_parameters().to_dict()
    mstar = d_moments.value_data.to_dict()
    params = conf['main']['parameters']
    moments = conf['main']['moments']

    # Store derivative of moment wrt param
    moments_1d = np.zeros([len(params), len(moments)])  

    # iterate over moments and parameters
    for ip, param in enumerate(params):
        rp = res[res.moving_param == param].sort_values('p_' + param)
        for im, moment in enumerate(moments):
            # compute derivative at true param using spline
            spline = smooth_cross(rp['p_' + param].to_numpy(), rp['m_' + moment].to_numpy())            
            moments_1d[ip, im] = spline.derivative()( pstar[param] ) 
            
    d_moments = d_moments.reindex(moments, copy=True)
    std_err_W = np.diag(d_moments['weight'] ** 2)
    std_err_S = np.diag(d_moments['value_sd'] ** 2)
    std_err_J = moments_1d @ std_err_W @ np.transpose(moments_1d)
    std_err_omega = moments_1d @ std_err_W @ std_err_S @ \
                    std_err_W @ np.transpose(moments_1d)  
    param_var = np.linalg.inv(std_err_J) @ std_err_omega @ np.linalg.inv(std_err_J)

    sensistivity_matrix =  - np.linalg.inv(moments_1d @ std_err_W @ np.transpose(moments_1d)) @ (moments_1d @ std_err_W)
    sensistivity_matrix = np.transpose(sensistivity_matrix)

    fig, ax = plt.subplots(figsize=(10, 10))

    im, cbar = heatmap(np.clip(sensistivity_matrix,-10,10),[moment_list[m] for m in moments], [parameter_list[p][0] for p in params], ax=ax,
                    cmap="coolwarm", cbarlabel="sensitivty")
    texts = annotate_heatmap(im, data = sensistivity_matrix, valfmt="{x:2.2f}")
    fig.savefig('../figures/figurew4-sensitivity.pdf',bbox_inches='tight')


    # all moments
    fig, axs = plt.subplots(len(moments), len(params), figsize=(20, 14), dpi=100, sharey='row', sharex='col')
    for ip, param in enumerate(params):
        rp = res[res.moving_param == param].sort_values('p_' + param)

        # plot all moments
        for im, moment in enumerate(moments):

            # plot evaluations
            axs[im , ip].plot(rp['p_' + param], rp['m_' + moment])
            # axs[im+1, ip].plot(rp['p_' + param], rp['m_' + moment],'o')
            axs[im , ip].errorbar(rp['p_' + param], rp['m_' + moment])

            # add the truth
            axs[im , ip].axhline(y = mstar[moment],linestyle=":")

            # add the p-init parameter
            axs[im , ip].axvline(x=pstar[param], linestyle=":", color="green")

            # add labels
            if (ip == 0): 
                axs[im, ip].set_ylabel(moment_list[moment],rotation=0, labelpad=70)

        # add param label
        axs[im, ip].set(xlabel=parameter_list[param][0])

    #fig.suptitle('Slices (' + all_res['msg'] + ' - ' + all_res['date'])
    fig.subplots_adjust(top=0.88)
    fig.savefig('../figures/figurew3-slices-moments.pdf',bbox_inches='tight')


    fig2, axs2 = plt.subplots(4, 3, figsize=(16, 20), dpi=60, sharey='all')
    for ip, param in enumerate(params):
        rp = res[res.moving_param == param].sort_values('p_' + param)

        ix = ip // 3
        iy = ip % 3
        
        # set log scaale
        axs2[ix, iy].set_yscale('log')
        
        # plotting the objective
        axs2[ix, iy].axvline(x=pstar[param], linestyle=":", color="green")
        axs2[ix, iy].set(xlabel=param)

        fitted = smooth_cross(rp['p_' + param].to_numpy(), np.log(rp['objective'].to_numpy()),0.01,1.0)
        objective_llr2 = np.exp( fitted(rp['p_' + param].to_numpy()) )

        axs2[ix, iy].plot(rp['p_' + param], objective_llr2)
        # axs2[ix, iy].plot(rp['p_' + param], rp['objective'],'o')
        
        axs2[ix, iy].axvline(x=pstar[param], linestyle=":", color="green")

        axs2[ix, iy].set_xlabel(parameter_list[param][0],fontsize=18)

    fig2.savefig('../figures/figurew2-slices-objective.pdf',bbox_inches='tight')


# ---------------------------------------
#         Extras tables
# ---------------------------------------

def table_parameters():

    df = pd.read_csv( PATH_RESULTS / Path('res_main_parameters.csv') ).set_index('Parameter')

    parameter_list = {
        'x_corr': [r'$\lambda_x$','persistence for worker productivity'],
        'z_corr': [r'$\lambda_z$','persistence for match quality'],
        'prod_var_x': [r'$\sigma_{x_0}$','dispersion for worker permanent productivity'],
        'prod_var_x2': [r'$\sigma_{x_1}$','dispersion for worker transitory productivity'],
        'prod_var_z': [r'$\sigma_z$','dispersion for match quality'],
        'efcost_sep': [r'$\gamma_0$', 'effort cost parameter'],
        'efcost_ce': [r"$\gamma_1$",'effort cost curvature'],
        'u_bf_m': [r"$b$","flow payment while unemployed"],
        'alpha': [r'$\alpha$','efficiency of the matching function'],
        's_job': [r'$\kappa$','on-the-job search efficiency'],
        'prod_err_w': [r'$m_w$', 'measurement error on earnings'],
        'prod_err_y': [r'$m_y$', 'measurement error on value added per worker']
    }

    tab = (pt.Table().setHeaders(['l', 'l', 'c']))

    for key, vals in parameter_list.items():
        value_model = df.loc[key]['Value']
        value_sd = df.loc[key]['StdErr']
        tab.append(pt.Row([vals[1],vals[0]])
                   .append([value_model], format="{:#2.2g}")
                   .setEndSpace(-4))
        tab.append(pt.Row(['',''])
                   .append([value_sd], format="{{ \\footnotesize ({:#2.2g}) }}")
                   .setEndSpace(3))

    tab.save_to_Tex( PATH_PAPER_FIG / Path('table2-parameters.tex') )
    print("done")


def table_model_fit():

        df = pd.read_csv( PATH_RESULTS / Path('res_main_fit.csv') ).set_index('mom')

        moment_list = {
            'pr_u2e': r'$Pr^\text{U2E}$',
            'pr_j2j': r'$Pr^\text{J2J}$',
            'pr_e2u': r'$Pr^\text{E2U}$',
            'var_w': r'$\text{Var}_{S^\text{E}} \big[ \log w_{it} \big]$',
            'mean_dw': r'${\E}_{S^\text{EE}} \big[ \Delta\log w_{it} \big]$',
            'var_dw': r'$\text{Var}_{S^\text{EE}} \big[ \Delta \log w_{it} \big]$',
            'cov_dwdw_l4': r'$\text{Cov}_{S^\text{EEE}} \big[ \Delta \log w_{it} ,  \Delta \log w_{it-1} \big]$',
            'mean_dw_j2j_2': r'${\E}_{S^\text{J2J}} \big[\log w_{it} - \log w_{it-2} \big]$',
            'w_u2e_ee_gap': r"${\E}_{S^\text{E}} \big[ \log w_{it} \big] - {\E}_{S^\text{U2E}} \big[ \log w_{it} \big]$",
            'var_w_longac': r"$\text{Cov}_{S^\text{UEUE}} \big[ \log w_{i,\tau_i(1)}, \log w_{i,\tau_i(2)} \big]$",
            'var_dy': r'$\text{Var}_{S^\text{S}} \big[ \Delta \log y_{it} \big]$',
            'cov_dydy_l4': r'$\text{Cov}_{S^\text{SS}} \big[ \Delta \log y_{it} ,  \Delta \log y_{it-1} \big]$',
            'cov_dydw': r'$\text{Cov}_{S^\text{S}} \big[ \Delta \log w_{it} ,  \Delta \log y_{it} \big]$',
            'cov_dydsep': r'$\text{Cov}_{S^\text{S}} \big[ \Delta \log(1-\tilde{p}_{it}) ,  \Delta \log y_{it} \big]$',
        }

        tab = (pt.Table()
                    .addRow(['','data','model'])
                    .setHeaders(['l','c','c'])
                    .addRule([(2,3)]))

        for key,name in moment_list.items():
            value_data = df.loc[key]['value_data']
            value_model = df.loc[key]['value_model']
            value_sd = df.loc[key]['value_sd']
            tab.append( pt.Row([name])
                            .append([ value_data ,value_model ], format="{:#2.2g}")
                            .setEndSpace(-5))
            tab.append( pt.Row([''])
                            .append([value_sd ], format="{{ \\footnotesize ({:#2.2g}) }}")
                            .append([''])
                            .setEndSpace(-1) )

        tab.save_to_Tex( PATH_PAPER_FIG / Path( 'table1-moments.tex') )

def table_stats():

        df = pd.read_csv( PATH_RESULTS / Path( 'data-stats.csv') )
        stats = df.loc[0].to_dict()

        show = lambda key,title,fmt : pt.Row([title]).append([ stats[key] ], format=fmt)

        stats['ydata_mean_earnings_ft'] = float(stats['ydata_mean_wage_ft']) + np.log(12) 
        stats['ydata_var_wage_ft'] = float(stats['ydata_sd_wage_ft'])**2
        tab =(pt.Table().setHeaders(['l','r'])
            .append( show('ydata_nyearobs',"Number of year observations","{:,.0f}"))
            .append( show('ydata_nyearobs_ft',"Number of year observations with 12 months worked","{:,.0f}"))
            .append( show('rdata_nwid',"Number of unique workers","{:,.0f}"))
            .append( show('rdata_nfid',"Number of unique firms","{:,.0f}"))
            .append( show('rdata_employed_month',"Employment share","{:,.2f}")
                        .setEndSpace(5))
            .append( show('ydata_mean_earnings_ft',"Mean log earnings among full-year observations","{:,.2f}"))
            .append( show('ydata_var_wage_ft',"Variance of log earnings among full-year observations","{:,.2f}"))
        )

        tab.save_to_Tex( PATH_PAPER_FIG / Path('tablew2-stats.tex') )
        #print(tab.toTex())


# ---------------------------------------
#         Plotting utilities
# ---------------------------------------

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color = textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# -----------------------------
#       OPTIMIZER 
# -----------------------------

def optimize_gen(step, grid_size):

    Path('build/opt/step{}'.format(step)).mkdir(parents=True, exist_ok=True)

    param_bounds = JMP_CONF['param_bounds']    
    d_moments = wd.get_data_moments(filename = "../" + JMP_CONF['main']['moment_file'] ) # should be able to get it from the results instead of pulling the file
    pstar = json.load(open("../results/parameters_at_optimal.json"))
    params = JMP_CONF['main']['parameters'] 

    np.random.seed(JMP_CONF['seeds']['optimizer'] + step)

    def collect_res(f):
        res = pd.read_pickle(f)
        par = json.load(open("{}.json".format(str(f).replace("_moments.pkl",""))))

        dd = d_moments.join(res['moments'] , how='left')
        objective = ( dd.eval(' (value_data -  value_model) *weight ').to_numpy() **2).mean()

        ret = {"{}".format(p):res['parameters'][p]  for p in params}
        ret['objective'] = objective

        if 'moving_param' in par.keys():
            ret['moving_param'] = par['moving_param']
        else:
            ret['moving_param'] = 'init'

        return(ret)

    def create_slice_eval(params, param_name, bounds, n):
        """ create a slice in the parameter space using bounds and number of points """
        local_param_list = []
        for v in np.linspace(bounds[0],bounds[1],n):
            param_temp = params.copy()
            param_temp['moving_param'] = param_name
            param_temp[param_name] = v
            param_temp['rseed'] = np.random.randint(1,1e5)
            local_param_list.append(param_temp)
        return(local_param_list)

    # we load the previous step output, update and compute new grid
    if step>1:
        all_res = [collect_res(f) for f in Path("../python/build/opt/step{}/".format(step-1)).glob("*.pkl")]
        all_res = pd.DataFrame(all_res)

        p_previous = all_res.query("moving_param == 'init'").iloc[0].to_dict()

        moving_param = list(set(all_res.moving_param).difference(['init']))[0]
        all_res = all_res.sort_values(moving_param)

        # compute the surrogate function and optimal value
        all_res = pd.DataFrame(all_res).sort_values(moving_param)
        X = all_res[moving_param].to_numpy()
        Y = all_res.objective.to_numpy()
        spline = smooth_cross(X, np.log(Y), 0.01,10)
        xlook = np.linspace(X[0],X[-1],1000)
        best_i = np.argmin(spline(xlook))
        print( "moving {} from {} to  {}".format(moving_param,p_previous[moving_param],xlook[best_i]))


        # we plot the outcome
        plt.figure(figsize=(6, 4), dpi=80)
        ax = plt.gca()

        # set log scaale
        ax.set_yscale('log')

        # plotting the objective
        ax.axvline(x=p_previous[moving_param], linestyle=":", color="green")
        objective_spline = np.exp(spline(X))
        ax.plot(X, Y,'o',fillstyle='none')
        ax.plot(X, objective_spline)
        ax.axvline(x= xlook[best_i],  color="red")
        ax.set_xlabel(moving_param,fontsize=12)
        plt.savefig(Path('build/opt/step{}/plot_update.png'.format(step-1)),bbox_inches='tight')

        # update the parameter
        p_current = pstar.copy()
        p_current.update(p_previous)
        p_current[moving_param] = xlook[best_i]

        # pick next parameter to update
        pi = (np.argmax([pg == moving_param for pg in params]) + 1) % len(params) 
        current_param = params[pi]
    else:
        p_current = json.load(open("../results/parameters_at_optimal.json"))
        current_param = params[0]

    # we prepare the grids and save the parameters to evaluate
    param_list = create_slice_eval(p_current, current_param, param_bounds[current_param], grid_size)

    # we save each of these to a json file 
    p_current['moving_param'] = "init"
    p_current['rseed'] = np.random.randint(1,1e5)
    json.dump(p_current, open(Path('build/opt/step{}/param_opt_pinit_s{}.json'.format(step,step)),'w'),indent=2)
    for (i,pl) in enumerate(param_list):
        json.dump(pl, open(Path('build/opt/step{}/param_opt_p{}_s{}.json'.format(step,i,step)),'w'),indent=2)

