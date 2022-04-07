"""
This script allows to evaluate the model at a list of parameter configuration.
It takes an json list of parameters at which to evaluate the model.
"""

import os

# limit to 1 core/thread - needs to be set before everything else
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import argparse,logging
import numpy as np
import pickle as pkl
from pathlib import Path
import wagedyn as wd
import pandas as pd
import json
from pathlib import Path
import results as results
import os

#logging.basicConfig(filename='jmp-interactive.log',format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
from datetime import datetime

# ----- parsing input arguments -------
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-p","--parameter",  help="parameter file to use (json)")
parser.add_argument("-m","--model",  help="file where to save the model",default="")
parser.add_argument("-v","--vdec",  help="file where to save the variance decomposition",default="")
parser.add_argument("-s","--simulate",  help="file where to save simulation results",default="")
parser.add_argument("-l","--logfile",  help="file to log to")
args = parser.parse_args()

# -------- load configuration file ---------
param_file = Path(args.parameter)
logging.info("loading file {}".format(param_file))

# load the parameters
pdict = json.loads(param_file.read_text())
p = wd.Parameters(pdict)

# set the seed if provided
if 'rseed' in pdict.keys():
    np.random.seed(pdict['rseed'])

# solve model
logging.info("solving the model")
model = wd.FullModel(p).solve(plot=False)

if len(args.model)>0:    
    logging.info("saving model to {}".format(args.model))
    model.save(args.model)

if len(args.vdec)>0:
    logging.info("computing variance decomposition".format(args.model))
    results.cf_simulate_level_var_dec(args.vdec, args.randomseed, noise=False, model = model)

if len(args.simulate)>0:    
    # simulate
    sim = wd.Simulator(model, p)
    moms_mean, moms_var = sim.simulate_moments_rep(p.sim_nrep)

    # save results
    extra = {'err_w1':model.error_w1, 'err_j':model.error_j, 'err_j1p':model.error_j1p, 'err_js':model.error_js, 'niter':model.niter}
    res = {'moments':moms_mean, 'moments_var':moms_var, 'parameters': model.get_parameters().to_dict(), 'extra':extra, 'inputs': pdict}

    logging.info("saving moments to {}".format(args.simulate))
    logging.info("{}".format(res))
    with open(args.simulate,"wb") as file:
        pkl.dump(res,file)
    #moments_file.write_text(json.dumps())


