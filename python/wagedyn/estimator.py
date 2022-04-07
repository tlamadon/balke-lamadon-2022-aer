"""
    file containing functions for estimation
"""

from wagedyn.modelfull import FullModel
from wagedyn.simulate import Simulator
import wagedyn.primitives as parameters

import numpy as np
import pandas as pd
import json

import logging

"""
    Simple extension of json encoder/decoder to handle data frames
    json.dump(slice, open("slice.json",'w'), cls = wd.JSONEncoder, indent=4)
"""
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_dict(orient='index')
        return json.JSONEncoder.default(self, obj)


def get_data_moments(filename):
    """
    Extracts and prepare the data moments using provdided csv file as input
    """

    smoms = pd.read_csv(filename).set_index('variable')

    dmoms = pd.DataFrame([
        ['pr_u2e',        smoms['main']['u2e'],          smoms['sd']['u2e']],
        ['pr_j2j',        smoms['main']['j2j'],          smoms['sd']['j2j']],
        ['pr_e2u',        smoms['main']['e2u'],          smoms['sd']['e2u']],
        ['cov_dydw',      smoms['main']['dwdy_cov'],     smoms['sd']['dwdy_cov']],
        ['var_dy',        smoms['main']['dy_ac_0'],      smoms['sd']['dy_ac_0']],
        ['cov_dydy_l4',   smoms['main']['dy_ac_1'],      smoms['sd']['dy_ac_1']],
        ['var_dw',        smoms['main']['dw_ac_0'],      smoms['sd']['dw_ac_0']],
        ['cov_dwdw_l4',   smoms['main']['dw_ac_1'],      smoms['sd']['dw_ac_1']],
        ['mean_dw',       smoms['main']['dw_mean'],      smoms['sd']['dw_mean']],
        ['mean_dw_j2j_2', smoms['main']['dw_j2j_2'],     smoms['sd']['dw_j2j_2']],
        ['w_u2e_ee_gap',  smoms['main']['w_u2e_ee_gap'], smoms['sd']['w_u2e_ee_gap']],
        ['var_w',         smoms['main']['w_var'],        smoms['sd']['w_var']],
        ['cov_dydsep',    smoms['main']['cov_dydlsep'],  smoms['sd']['cov_dydlsep']],
        ['var_w_longac',  smoms['main']['w_long_ac'],    smoms['sd']['w_long_ac']],
    ], columns=['mom', 'value_data', 'value_sd']).set_index('mom')

    # simply scale the moments
    dmoms['weight'] = 1/dmoms['value_data']
    dmoms.loc['cov_dwdw_l4', 'weight'] = dmoms['weight']['var_dw']
    dmoms.loc['cov_dydy_l4', 'weight'] = dmoms['weight']['var_dy']
    # increase wieght of pass-through moment
    dmoms.loc['cov_dydw', 'weight'] = 1/0.001
    # make sure all weights are positive
    dmoms['weight'] = np.abs(dmoms['weight'])

    return(dmoms)

def get_data_moments_old(filename):
    """
    Extracts and prepare the data moments using provdided csv file as input
    """

    smoms = pd.read_csv(filename).set_index('variable')

    dmoms = pd.DataFrame([
        ['pr_u2e',       smoms['main']['u2e'],         smoms['sd']['u2e']],
        ['pr_j2j',       smoms['main']['j2j'],         smoms['sd']['j2j']],
        ['pr_e2u',       smoms['main']['e2u'],         smoms['sd']['e2u']],
        ['cov_dydw',     smoms['main']['dwdy_cov'],    smoms['sd']['dwdy_cov']],
        ['var_dy',       smoms['main']['y_var_mu'],    smoms['sd']['y_var_mu']],
        ['var_dw',       smoms['main']['w_var_mu'],    smoms['sd']['w_var_mu']],
        ['mean_dw',      smoms['main']['dw_mean'],     smoms['sd']['dw_mean']],
        ['w_u2e_ee_gap', smoms['main']['w_mean'] - smoms['main']['w_mean_u2e'],
         np.sqrt(smoms['sd']['w_mean'] ** 2 + smoms['sd']['w_mean_u2e'] ** 2)],
        ['var_w',        smoms['main']['w_var'],       smoms['sd']['w_var']],
        ['cov_dwdw_l4',  0.0,                          0.001],
        ['var_w_longac', smoms['main']['w_long_ac'],   smoms['sd']['w_long_ac']],
    ], columns=['mom', 'value_data', 'value_sd']).set_index('mom')

    # simply scale the moments
    dmoms['weight'] = 1/dmoms['value_data']
    dmoms.loc['cov_dwdw_l4', 'weight'] = dmoms['weight']['var_dw']
    # increase wieght of pass-through moment
    dmoms.loc['cov_dydw', 'weight'] = 1/0.005

    return(dmoms)


def compute_objective(params, moments_file , tb=False, simple_only=False):
    """
    This function solves the model for the provided set of parameters
    to overwrite, and returns all information related to the computation
    of the model and the simulated moments.

    :param params:
    :return: strtucture with moments and parameters
    """

    log = logging.getLogger('ObjectiveFunction')
    log.setLevel(logging.INFO)

    # set the seed if provided
    if 'rseed' in params.keys():
        np.random.seed(params['rseed'])

    # make sure that nt_burn is an integer
    if 'sim_nt_burn' in params.keys():
        params['sim_nt_burn'] = int(round(params['sim_nt_burn']))

    # update using the input parameters
    p = parameters.Parameters(params)

    # solve the model suing the parameters
    log.info("solving the model")
    if simple_only:
        model = FullModel(p, tb).solve_simple_only()
    else:
        model = FullModel(p, tb).solve()

    # simulate the model mutliple times
    log.info("simulating")
    sim = Simulator(model, p)
    moms_mean, moms_var = sim.simulate_moments_rep(p.sim_nrep)

    # get moments from data and attach
    log.debug("loading moments from {}".format(moments_file))
    dmoms = get_data_moments(filename = moments_file ).join(moms_mean).join(moms_var)

    # compute the objective function
    objective = np.power( dmoms.weight * (dmoms.value_data - dmoms.value_model), 2).mean()
    log.info("objective = {:2.4e}".format(objective))

    # save results
    extra = {'err_w1':model.error_w1, 'err_j':model.error_j, 'err_j1p':model.error_j1p, 'err_js':model.error_js, 'niter':model.niter}
    res = {'objective':objective, 'moments' : dmoms , 'parameters': params, 'extra' : extra}
    return(res)
