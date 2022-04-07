import pytest
import numpy as np
import wagedyn as wd
from scipy.misc import derivative


# content of test_sample.py
def test_utility():
        p = wd.Parameters()
        p.tax_lambda = 0.9
        p.tax_tau = 1.1

        pref = wd.Preferences(p)

        input_w = np.linspace(0.01, 10, 1000)
        input_u = pref.utility(input_w)

        assert np.allclose( 
                pref.inv_utility(pref.utility(input_w)), 
                input_w), \
                     "Utility function and its inverse utility are not compatible"

        assert np.allclose( 
                pref.utility_1d(input_w), 
                derivative(pref.utility, input_w, dx=1e-7)), \
                     "Utility function and its derivative are not compatible"

        assert np.allclose( 
                pref.inv_utility_1d(input_u), 
                derivative(pref.inv_utility, input_u, dx=1e-7)), \
                     "Inverse utility function and its derivative are not compatible"


def test_effort():
        p = wd.Parameters()
        pref = wd.Preferences(p)

        delta_grid = np.linspace( p.efcost_sep , 1, 100)

        cp = derivative(pref.effort_cost, delta_grid, dx=1e-7)
        delta_hat = pref.inv_effort_cost_1d(cp)

        assert np.allclose(delta_grid, delta_hat), \
                     "Inverse of derivative of effort cost effort cost are not compatible"

 
def test_taxes():
        p1 = wd.Parameters()
        p2 = wd.Parameters()
        p2.tax_lambda = 0.9
        p2.tax_tau = 1.2
        pref = wd.Preferences(p2)
        input_w = np.linspace(0.01, 10, 1000)

        assert np.allclose( 
                pref.utility(input_w), 
                (p1.u_a * np.power( 1.2 * np.power(input_w, 0.9), (1.0- p1.u_rho)) - p1.u_b)/ (1-p1.u_rho)) , \
                        "applying tax parameter does not work"



if __name__ == '__main__':
    unittest.main()