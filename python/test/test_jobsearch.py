import pytest
import numpy as np
import wagedyn as wd

@pytest.fixture
def job_search():
        p = wd.Parameters()
        p.max_iter = 200
        p.num_v = 1000
        mod = wd.SimpleModel(p)
        mod.solve_with_effort()

        x = 10
        js = wd.JobSearch()
        js.update(mod.Vf_W1[p.z_0,:,x], mod.prob_find_vx[:, x],False)

        return js

def test_re(job_search):
    re = job_search.re( job_search.input_V2 )
    assert (np.diff(re) <= 0 ).all(), "re is not decreasing"
    assert ( re >= -1e-6 ).all() , "re is positive everywhere"
    assert np.abs( job_search.re(job_search.e0)) < 1e-10 , " r(e0) is not 0"

def test_pe(job_search):
    pe = job_search.re( job_search.input_V2 )
    assert ( np.diff(pe) <= 0 ).all(), "pe is not decreasing"
    assert ( pe >= -1e-6 ).all()         , "pe is positive everywhere"
    assert  np.abs( job_search.pe(job_search.e0)) < 1e-10 , " p(e0) is not 0"
    assert  np.abs( job_search.pe(job_search.e0 + 10)) < 1e-10 , " p(e0+10) is not 0"

def test_ve(job_search):
    ve = job_search.ve( job_search.input_V2 )
    #self.assertTrue( ( ve >= self.js.input_V2 ).all(), "ve is not lager than e everywhere")
    assert  (np.diff(ve) >= 0).all()  , "ve is not increasing in e everywhere"
    assert  (ve <= job_search.e0).all() , "ve is not less than Vmax everywhere"

def test_vere(job_search):
    ve = job_search.ve( job_search.input_V2 )
    pe = job_search.pe( job_search.input_V2 )
    re = job_search.re( job_search.input_V2 )
    
    assert np.allclose( 
         re,
         pe*(ve - job_search.input_V2 ),rtol=1e-3), \
             "re and pe,pe do not mactch"

if __name__ == '__main__':
    unittest.main()