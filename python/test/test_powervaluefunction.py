import unittest
from wagedyn.search import JobSearch
from wagedyn.modelsimple import SimpleModel
from wagedyn.primitives import Parameters

class TestPowerValueFunction(unittest.TestCase):

    def setUp(self):
        p = Parameters()
        p.max_iter = 200
        p.num_v = 1000
        mod = SimpleModel(p)
        mod.solve_with_effort()

        x = 10
        js = JobSearch()
        js.update(mod.Vf_W1[p.z_0,:,x], mod.prob_find_vx[:, x],False)

        self.js=js

    def test_re(self):
        pass

if __name__ == '__main__':
    unittest.main()