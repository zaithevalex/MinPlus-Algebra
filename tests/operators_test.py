import unittest
from scipy.interpolate import interp1d
import lib.operators as ops
import lib.curves as curves
import pandas as pd
import numpy as np

class TestOperators(unittest.TestCase):
    def test_AddConst(self):
        class TestAddConstClass:
            def __init__(self, XSet: list, func = None, YSet: list = None, const = 0., expectedXSet=None, expectedY=None):
                self.Xset = XSet
                self.func = func
                self.Yset = YSet
                self.const = const
                self.expectedXSet = expectedXSet
                self.expectedY = expectedY

        cases = []
        cases.append(TestAddConstClass(
            [],
            func=None,
            YSet=None,
            const=0.,
            expectedXSet=None,
            expectedY=None))
        cases.append(TestAddConstClass(
            XSet=[1.0, 2.0, 3.0],
            func=curves.linearCurve,
            const=1.0,
            expectedXSet=[1.0, 2.0, 3.0],
            expectedY=interp1d(
                pd.DataFrame({'XSet' : [1.0, 2.0, 3.0]})['XSet'],
                pd.DataFrame({'func' : np.array([(curves.linearCurve(1.0, 1.0, x) + 1.0) for x in [1.0, 2.0, 3.0]])})['func'],
                fill_value='extrapolate')))
        cases.append(TestAddConstClass(
            [1.0, 2.0, 3.0],
            YSet=[1.0, 2.0, 3.0],

        ))

        for case in cases:
            want1, want2 = ops.AddConst(case.Xset, func=case.func, YSet=case.Yset, const=case.const)
            self.assertEqual(want1, case.expectedXSet)

            if type(case.expectedY).__name__ == 'interp1d':
                self.assertTrue(np.array_equal(want2.x, case.expectedY.x))
                self.assertTrue(np.array_equal(want2.y, case.expectedY.y))
            else:
                self.assertEqual(want2, case.expectedY)