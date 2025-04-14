import lib.operators as ops
import numpy as np
import pandas as pd
import scipy.integrate as spi
from scipy.interpolate import interp1d
import unittest

# CONSTS:
EPS = 1e-5

# TEST FUNCTIONS:
def testBetaTransferCurve1(x):
    if x <= 2:
        return 0
    return 3 * (x - 2)

def testBetaTransferCurve2(x):
    if x <= -1:
        return 0
    return 3 * (x - 2) + 9

def testLinearCurve(x):
    return 1 * np.float64(x) + 1

def testPiecewiseCurve1(x):
    if x <= 0:
        return 1
    elif x >= 0 and x <= 2:
        return 1 + x
    elif x > 2 and x <= 20.15:
        return 3 + 0.25 * (x - 2)

    return 15

def testPiecewiseCurve2(x):
    if x <= 3:
        return 0
    elif x > 3 and x <= 5:
        return (x - 3)
    elif x > 5 and x <= 12:
        return 2

    return 7 + 0.5 * (x - 12)

def testSquareCurve1(x):
    return 1 * np.float64(x * x)

def testSquareCurve2(x):
    return 1 * np.float64(x) ** 2 + 1

# TESTS:
class TestOperators(unittest.TestCase):
    def test_AddConst(self):
        class TestAddConstClass:
            def __init__(self,
                         XSet: list,
                         func = None,
                         YSet: list = None,
                         const = 0.,
                         expectedXSet = None,
                         expectedY: [interp1d, list, None] = None):
                self.Xset = XSet
                self.func = func
                self.Yset = YSet
                self.const = const
                self.expectedXSet = expectedXSet
                self.expectedY = expectedY

        cases = []
        cases.append(TestAddConstClass(
            [],
            const = 0.,
            expectedXSet = None,
            expectedY = None))
        cases.append(TestAddConstClass(
            [1., 2., 3.],
            func = testLinearCurve,
            const = 1.,
            expectedXSet = [1., 2., 3.],
            expectedY = interp1d(
                pd.DataFrame({'XSet' : [1., 2., 3.]})['XSet'],
                pd.DataFrame({'func' : np.array([(testLinearCurve(x) + 1.) for x in [1., 2., 3.]])})['func'],
                fill_value = 'extrapolate')))
        cases.append(TestAddConstClass(
            [1., 2., 3.],
            YSet = [1., 2., 3.],
            const = 1.,
            expectedXSet = [1., 2., 3.],
            expectedY = [2., 3., 4.],
        ))

        for case in cases:
            val1, val2 = ops.AddConst(case.Xset, func=case.func, YSet=case.Yset, const=case.const)
            self.assertTrue(np.array_equal(val1, case.expectedXSet))

            if type(case.expectedY) == interp1d:
                self.assertTrue(np.array_equal(val2.x, case.expectedY.x))
                self.assertTrue(np.array_equal(val2.y, case.expectedY.y))

            else:
                self.assertEqual(val2, case.expectedY)

    def test_ConvertDataSetToLinearFunction(self):
        class TestConvertDataSetToLinearFunctionClass:
            def __init__(self, XSet: list, YSet: list, expectedInterp1d: [interp1d, list, None] = None):
                self.XSet = XSet
                self.YSet = YSet
                self.expectedInterp1d = expectedInterp1d

        cases = []
        cases.append(TestConvertDataSetToLinearFunctionClass(
            [1., 2., 3.],
            [1., 2.],
            "XSet and YSet must have same length",
        ))
        cases.append(TestConvertDataSetToLinearFunctionClass(
            [],
            [],
            "XSet must have length > 0",
        ))
        cases.append(TestConvertDataSetToLinearFunctionClass(
            [],
            [1., 2., 3.],
            "XSet must have length > 0",
        ))
        cases.append(TestConvertDataSetToLinearFunctionClass(
            [1., 2., 3.],
            [],
            "YSet must have length > 0",
        ))
        cases.append(TestConvertDataSetToLinearFunctionClass(
            [1., 2., 3.],
            [1., 2., 3.],
            interp1d(
                pd.DataFrame({'XSet': [1., 2., 3.]})['XSet'],
                pd.DataFrame({'func': [1., 2., 3.]})['func'],
                fill_value='extrapolate')))

        for case in cases:
            try:
                val = ops.ConvertDataSetToLinearFunction(case.XSet, case.YSet)

                if type(val) == interp1d:
                    self.assertTrue(np.array_equal(val.x, case.expectedInterp1d.x))
                    self.assertTrue(np.array_equal(val.y, case.expectedInterp1d.y))

                else:
                    self.assertEqual(val, case.expectedInterp1d)

            except ValueError as e:
                self.assertEqual(str(e), case.expectedInterp1d)

    def test_ConvertFunctionToDataSet(self):
        class TestConvertFunctionToDataSet:
            def __init__(self, XSet: list, func, expectedXSet, expectedYSet):
                self.XSet = XSet
                self.func = func
                self.expectedXSet = expectedXSet
                self.expectedYSet = expectedYSet

        cases = []
        cases.append(TestConvertFunctionToDataSet(
            [1., 2., 3., 4, 5.],
            testLinearCurve,
            [1., 2., 3., 4., 5.],
            [2., 3., 4., 5., 6.],
        ))

        for case in cases:
            val1, val2 = ops.ConvertFunctionToDataSet(case.XSet, case.func)
            self.assertTrue(np.array_equal(val1, case.expectedXSet))
            self.assertTrue(np.array_equal(val2, case.expectedYSet))

    def test_L1Norm(self):
        class TestL1NormClass:
            def __init__(self, XSet: list, func1, func2, expectedL1Norm):
                self.XSet = XSet
                self.func1 = func1
                self.func2 = func2
                self.expectedL1Norm = expectedL1Norm

        cases = []
        cases.append(TestL1NormClass(
            [1., 2., 3.],
            testLinearCurve,
            testSquareCurve2,
            spi.simpson(np.abs(testLinearCurve([1., 2., 3.]) - testSquareCurve2([1., 2., 3.])), [1., 2., 3.])))

        for case in cases:
            self.assertEqual(ops.L1Norm(case.XSet, case.func1, case.func2), case.expectedL1Norm)

    def test_MinimizeL1Norm(self):
        class TestMinimizeL1NormClass:
            def __init__(self,
                         XSet: list,
                         func1,
                         func2,
                         bottom_border = 0.,
                         top_border = 0.,
                         step = 0.,
                         expectedL1Norm: float = 0.):
                self.XSet = XSet
                self.func1 = func1
                self.func2 = func2
                self.bottom_border = bottom_border
                self.top_border = top_border
                self.step = step
                self.expectedL1Norm = expectedL1Norm

        cases = []
        cases.append(TestMinimizeL1NormClass(
            [1., 2., 3.],
            testLinearCurve,
            testSquareCurve2,
            0.,
            5.,
            0.1,
            2.
        ))

        for case in cases:
            self.assertTrue(abs(case.expectedL1Norm - ops.MinimizeL1Norm(case.XSet,
                                                                         case.func1,
                                                                         case.func2,
                                                                         case.bottom_border,
                                                                         case.top_border,
                                                                         case.step)) < EPS)

    def test_MinPlusConvolution(self):
        class TestMinPlusConvolutionClass:
            def __init__(self,
                         XSet: list,
                         func1 = None,
                         func2 = None,
                         YSet1: list = None,
                         YSet2: list = None,
                         expectedXSet: list = None,
                         expectedY: [str, interp1d, list, None] = None):
                self.XSet = XSet
                self.func1 = func1
                self.func2 = func2
                self.YSet1 = YSet1
                self.YSet2 = YSet2
                self.expectedXSet = expectedXSet
                self.expectedY = expectedY

        cases = []
        cases.append(TestMinPlusConvolutionClass(
            XSet=[1., 2., 3., 4., 5.],
        ))
        cases.append(TestMinPlusConvolutionClass(
            XSet = [-1., 2., 3., 4., 5.],
            YSet1 = [1., 2., 3., 4., 5.],
            YSet2 = [1., 2., 3., 4., 5.],
            expectedXSet = [-1., 2., 3., 4., 5.],
            expectedY = "('left border of the domain of definition interval must be positive(> 0), but expected:', -1.0)"
        ))
        cases.append(TestMinPlusConvolutionClass(
            XSet = [1., 2., 3., 4., 5.],
            func1 = testLinearCurve,
            func2 = testBetaTransferCurve1,
            expectedY = interp1d(
                pd.DataFrame({'XSet': [1., 2., 3., 4., 5.]})['XSet'],
                pd.DataFrame({'func': [2., 2., 3., 4., 5.]})['func'],
                fill_value='extrapolate')))
        cases.append(TestMinPlusConvolutionClass(
            XSet = [1., 2., 3., 4., 5.],
            YSet1 = [1., 2., 3., 4., 5.],
            YSet2 = [0., 0., 3., 6., 9.],
            expectedXSet = [1., 2., 3., 4., 5.],
            expectedY = [1., 1., 2., 3., 4.],
        ))

        for case in cases:
            try:
                val1, val2 = ops.MinPlusConvolution(XSet  = case.XSet,
                                                    func1 = case.func1,
                                                    func2 = case.func2,
                                                    YSet1 = case.YSet1,
                                                    YSet2 = case.YSet2)
                self.assertTrue(np.array_equal(val1, case.expectedXSet))
                self.assertTrue(np.array_equal(val2, case.expectedY))

            except ValueError as e:
                self.assertEqual(str(e), case.expectedY)

            except TypeError as e:
                val = ops.MinPlusConvolution(XSet  = case.XSet,
                                             func1 = case.func1,
                                             func2 = case.func2)
                self.assertTrue(np.array_equal(val.x, case.expectedY.x))
                self.assertTrue(np.array_equal(val.y, case.expectedY.y))

    def test_MinPlusDeconvolution(self):
        class TestMinPlusConvolutionClass:
            def __init__(self,
                         XSet: list,
                         func1 = None,
                         func2 = None,
                         YSet1: list = None,
                         YSet2: list = None,
                         expectedXSet: list = None,
                         expectedY: [interp1d, list, None] = None):
                self.XSet = XSet
                self.func1 = func1
                self.func2 = func2
                self.YSet1 = YSet1
                self.YSet2 = YSet2
                self.expectedXSet = expectedXSet
                self.expectedY = expectedY

        cases = []
        cases.append(TestMinPlusConvolutionClass(
            XSet=[1., 2., 3., 4., 5.],
        ))
        cases.append(TestMinPlusConvolutionClass(
            XSet = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 5.],
            func1 = testLinearCurve,
            func2 = testBetaTransferCurve1,
            expectedY = interp1d(
                pd.DataFrame({'XSet': [-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.]})['XSet'],
                pd.DataFrame({'func': [0., 0., 0., 1., 2., 3., 4., 5., 6., 6., 6.]})['func'],
                fill_value='extrapolate')))
        cases.append(TestMinPlusConvolutionClass(
            XSet = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 5.],
            YSet1 = [-4., -3., -2., -1., 0., 1., 2., 3., 4., 6.],
            YSet2 = [0., 0., 0., 0., 0., 3., 6., 9., 12., 18.],
            expectedXSet = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.],
            expectedY = [0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3.]))

        for case in cases:
            try:
                val1, val2 = ops.MinPlusDeconvolution(XSet  = case.XSet,
                                                      func1 = case.func1,
                                                      func2 = case.func2,
                                                      YSet1 = case.YSet1,
                                                      YSet2 = case.YSet2)

                self.assertTrue(np.array_equal(val1, case.expectedXSet))
                self.assertTrue(np.array_equal(val2, case.expectedY))

            except TypeError as e:
                val = ops.MinPlusDeconvolution(XSet=case.XSet,
                                               func1=case.func1,
                                               func2=case.func2,
                                               YSet1=case.YSet1,
                                               YSet2=case.YSet2)

                self.assertTrue(np.array_equal(val.x, case.expectedY.x))
                self.assertTrue(np.array_equal(val.y, case.expectedY.y))

    def test_SubAddClosure(self):
        class TestSubAddClosureClass:
            def __init__(self,
                         XSet: list,
                         func = None,
                         YSet: list = None,
                         amountConvolutions: int = 2,
                         expectedXSet: list = None,
                         expectedY: [interp1d, list, None] = None):
                self.XSet = XSet
                self.func = func
                self.YSet = YSet
                self.amountConvolutions = amountConvolutions
                self.expectedXSet = expectedXSet
                self.expectedY = expectedY

        cases = []
        cases.append(TestSubAddClosureClass(
            XSet=[1., 2., 3., 4., 5.],
        ))
        cases.append(TestSubAddClosureClass(
            XSet = [1., 2., 3., 4., 5.],
            func = testSquareCurve1,
            amountConvolutions = 0,
            expectedY = interp1d(
                pd.DataFrame({'XSet': [1.,  2.,  3.,  4.,  5.]})['XSet'],
                pd.DataFrame({'func': [1., 4., 9., 16., 25.]})['func'],
                fill_value='extrapolate')))
        cases.append(TestSubAddClosureClass(
            XSet = [1., 2., 3., 4., 5.],
            func = testSquareCurve1,
            amountConvolutions = 10,
            expectedY = interp1d(
                pd.DataFrame({'XSet': [1., 2., 3., 4., 5.]})['XSet'],
                pd.DataFrame({'func': [1., 4., 8., 12., 16.]})['func'],
                fill_value='extrapolate')))
        cases.append(TestSubAddClosureClass(
            XSet = [1., 2., 3., 4., 5.],
            YSet = [1., 4., 9., 16., 25.],
            amountConvolutions = 0,
            expectedXSet = [1., 2., 3., 4., 5.],
            expectedY = [1., 4., 9., 16., 25.],
        ))
        cases.append(TestSubAddClosureClass(
            XSet = [1., 2., 3., 4., 5.],
            YSet = [1., 4., 9., 16., 25.],
            amountConvolutions = 10,
            expectedXSet = [1., 2., 3., 4., 5.],
            expectedY = [1., 4., 8., 12., 16.],
        ))

        for case in cases:
            try:
                val1, val2 = ops.SubAddClosure(XSet = case.XSet,
                                               func = case.func,
                                               YSet = case.YSet,
                                               amountConvolutions = case.amountConvolutions)
                self.assertTrue(np.array_equal(val1, case.expectedXSet))
                self.assertTrue(np.array_equal(val2, case.expectedY))

            except TypeError as e:
                val = ops.SubAddClosure(XSet=case.XSet,
                                        func=case.func,
                                        YSet = case.YSet,
                                        amountConvolutions = case.amountConvolutions)
                self.assertTrue(np.array_equal(val.x, case.expectedY.x))
                self.assertTrue(np.array_equal(val.y, case.expectedY.y))