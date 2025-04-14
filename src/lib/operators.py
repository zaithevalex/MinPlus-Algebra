import numpy as np
import pandas as pd
import scipy.integrate as spi
from scipy.interpolate import interp1d

# AddConst adds the constant to the func or dataset (YSet).
def AddConst(XSet: list, func = None, YSet: list = None, const = 0.):
    if func is not None:
        YSet = np.array([(func(x) + const) for x in XSet])
        return XSet, ConvertDataSetToLinearFunction(np.linspace(XSet[0], XSet[len(XSet) - 1], len(YSet)), YSet)

    elif YSet is not None:
        for i in range(len(YSet)):
            YSet[i] += const
        return np.linspace(XSet[0], XSet[len(XSet) - 1], len(YSet)), YSet

    else:
        return None, None

# ConvertDataSetToLinearFunction converts data from an array(XSet, YSet) to a piecewise linear function (f(x) = kx + b).
def ConvertDataSetToLinearFunction(XSet: list, YSet: list):
    if len(XSet) == 0:
        raise ValueError("XSet must have length > 0")

    elif len(YSet) == 0:
        raise ValueError("YSet must have length > 0")

    elif len(XSet) != len(YSet):
        raise ValueError("XSet and YSet must have same length")

    x_df = {'XSet' : XSet}
    func_df = {'func' : YSet}

    return interp1d(pd.DataFrame(x_df)['XSet'], pd.DataFrame(func_df)['func'], fill_value='extrapolate')

def ConvertFunctionToDataSet(XSet: list, func):
    YSet = np.array([func(x) for x in XSet])
    return np.linspace(XSet[0], XSet[len(XSet) - 1], len(YSet)), YSet

# L1Norm searches l1 norm: ||f - g||1 for two given functions(func1, func2) on the domain of definition(XSet).
def L1Norm(XSet: list, func1, func2):
    return spi.simpson(np.abs(func1(XSet) - func2(XSet)), XSet)

# MinimizeL1Norm solves the optimization problem by minimizing the L1 norm for two functions(func1, func2)
# а within a given range(start from bottom_border to top_border) with given step
# on the domain of definition(XSet).
def MinimizeL1Norm(XSet: list, func1, func2, bottom_border=0., top_border=0., step=0.):
    Wmin, mn = float("+inf"), float("+inf")

    while bottom_border <= top_border:
        x, f = AddConst(XSet, func=func1, const=bottom_border)
        norm = L1Norm(x, f, func2)

        if norm < mn:
            mn = norm
            Wmin = bottom_border

        bottom_border += step

    return Wmin

# MinPlusConvolution collapses two functions(func1, func2) or two datasets(YSet1, YSet2)
# according to the inf{0 <= s <= t | func1(t + s) - func2(s)} rule  on the domain of definition(XSet).
def MinPlusConvolution(XSet: list, func1 = None, func2 = None, YSet1: list = None, YSet2: list = None):
    if XSet[0] < 0:
        raise ValueError("left border of the domain of definition interval must be positive(> 0), but expected:", XSet[0])

    if func1 is not None and func2 is not None:
        c = convolve(np.array([func1(x) for x in XSet]), np.array([func2(x) for x in XSet]))
        return ConvertDataSetToLinearFunction(XSet, c)

    elif YSet1 is not None and YSet2 is not None:
        c = convolve(YSet1, YSet2)
        return np.linspace(XSet[0], XSet[len(XSet) - 1], len(c)), c

    else:
        return None, None

# MinPlusConvolution is operation of the type sup{u >= 0 | func1(t + u) - func2(u)}
# for two functions on the domain of definition(XSet).
def MinPlusDeconvolution(XSet: list, func1 = None, func2 = None, YSet1: list = None, YSet2: list = None):
    if func1 is not None and func2 is not None:
        dec_x, dec_y = deconvolve(XSet,
                                  ConvertDataSetToLinearFunction(XSet, np.array([func1(x) for x in XSet])),
                                  ConvertDataSetToLinearFunction(XSet, np.array([func2(x) for x in XSet])))
        return ConvertDataSetToLinearFunction(dec_x, dec_y)

    elif YSet1 is not None and YSet2 is not None:
        return deconvolve(XSet, ConvertDataSetToLinearFunction(XSet, YSet1), ConvertDataSetToLinearFunction(XSet, YSet2))

    else:
        return None, None

# SubAddClosure is operation of the type:
# delta_0 ^ func ^ MinPlusConvolution(func, func) ^ MinPlusConvolution(func, MinPlusConvolution(func, func)) ^ ...
# where func is func or YSet.
# amountConvolutions is amount of convolutions current func or YSet.
def SubAddClosure(XSet: list, func = None, YSet: list = None, amountConvolutions: int = 3):
    if func is not None:
        func_set = np.array([func(x) for x in XSet])
        for _ in range(amountConvolutions):
            func_set = convolve(func_set, func_set)

        return ConvertDataSetToLinearFunction(np.linspace(XSet[0], XSet[len(XSet) - 1], len(func_set)), func_set)

    elif YSet is not None:
        for _ in range(amountConvolutions):
            YSet = convolve(YSet, YSet)

        return np.linspace(XSet[0], XSet[len(XSet) - 1], len(YSet)), YSet

    else:
        return None, None

# convolve is general method of convolution cases.
def convolve(YSet1, YSet2):
    convSet = []
    for i in range(len(YSet1)):
        tmp = []
        for j in range(i + 1):
            if j == 0:
                tmp.append(YSet1[i - j])
            else:
                tmp.append(YSet1[i - j] + YSet2[j])

        convSet.append(min(tmp))

    return convSet

# deconvolve is general method of deconvolution cases.
def deconvolve(XSet: list, func1, func2):
    deconvolvedSet = []
    external = XSet[0]
    step = (XSet[len(XSet) - 1] - external) / len(XSet)

    while external <= XSet[len(XSet) - 1]:
        tmp = [0] * (len(XSet) + 1)

        if external <= 0:
            internal = 0.0
            internalCounter = 0
            while internal <= XSet[len(XSet) - 1]:
                tmp[internalCounter] = func1(external + internal) - func2(internal)
                internalCounter += 1
                internal += step
        else:
            internal = 0.0
            internalCounter = 0
            while internal <= XSet[len(XSet) - 1] - external:
                tmp[internalCounter] = func1(external + internal) - func2(internal)
                internalCounter += 1
                internal += step

        deconvolvedSet.append(max(tmp))
        external += step

    return np.linspace(XSet[0], XSet[len(XSet) - 1], len(deconvolvedSet)), deconvolvedSet
