import numpy as np

# linearCurve is the method that describes curve: f(x) = kx + b.
def linearCurve(x, k: float = 1.0, b: float = 1.0):
    return k * np.float64(x) + b

# betaTransferCurve is the method that describes curve f(x):
# 0,        x <= T;
# R(x - T), x >= T.
def betaTransferCurve(R: float, T: float, x):
    if x <= T:
        return 0

    return R * (x - T)