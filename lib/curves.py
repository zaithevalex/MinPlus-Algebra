import numpy as np

# betaTransferCurve is the method that describes curve f(x):
# 0,        x <= T;
# R(x - T), x >= T.
def betaTransferCurve(x, R: float = 3., T: float = 5.):
    if x <= T:
        return 0

    return R * (x - T)

def betaTransferCurveShifted(x, R: float = 3., T: float = 5.):
    if x <= -T:
        return 0
    return R * (x - T) + 2 * R * T


# linearCurve is the method that describes curve: f(x) = kx + b.
def linearCurve(x, k: float = 1., b: float = 5.):
    return k * np.float64(x) + b

# squareCurve is the method that describes curve: f(x) = ax^2 + bx + c.
def squareCurve(x, a: float = 1., b: float = 0., c: float = 0.):
    return a * np.float64(x * x) + b * np.float64(x) + c