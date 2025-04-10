import lib.curves as curves
import lib.operators as ops
import matplotlib.pyplot as plt
import numpy as np

defArea = np.linspace(-20, 20, 1000)

plt.subplot(1, 2, 1)
plt.plot(defArea, np.array([curves.betaTransferCurveShifted(x, 3, 10) for x in defArea]),
         color = 'green',
         label = 'beta transfer curve')
plt.plot(defArea, np.array([curves.linearCurve(x, 1, 20) for x in defArea]),
         color = 'blue',
         label = 'linear curve')

x, y = ops.MinPlusDeconvolution(defArea,
                                YSet1 = np.array([curves.betaTransferCurveShifted(x, 3, 10) for x in defArea]),
                                YSet2 = np.array([curves.linearCurve(x, 1, 50) for x in defArea]))
plt.plot(x, y, color = 'red', label = 'minPlus-deconvolution')

plt.legend("datasets convolution")
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')

plt.subplot(1, 2, 2)
plt.plot(defArea, np.array([curves.betaTransferCurveShifted(x, 3, 10) for x in defArea]),
         color = 'green',
         label = 'beta transfer curve')
plt.plot(defArea, np.array([curves.linearCurve(x, 1, 20) for x in defArea]),
         color = 'blue',
         label = 'linear curve')

f = ops.MinPlusDeconvolution(defArea,
                             func1 = curves.betaTransferCurveShifted,
                             func2 = curves.linearCurve)
plt.plot(defArea, np.array([f(x) for x in defArea]), color = 'red', label = 'minPlus-deconvolution')

plt.legend("functions convolution")
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')

plt.show()