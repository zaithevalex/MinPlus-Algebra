import lib.curves as curves
import lib.operators as ops
import matplotlib.pyplot as plt
import numpy as np

defArea = np.linspace(0, 20, 1000)

plt.subplot(1, 2, 1)
plt.plot(defArea, np.array([curves.linearCurve(x, 1, 5) for x in defArea]),
         color = 'blue',
         label = 'linear curve')
plt.plot(defArea, np.array([curves.betaTransferCurve(x, 3, 5) for x in defArea]),
         color = 'green',
         label = 'beta transfer curve')

x, y = ops.MinPlusConvolution(defArea,
                              YSet1 = np.array([curves.betaTransferCurve(x, 3, 5) for x in defArea]),
                              YSet2 = np.array([curves.linearCurve(x, 1, 5) for x in defArea]))
plt.plot(x, y, color = 'red', label = 'minPlus-convolution')
plt.legend("datasets convolution")
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')

plt.subplot(1, 2, 2)
plt.plot(defArea, np.array([curves.linearCurve(x, 1, 5) for x in defArea]),
         color = 'blue',
         label = 'linear curve')
plt.plot(defArea, np.array([curves.betaTransferCurve(x, 3, 5) for x in defArea]),
         color = 'green',
         label = 'beta transfer curve')
f = ops.MinPlusConvolution(defArea, func1 = curves.betaTransferCurve, func2 = curves.linearCurve)
plt.plot(x, np.array([f(x) for x in defArea]), color = 'orange', label = 'minPlus-convolution')
plt.legend("functions convolution")
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')

plt.show()