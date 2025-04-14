import lib.curves as curves
import lib.operators as ops
import matplotlib.pyplot as plt
import numpy as np

_, axs = plt.subplots(1, 2, figsize = (10, 5))

defArea = np.linspace(0, 20, 1000)
axs[0].plot(defArea, np.array([curves.linearCurve(x, 1, 5) for x in defArea]),
            color = 'blue',
            label = 'linear curve')
axs[0].plot(defArea, np.array([curves.betaTransferCurve(x, 3, 5) for x in defArea]),
            color = 'green',
            label = 'beta transfer curve')

x, y = ops.MinPlusConvolution(defArea,
                              YSet1 = np.array([curves.betaTransferCurve(x, 3, 5) for x in defArea]),
                              YSet2 = np.array([curves.linearCurve(x, 1, 5) for x in defArea]))
axs[0].plot(x, y, color = 'red', label = 'minPlus-convolution')

axs[0].set_title("datasets convolution")
axs[0].set_xlabel('x')
axs[0].set_ylabel('y(x)')
axs[0].grid()
axs[0].legend()

axs[1].plot(defArea, np.array([curves.linearCurve(x, 1, 5) for x in defArea]),
            color = 'blue',
            label = 'linear curve')
axs[1].plot(defArea, np.array([curves.betaTransferCurve(x, 3, 5) for x in defArea]),
            color = 'green',
            label = 'beta transfer curve')

f = ops.MinPlusConvolution(defArea, func1 = curves.betaTransferCurve, func2 = curves.linearCurve)
axs[1].plot(x, np.array([f(x) for x in defArea]), color = 'red', label = 'minPlus-convolution')

axs[1].set_title("functions convolution")
axs[1].set_xlabel('x')
axs[1].set_ylabel('y(x)')
axs[1].grid()
axs[1].legend()

plt.show()