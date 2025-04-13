import matplotlib.pyplot as plt
import minplus_algebra as mpalg
import numpy as np

_, axs = plt.subplots(1, 2, figsize = (10, 5))

defArea = np.linspace(0, 20, 1000)
axs[0].plot(defArea, np.array([mpalg.linearCurve(x, 1, 5) for x in defArea]),
         color = 'blue',
         label = 'linear curve')
axs[0].plot(defArea, np.array([mpalg.betaTransferCurve(x, 3, 5) for x in defArea]),
         color = 'green',
         label = 'beta transfer curve')

x, y = mpalg.MinPlusConvolution(defArea,
                              YSet1 = np.array([mpalg.betaTransferCurve(x, 3, 5) for x in defArea]),
                              YSet2 = np.array([mpalg.linearCurve(x, 1, 5) for x in defArea]))
axs[0].plot(x, y, color = 'red', label = 'minPlus-convolution')

axs[0].set_title("datasets convolution")
axs[0].set_xlabel('x')
axs[0].set_ylabel('y(x)')
axs[0].grid()
axs[0].legend()

axs[1].plot(defArea, np.array([mpalg.linearCurve(x, 1, 5) for x in defArea]),
            color = 'blue',
            label = 'linear curve')
axs[1].plot(defArea, np.array([mpalg.betaTransferCurve(x, 3, 5) for x in defArea]),
            color = 'green',
            label = 'beta transfer curve')

f = mpalg.MinPlusConvolution(defArea, func1 = mpalg.betaTransferCurve, func2 = mpalg.linearCurve)
axs[1].plot(x, np.array([f(x) for x in defArea]), color = 'red', label = 'minPlus-convolution')

axs[1].set_title("functions convolution")
axs[1].set_xlabel('x')
axs[1].set_ylabel('y(x)')
axs[1].grid()
axs[1].legend()

plt.show()