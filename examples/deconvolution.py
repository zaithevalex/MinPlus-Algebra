import matplotlib.pyplot as plt
import minplus_algebra as mpalgebra
import numpy as np

_, axs = plt.subplots(1, 2, figsize = (10, 5))

defArea = np.linspace(0, 50, 1000)
piecewiseCurve1YSet = np.array([mpalgebra.testPiecewiseCurve1(x) for x in defArea])
piecewiseCurve2YSet = np.array([mpalgebra.testPiecewiseCurve2(x) for x in defArea])

axs[0].plot(defArea, piecewiseCurve1YSet, label='piecewise curve 1', color='blue')
axs[0].plot(defArea, piecewiseCurve2YSet, color = 'green', label = 'piecewise curve 2')

x, y = mpalgebra.MinPlusDeconvolution(defArea, YSet1 = piecewiseCurve1YSet, YSet2 = piecewiseCurve2YSet)
axs[0].plot(x, y, color = 'red', label = 'minplus-deconvolution')

axs[0].set_title("datasets deconvolution")
axs[0].set_xlabel('x')
axs[0].set_ylabel('y(x)')
axs[0].grid()
axs[0].legend()

axs[1].plot(defArea, piecewiseCurve1YSet,
         color = 'blue',
         label = 'piecewise curve 1')
axs[1].plot(defArea, piecewiseCurve2YSet,
         color = 'green',
         label = 'piecewise curve 2')

f = mpalgebra.MinPlusDeconvolution(defArea, func1 = mpalgebra.testPiecewiseCurve1, func2 = mpalgebra.testPiecewiseCurve2)
axs[1].plot(defArea, np.array([f(x) for x in defArea]), color = 'red', label = 'minPlus-deconvolution')

axs[1].set_title("functions deconvolution")
axs[1].set_xlabel('x')
axs[1].set_ylabel('y(x)')
axs[1].grid()
axs[1].legend()

plt.show()