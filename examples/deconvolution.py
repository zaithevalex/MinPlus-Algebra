import lib.curves as curves
import lib.operators as ops
import matplotlib.pyplot as plt
import numpy as np
import tests.operators_test

defArea = np.linspace(-20, 50, 1000)
piecewiseCurve1YSet = np.array([tests.operators_test.testPiecewiseCurve1(x) for x in defArea])
piecewiseCurve2YSet = np.array([tests.operators_test.testPiecewiseCurve2(x) for x in defArea])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')

plt.plot(defArea, piecewiseCurve1YSet, label='piecewise curve 1', color='blue')
plt.plot(defArea, piecewiseCurve2YSet, color = 'green', label = 'piecewise curve 2')

x, y = ops.MinPlusDeconvolution(defArea, YSet1 = piecewiseCurve1YSet, YSet2 = piecewiseCurve2YSet)
plt.plot(x, y, color = 'red', label = 'minplus-deconvolution')

plt.subplot(1, 2, 2)
plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')

plt.plot(defArea, piecewiseCurve1YSet,
         color = 'blue',
         label = 'piecewise curve 1')
plt.plot(defArea, piecewiseCurve2YSet,
         color = 'green',
         label = 'piecewise curve 2')

f = ops.MinPlusDeconvolution(defArea,
                             func1 = tests.operators_test.testPiecewiseCurve1,
                             func2 = tests.operators_test.testPiecewiseCurve2)
plt.plot(defArea, np.array([f(x) for x in defArea]), color = 'red', label = 'minPlus-deconvolution')

plt.show()