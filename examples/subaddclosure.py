import lib.curves as curves
import lib.operators as ops
import matplotlib.pyplot as plt
import numpy as np

defArea = np.linspace(0, 20, 1000)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.legend("dataset convolution")
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')

plt.plot(defArea, np.array([curves.squareCurve(x, 1, 0, 0) for x in defArea]),
         color = 'blue',
         label = 'square curve')

x, y = ops.SubAddClosure(defArea,
                         YSet = np.array([curves.squareCurve(x, 1, 0, 0) for x in defArea]),
                         amountConvolutions = 1)
plt.plot(x, y, color = 'green', label = '1st degree sub-add closure')

x, y = ops.SubAddClosure(defArea,
                         YSet = np.array([curves.squareCurve(x, 1, 0, 0) for x in defArea]),
                         amountConvolutions = 2)
plt.plot(x, y, color = 'red', label = '2nd degree sub-add closure')

plt.subplot(1, 2, 2)
plt.legend("function convolution")
plt.grid()
plt.xlabel('x')
plt.ylabel('y(x)')

plt.plot(defArea, np.array([curves.squareCurve(x, 1, 0, 0) for x in defArea]),
         color = 'blue',
         label = 'square curve')

f = ops.SubAddClosure(defArea,
                      func = curves.squareCurve,
                      amountConvolutions = 1)
plt.plot(defArea, np.array([f(x) for x in defArea]), color = 'green', label = '1st degree sub-add closure')

f = ops.SubAddClosure(defArea,
                      func = curves.squareCurve,
                      amountConvolutions = 2)
plt.plot(defArea, np.array([f(x) for x in defArea]), color = 'red', label = '2nd degree sub-add closure')

plt.show()