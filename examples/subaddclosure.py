import lib.curves as curves
import lib.operators as ops
import matplotlib.pyplot as plt
import numpy as np

_, axs = plt.subplots(1, 2, figsize=(10, 5))

defArea = np.linspace(0, 20, 1000)

axs[0].plot(defArea, np.array([curves.squareCurve(x, 1, 0, 0) for x in defArea]),
            color = 'blue',
            label = 'square curve')

x, y = ops.SubAddClosure(defArea,
                         YSet = np.array([curves.squareCurve(x, 1, 0, 0) for x in defArea]),
                         amountConvolutions = 1)
axs[0].plot(x, y, color = 'green', label = '1st degree sub-add closure')

x, y = ops.SubAddClosure(defArea,
                         YSet = np.array([curves.squareCurve(x, 1, 0, 0) for x in defArea]),
                         amountConvolutions = 2)
axs[0].plot(x, y, color = 'red', label = '2nd degree sub-add closure')

axs[0].set_title("datasets sub-add closure")
axs[0].set_xlabel('x')
axs[0].set_ylabel('y(x)')
axs[0].grid()
axs[0].legend()

axs[1].plot(defArea, np.array([curves.squareCurve(x, 1, 0, 0) for x in defArea]),
         color = 'blue',
         label = 'square curve')

f = ops.SubAddClosure(defArea,
                      func = curves.squareCurve,
                      amountConvolutions = 1)
axs[1].plot(defArea, np.array([f(x) for x in defArea]), color = 'green', label = '1st degree sub-add closure')

f = ops.SubAddClosure(defArea,
                      func = curves.squareCurve,
                      amountConvolutions = 2)
axs[1].plot(defArea, np.array([f(x) for x in defArea]), color = 'red', label = '2nd degree sub-add closure')

axs[1].set_title("functions sub-add closure")
axs[1].set_xlabel('x')
axs[1].set_ylabel('y(x)')
axs[1].grid()
axs[1].legend()

plt.show()