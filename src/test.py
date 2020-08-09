import numpy as np
import scipy.linalg

from sklearn.linear_model import LinearRegression



# x = np.arange(1, 21)
# y = 2 * x
# y[10:] += 10
# x = np.column_stack((x, np.ones(20)))

x = np.array((1,2,3,4))
y = 2 * x
y = y + 8
x = np.column_stack((x, np.ones(4)))
# print(x)
# print(y)

reg = LinearRegression(fit_intercept=False).fit(x, y)
