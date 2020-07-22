import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.sandbox.regression.predstd import wls_prediction_std
# from sklearn.linear_model import LinearRegression

from scipy.optimize import root, brentq

# h = 5

# for i in range(h+1):
#     print(i)

# def RSS(j, i):
#     return j + 2*i

# index = np.array([1,2,3,4])
# RSS_fun = lambda i: RSS(1, i)
# retval = [RSS(1, i) for i in index]
# print(retval)


fun = lambda x, y, z: 4*x + y + z


rt = brentq(fun, a=-10, b=10, args=(3,4))
print(rt)
