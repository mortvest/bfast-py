import numpy as np


x = np.array([[3, 4], [5, 6]])
xr = x[1]

res = (xr @ x @ xr)
print(res)
