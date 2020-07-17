import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LinearRegression


X = np.arange(1,21).reshape(20,1)
y = 1.2 * X
y[10:] += 5


X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit(method="qr")
print(results.summary())

reg = LinearRegression(fit_intercept=True).fit(X, y)
print(reg.coef_, reg.intercept_)

