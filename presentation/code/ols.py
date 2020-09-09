import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


np.random.seed(12)

n_samples = 100
x = np.linspace(0, 10, n_samples)

y = 25 + 2 * x**1.2 + 3 * np.sin(x) + np.random.normal(0, 0.5, n_samples)

model = np.column_stack((np.ones(n_samples), x**1.2, np.sin(x)))

fm = sm.OLS(y, model, missing='drop').fit()
predicted = fm.predict()

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot()

# ax.plot(x, y, label=r"$f(x) = 25 + 2 \cdot x^{1.2} + 3\cdot \sin(x) + \mathcal{N}(0, 0.5)$")
ax.plot(x, y, label=r"$f(x)$ with added Gaussian noise")
ax.plot(x, predicted, label="OLS prediction")
ax.legend()
plt.xlabel("x")
plt.ylabel("y")
# plt.show()


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.savefig("../imgs/ols1.png", bbox_inches ="tight")

print(fm.params)
