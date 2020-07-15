import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import utils


class EFP():
    """
    Empirical fluctuation process. For now, only the Ordinary Least Squares MOving
    SUM (OLS-MOSUM) test is supported
    """
    def __init__(self, X, y, h, p_type="OLS-MOSUM", functional="max"):
        if p_type != "OLS-MOSUM":
            raise ValueError("Process type {} is not supported".format(p_type))

        if functional != "max":
            raise ValueError("Functional {} is not supported".format(functional))

        n, k = X.shape
        fm = utils.qr_lm(X, y)

        # residuals
        # reg = LinearRegression().fit(X, y)
        # e = y - reg.predict(X)
        e = y - X@fm

        sigma = np.sqrt(np.sum(e**2)/(n-k))
        nh = np.floor(n*h)

        e_zero = np.insert(e, 0, 0)

        process = np.cumsum(e_zero)
        process = process[int(nh):] - process[:(n-int(nh)+1)]
        process = process / (sigma * np.sqrt(n))

        end_t = int(n - np.floor(0.5 + nh / 2))
        process = process[:end_t]

        # self.coefficients = reg.coef_
        self.coefficients = fm
        self.sigma = sigma
        self.process = process
        self.par = h


    def p_value(x, h, k, max_k=6, table_dim=10):
        k = min(k, max_k)
        crit_table = utils.sc_me[((k - 1) * table_dim):(k * table_dim),:]
        tablen = crit_table.shape[1]
        tableh = np.arange(1, table_dim + 1) * 0.05
        tablep = np.array((0.1, 0.05, 0.025, 0.01))
        tableipl = np.zeros(tablen)

        for i in range(tablen):
            tableipl[i] = np.interp(h, tableh, crit_table[:, i])

        tableipl = np.insert(tableipl, 0, 0)
        tablep = np.insert(tablep, 0, 1)

        p = np.interp(x, tableipl, tablep)

        return(p)

    def sctest(self):
        h = self.par
        x = self.process
        # _, k = x.shape
        k = 1
        stat = np.max(np.abs(x))
        p_value = EFP.p_value(stat, h, k)

        return(stat, p_value)
        # return(p_value)


if __name__ == "__main__":
    x = np.arange(1,21).reshape(20,1)
    y = 1.2 * x
    y[10:] = (1.2 * x)[10:]
    efp = EFP(x, y, 0.15)
    p_value = efp.sctest()[1]
    level = 0.05

    if p_value <= level:
        print("Breakpoint detected")
        print("p_value", p_value)
        print("level", level)
    else:
        print("no breakpoint detected")

    plt.plot(x.reshape(20), y)
    plt.show()


