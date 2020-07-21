import numpy as np
import matplotlib.pyplot as plt

import datasets
import utils


class EFP():
    def __init__(self, X, y, h, deg=1, p_type="OLS-MOSUM"):
        """
        Empirical fluctuation process. For now, only the Ordinary Least Squares MOving
        SUM (OLS-MOSUM) is supported

        :param X: matrix of x-values
        :param y: vector of y
        :param h: bandwidth parameter for the MOSUM process
        :param deg: degree of the polynomial to be fit [0,1]
        :param p_type: process type. Only OLS-MOSUM is supported
        :returns: instance of Empirical Fluctuation Process
        :raises ValueError: wrong type of process
        """
        if p_type != "OLS-MOSUM":
            raise ValueError("Process type {} is not supported".format(p_type))

        n, k = X.shape

        # fit linear model
        fm = np.polyfit(X.flatten(), y, deg=deg)

        # residuals
        if deg==0:
            e = y - fm[0]
        else:
            e = y - fm @ np.vstack((X.T, np.ones(n)))

        sigma = np.sqrt(np.sum(e**2) / (n - k))
        nh = np.floor(n * h)

        e_zero = np.insert(e, 0, 0)

        process = np.cumsum(e_zero)
        process = process[int(nh):] - process[:(n - int(nh) + 1)]
        process = process / (sigma * np.sqrt(n))

        end_t = int(n - np.floor(0.5 + nh / 2))
        process = process[:end_t]

        self.coefficients = fm
        self.sigma = sigma
        self.process = process
        self.par = h


    def p_value(x, h, k, max_k=6, table_dim=10):
        """
        Returns the p value for the process.

        :param x: result of application of the functional
        :param h: bandwidth parameter
        :param k: number of rows of matrix X
        :returns: p value for the process
        """
        k = min(k, max_k)
        # print(k)
        crit_table = utils.sc_me[((k - 1) * table_dim):(k * table_dim),:]
        print("crit_table", crit_table)
        tablen = crit_table.shape[1]
        print("tablen", tablen)
        tableh = np.arange(1, table_dim + 1) * 0.05
        print("tableh", tableh)
        tablep = np.array((0.1, 0.05, 0.025, 0.01))
        tableipl = np.zeros(tablen)

        for i in range(tablen):
            tableipl[i] = np.interp(h, tableh, crit_table[:, i])

        print(tableipl)
        tableipl = np.insert(tableipl, 0, 0)
        tablep = np.insert(tablep, 0, 1)

        print(x)
        p = np.interp(x, tableipl, tablep)

        return(p)

    def sctest(self, functional="max"):
        """
        Performs a generalized fluctuation test.

        :param functional: functional type. Only max is supported
        :raises ValueError: wrong type of functional
        :returns: a tuple of applied functional and p value
        """
        if functional != "max":
            raise ValueError("Functional {} is not supported".format(functional))

        h = self.par
        x = self.process
        if functional=="max":
            k = 1
        stat = np.max(np.abs(x))
        p_value = EFP.p_value(stat, h, k)

        return(stat, p_value)


if __name__ == "__main__":
    # y = datasets.nhtemp
    y = datasets.nile
    x = np.arange(1, y.shape[0] + 1).reshape(y.shape[0], 1)

    # x = np.arange(1,21).reshape(20,1)
    # y = 1.2 * x
    # y[10:] += 5

    # value in BFAST
    h = 0.15

    efp = EFP(x, y, h)
    stat, p_value = efp.sctest()
    level = 0.05

    if p_value <= level:
        print("Breakpoint detected")
    else:
        print("No breakpoint detected")

    print("p_value", p_value)
    print("stat", stat)

