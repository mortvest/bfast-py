import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from recresid import recresid
from datasets import nile, nile_dates, uk_driver_deaths, uk_driver_deaths_dates


class Breakpoints():
    def __init__(self, X, y, h=0.15, breaks=None):
        def RSSi(i):
            """
            Compute i'th row of the RSS diagonal matrix, i.e,
            the recursive residuals for segments starting at i = 1:(n-h+1)
            """
            if intercept_only:
                arr1 = np.arange(1, (n-i+1))
                arr2 = arr1[:-1]
                ssr = (y[i:] - np.cumsum(y[i:]) / arr1)[1:] * np.sqrt(1 + 1 / arr2)
            else:
                ssr = recresid(X[i:], y[i:])
            return np.concatenate((np.repeat(np.nan, k), np.cumsum(ssr**2)))

        n, k = X.shape
        self.nobs = n
        logging.debug("n = {}, k = {}".format(n, k))

        intercept_only = np.allclose(X, 1)
        logging.debug("intercept_only = {}".format(intercept_only))

        h = int(np.floor(n * h))
        self.h = h
        logging.debug("h = {}".format(h))

        max_breaks = int(np.ceil(n / h) - 2)
        if breaks is None:
            breaks = max_breaks
        elif breaks > max_breaks:
            logging.warning("requested number of breaks = {} too large, changed to {}".
                            format(breaks, max_breaks))
            breaks = max_breaks

        logging.debug("breaks = {}".format(breaks))
        ## compute optimal previous partner if observation i is the mth break
        ## store results together with RSSs in RSS_table

        ## breaks = 1

        self.RSS_triang = np.array([RSSi(i) for i in np.arange(n-h+1).astype(int)], dtype=object)

        logging.debug("RSS_triang:\n{}".format(self.RSS_triang))


        index = np.arange((h - 1), (n - h)).astype(int)
        logging.debug("index:\n{}".format(index))

        break_RSS = np.array([self.RSS(0, i) for i in index])
        logging.debug("break_RSS:\n{}".format(break_RSS))

        RSS_table = np.column_stack((index, break_RSS))
        logging.debug("RSS_table:\n{}".format(RSS_table))

        ## breaks >= 2
        RSS_table = self.extend_RSS_table(RSS_table, breaks)
        logging.debug("extended RSS_table:\n{}".format(RSS_table))

        opt = self.extract_breaks(RSS_table, breaks).astype(int)
        logging.debug("breakpoints extracted:\n{}".format(opt))

        self.breakpoints = opt
        self.nreg = k
        self.y = y
        self.X = X

        # find the optimal amount of breakpoints using Bayesian Information Criterion
        self.breakpoints_for_m()

    def RSS(self, i, j):
        return self.RSS_triang[i][j - i]

    def extend_RSS_table(self, RSS_table, breaks):
        _, ncol = RSS_table.shape
        h = self.h
        n = self.nobs
        if (breaks * 2) > ncol:
            for m in range((int(ncol/2) + 1), breaks - 1, -1):
                my_index = np.arange((m * h) - 1, (n - h))
                index_arr = np.arange((m-1)*2 - 2, (m-1)*2)
                my_RSS_table = RSS_table[:, index_arr]
                nans = np.repeat(np.nan, my_RSS_table.shape[0])
                my_RSS_table = np.column_stack((my_RSS_table, nans, nans))
                for i in my_index:
                    pot_index = np.arange((m - 1) * h - 1, (i - h + 1)).astype(int)
                    fun = lambda j: my_RSS_table[j - h + 1, 1] + self.RSS(j + 1, i)
                    break_RSS = np.vectorize(fun)(pot_index)
                    opt = np.nanargmin(break_RSS)
                    my_RSS_table[i - h + 1, np.array((2, 3))] = np.array((pot_index[opt], break_RSS[opt]))
                RSS_table = np.column_stack((RSS_table, my_RSS_table[:, np.array((2,3))]))
        return(RSS_table)

    def extract_breaks(self, RSS_table, breaks):
        """
        extract optimal breaks
        """
        _, ncol = RSS_table.shape
        n = self.nobs
        h = self.h

        if (breaks * 2) > ncol:
            raise ValueError("compute RSS_table with enough breaks before")

        index = RSS_table[:, 0].astype(int)
        fun = lambda i: RSS_table[int(i - self.h + 1), int(breaks * 2 - 1)] + self.RSS(i + 1, n - 1)
        break_RSS = np.vectorize(fun)(index)
        opt = [index[np.nanargmin(break_RSS)]]

        if breaks > 1:
            for i in 2 * np.arange(breaks, 1, -1) - 2:
                opt.insert(0, RSS_table[opt[0] - h + 1, i])
        return(np.array(opt))

    def logLik(self):
        """
        log-likelihood of the model
        """
        n = self.nobs
        bp = self.breakpoints
        df = (self.nreg + 1) * (len(bp[[~np.isnan(bp)]]) + 1)
        logL = -0.5 * n * (np.log(self.RSS) + 1 - np.log(n) + np.log(2 * np.pi))
        return (logL, df)

    def breakpoints_for_m(self, breaks=None):
        if breaks is None:
            sbp = self.summary
            breaks = np.argmin(sbp.RSS[1,:]) - 1
        if breaks < 1:
            return None
        else:
            RSS_tab = self.extend_RSS_table(self.RSS_table, breaks)
            breakpoints = self.extract_breaks(RSS_tab, breaks)
            self.breakpoints = breakpoints

    def summary(self, breaks=None, sort=True, format_times=None):
        if breaks is None:
            breaks = self.RSS_table.shape[1]/2

        n = self.nobs
        RSS = c(self.RSS(1, n), rep(NA, breaks))
        BIC = c(n * (log(RSS[1]) + 1 - log(n) + log(2*pi)) + log(n) * (object$nreg + 1),
                rep(NA, breaks))
        bp = self.breakpoints_for_m(breaks)
        RSS[breaks + 1] = bp$RSS
        BIC[breaks + 1] = AIC(bp, k = log(n))
        bp = bp.breakpoints

        if breaks > 1:
            for m in range(breaks - 1, 0, -1):
                bpm = breakpoints(object, breaks=m)
                RSS[m+1] = bpm$RSS
                BIC[m+1] = AIC(bpm, k = log(n))
        RSS = np.vstack((RSS, BIC))


    def breakfactor(self):
        breaks = self.breakpoints
        nobs = self.nobs
        if np.isnan(breaks).all():
            return (np.repeat(1, nobs), np.array(["segment1"]))

        nbreaks = breaks.shape[0]
        v = np.insert(np.diff(np.append(breaks, nobs)), 0, breaks[0])
        fac = np.repeat(np.arange(1, nbreaks + 2), v)
        labels = np.array(["segment" + str(i) for i in range(1, nbreaks + 2)])
        # labels[fac-1]
        return(fac, labels)


if __name__ == "__main__":
    # set up logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",
                        default="warning",
                        help="set the logging level, default is WARNING")
    args = parser.parse_args()
    log_level = args.log
    logging.basicConfig(level=getattr(logging, log_level.upper()))


    print("Testing synthetic")
    # Synthetic dataset with two breakpoints x = 15 and 35
    X = np.ones(50).reshape((50, 1))
    y = np.arange(50) * 2
    y[15:] += 30
    y[35:] += 25

    n_breaks = 2

    bp = Breakpoints(X, y, breaks=n_breaks).breakpoints
    print("Breakpoints:", bp + 1)
    print()


    # Nile dataset with a single breakpoint. Ashwan dam was built in 1898
    print("Testing nile")

    y = nile
    X = np.ones(y.shape[0]).reshape((y.shape[0], 1))

    nile_breaks = 1
    bp_nile = Breakpoints(X, y, breaks=1)
    bp_nile_arr = bp_nile.breakpoints
    bp_nile_bf = bp_nile.breakfactor()[0]

    # plt.plot(nile_dates[bp_nile_bf == 1], nile[bp_nile_bf == 1])
    # plt.plot(nile_dates[bp_nile_bf == 2], nile[bp_nile_bf == 2])
    # plt.show()


    # bp_nile = Breakpoints(X, y, breaks=nile_breaks).breakpoints
    nile_break_date = nile_dates[bp_nile_arr]
    print("Breakpoints:", nile_break_date)
    print()

    # UK Seatbelt data. Has at least two break points: one in 1973 and one in 1983
    print("Testing UK Seatbelt data")

    y = uk_driver_deaths
    X = np.ones(y.shape[0]).reshape((y.shape[0], 1))
    uk_breaks = 2

    # plt.plot(uk_driver_deaths_dates, uk_driver_deaths)
    # plt.show()

    bp_uk = Breakpoints(X, y, breaks=uk_breaks).breakpoints
    uk_break_dates = uk_driver_deaths_dates[bp_uk]
    if uk_break_dates.shape[0] > 0:
        print("Breakpoints", uk_break_dates)


    # # seatbelt <- cbind(seatbelt, lag(seatbelt, k = -1), lag(seatbelt, k = -12))
