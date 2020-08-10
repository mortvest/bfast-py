import numpy as np
from scipy.stats import norm
from scipy.optimize import root, brentq

from recresid import recresid


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

        def extend_RSS_table(RSS_table, breaks):
            n, k = RSS_table.shape
            if (breaks * 2) > n:
                for(m in (n/2 + 1):breaks):
                    my_index = (m*h):(n-h)
                    my_RSS.table = RSS.table[,c((m-1)*2 - 1, (m-1)*2)]
                    my_RSS.table = cbind(my.RSS.table, NA, NA)
                    for(i in my.index):
                        pot_index = ((m-1)*h):(i - h)
                        break_RSS = sapply(pot_index, function(j) my.RSS.table[as.character(j), 2] + RSS(j+1,i))
                        opt = which_min(break_RSS)
                        my_RSS_table[as.character(i), 3:4] = c(pot.index[opt], break.RSS[opt])
                    RSS_table = cbind(RSS.table, my.RSS.table[,3:4])
            return(RSS_table)

        def extract_breaks(RSS_table, breaks):
            """
            extract optimal breaks
            """
            n, _ = RSS_table.shape
            if (breaks * 2) > n:
                raise ValueError("compute RSS_table with enough breaks before")

            index = RSS_table[:, 0]
            break_RSS = np.array([RSS_table[as.character(i),breaks*2] + RSS(i + 1, n) for i in index])

            opt = index[np.argmin(break_RSS)]
            if breaks > 1:
                for i in np.arange(breaks, 1, -1) - 1:
                    opt = [RSS_table[as.character(opt[1]), i], opt]
            return(opt)


        n, k = X.shape
        intercept_only = np.allclose(X, 1)

        h = np.floor(n * h)

        max_breaks = np.ceil(n / h) - 2
        if breaks is None:
            breaks = max_breaks
        elif breaks > max_breaks:
            print("requested number of breaks = {} too large, changed to n/h={}".
                  format(breaks, max_breaks))
            breaks = max_breaks

        ## compute optimal previous partner if observation i is the mth break
        ## store results together with RSSs in RSS_table

        ## breaks = 1
        RSS_triang = np.array([RSSi(i) for i in np.arange(:(n-h+1))], dtype=object)

        def RSS(i, j): return RSS_triang[i][j - i]

        index = np.arange((h-1), (n-h))
        break_RSS = np.array([RSS(0, i) for i in index])

        RSS_table = np.column_stack(index, break_RSS)

        ## breaks >= 2
        RSS_table = extend_RSS_table(RSS_table, breaks)
        opt = extract_breaks(RSS_table, breaks)

        self.breakpoints = opt
        self.nobs = n
        self.nreg = k
        self.y = y
        self.X = X
        self.RSS = RSS
        self.RSS_triang = RSS_triang
        self.RSS_table = RSS_table

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

    def logLik(self):
        """
        log-likelihood of the model
        """
        n = self.nobs
        bp = self.breakpoints
        df = (self.nreg + 1) * (len(bp[[~np.isnan(bp)]]) + 1)
        logL = -0.5 * n * (np.log(self.RSS) + 1 - np.log(n) + np.log(2 * np.pi))
        return (logL, df)


if __name__ == "__main__":
    x = -5.0
