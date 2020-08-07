import numpy as np
from scipy.stats import norm
from scipy.optimize import root, brentq

from recresid import recresid


class Breakpoints():
    def __init__(self, X, y, h=0.15, breaks=None, data=[]):
        def RSS(i, j):
            """
            function to extract the RSS(i,j) from RSS.triang
            """
            RSS_triang = np.array([RSSi(i) for i in np.arange(0:(n-h+1))])
            return(RSS.triang[[i]][j - i + 1])

        def RSSi(i):
            """
            compute ith row of the RSS diagonal matrix, i.e,
            the recursive residuals for segments starting at i = 1:(n-h+1)
            """
            if intercept_only:
                ssr = (y[i:n] - np.cumsum(y[i:n])/(1L:(n-i+1L)))[-1L] * np.sqrt(1L + 1L/(1L:(n-i)))
            else:
                ssr = recresid(X[i:n,,drop = FALSE],y[i:n])
            return [np.repeat(np.nan, k), np.cumsum(ssr**2)]

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

        n, k = X.shape
        intercept_only = np.allclose(X, 1)

        h = np.floor(n * h)

        max_breaks = np.ceil(n / h) - 2
        if breaks is None:
            breaks = max_breaks
        elif breaks > max_breaks:
            print("requested number of breaks = {} too large, changed to {}".format(breaks, max_breaks))
            breaks = max_breaks

        ## compute optimal previous partner if observation i is the mth break
        ## store results together with RSSs in RSS_table

        ## breaks = 1
        index = np.arange((h-1), (n-h))

        break_RSS = np.array([RSS(1, i) for i in index])

        RSS_table = np.column_stack(index, break_RSS)

        ## breaks >= 2
        RSS_table = Breakpoints.extend_RSS_table(RSS_table, breaks)
        opt = Breakpoints.extract_breaks(RSS_table, breaks)

        self.breakpoints = opt
        self.nobs = n
        self.nreg = k
        self.y = y
        self.X = X


    def breakfactor(self):
        breaks = self.breakpoints
        nobs = self.nobs
        if np.isnan(breaks).all():
           # return(np.repeat("segment1", self.nobs))
           return (np.repeat(1, nobs), np.array(["segment1"]))

        nbreaks = breaks.shape[0]
        v = np.insert(np.diff(np.append(breaks, nobs)), 0, breaks[0])
        fac = np.repeat(np.arange(1, nbreaks + 2), v)
        labels = np.array(["segment" + str(i) for i in range(1, nbreaks + 2)])
        # labels[fac-1]
        return(fac, labels)


if __name__ == "__main__":
    x = -5.0
