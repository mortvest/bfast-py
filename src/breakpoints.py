import numpy as np
from scipy.stats import norm
from scipy.optimize import root, brentq


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
                    opt = [(RSS_table[as.character(opt[1]),i], opt]
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
                        my_RSS_table[as.character(i), 3:4] <- c(pot.index[opt], break.RSS[opt])
                    RSS_table = cbind(RSS.table, my.RSS.table[,3:4])
            return(RSS_table)

        n, k = X.shape
        intercept_only = np.allclose(X, 1)

        h = np.floor(n * h)

        max_breaks = np.ceil(n / h) - 2
        if breaks is None:
            breaks = max_breaks
        elif breaks > max_breaks:
            breaks0 = breaks
            print("requested number of breaks = {} too large, changed to {}".format(breaks0, max_breaks))
            breaks = max_breaks

        ## compute optimal previous partner if observation i is the mth break
        ## store results together with RSSs in RSS_table

        ## breaks = 1
        index = np.arange((h-1), (n-h))

        break_RSS = np.array([Breakpoints.RSS(1, i, n, h) for i in index])

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
        if np.isnan(breaks).all():
            return(np.repeat("segment1", self.nobs))
        nbreaks = breaks.shape[0]
        fac = rep(1:(nbreaks + 1), c(breaks[1], diff(c(breaks, obj$nobs))))

        labels = paste("segment", 1:(nbreaks+1), sep = "")
        fac = factor(fac, labels=labels)
        return(fac)

    def residuals(self, deg=1):
        X = self.X
        y = self.y
        n = self.nobs
        obp = self.breakpoints
        bp = np.copy(obp)

        if np.isnan(bp).any():
            nbp = 0
            bp = np.array((0, n))
        else:
            nbp = bp.shape[0]
            bp = np.array((0, bp, n))

        rval = []
        for i in range(nbp+1):
            X2 = X[(bp[i]+1):bp[i+1],,drop = FALSE]
            y2 = y[(bp[i]+1):bp[i+1]]

            fm = np.polyfit(X2.flatten(), y2, deg=deg)

            # residuals
            if deg == 0:
                residuals = y - fm[0]
            else:
                residuals = y - fm @ np.vstack((X.T, np.ones(X2.shape[0])))

            rval.append(residuals)

        rval = np.array(rval)
        return(np.rval)

    def confint(self, level=0.95):
        X = self.X
        y = self.y
        n = self.nobs
        a2 = (1 - level) / 2

        myprod = function(delta, mat) as.vector(crossprod(delta, mat) %*% delta)

        bp = self.breakpoints

        if np.isnan(bp).any():
            raise ValueError("cannot compute confidence interval when `breaks = 0'")

        nbp = bp.shape[0]
        upper = np.repeat(0, nbp)
        lower = np.repeat(0, nbp)
        bp = [0, bp, n]

        res = self.residuals()
        sigma1 = np.sum(res ** 2) / n
        sigma2 = np.copy(sigma1)

        Q1 = crossprod(X) / n
        Q2 = np.copy(Q1)

        Omega1 = sigma1 * Q1
        Omega2 = np.copy(Omega1)

        xi = 1

        X2 = X[(bp[1]+1):bp[2],,drop = FALSE]
        y2 = y[(bp[1]+1):bp[2]]
        fm2 = lm(y2 ~ 0+ X2)
        beta2 = coef(fm2)
        Q2 = crossprod(X2) / nrow(X2)

        for i in range(2, nbp+1):
            X1 = X2
            y1 = y2
            beta1 = beta2
            sigma1 = sigma2
            Q1 = Q2
            Omega1 = Omega2

            X2 = X[(bp[i]+1):bp[i+1],,drop = FALSE]
            y2 = y[(bp[i]+1):bp[i+1]]
            fm2 = lm(y2 ~ 0 + X2)
            beta2 = coef(fm2)
            delta = beta2 - beta1

            Q2 = crossprod(X2) / nrow(X2)

            Oprod1 = myprod(delta, Omega1)
            Oprod2 = myprod(delta, Omega2)
            Qprod1 = myprod(delta, Q1)
            Qprod2 = myprod(delta, Q2)

            xi = Qprod2 / Qprod1
            phi1 = np.sqrt(sigma1)
            phi2 = np.sqrt(sigma2)

            p0 = pargmaxV(0, phi1=phi1, phi2=phi2, xi=xi)

            if np.isnan(p0) or p0 < a2 or p0 > (1 - a2):
                print("Confidence interval {} cannot be computed: P(argmax V <= 0) = {}".format(i-1, p0))
                upper[i-1] = np.nan
                lower[i-1] = np.nan
            else:
                ub, lb = 0, 0
                while pargmaxV(ub, phi1=phi1, phi2=phi2, xi=xi) < (1 - a2):
                    ub += 1000

                while pargmaxV(lb, phi1=phi1, phi2=phi2, xi=xi) > a2:
                    lb -= 1000

                # find roots
                myfun = lambda x, level, xi, phi1, phi2 : pargmaxV(x, xi, phi1, phi2) - level
                upper[i-1] = brentq(myfun, a=0, b=ub, args=((1-a2), xi, phi1, phi2))
                lower[i-1] = brentq(myfun, a=lb, b=0, args=(a2, xi, phi1, phi2))

                upper[i-1] = upper[i-1] * phi1 ** 2 / Qprod1
                lower[i-1] = lower[i-1] * phi1 ** 2 / Qprod1

        bp = bp[nbp+2:]
        bp = np.column_stack((bp - np.ceil(upper), bp, bp - np.floor(lower)))
        return bp


def pargmaxV(x, xi=1, phi1=1, phi2=1):
    phi = xi * (phi2 / phi1)**2

    def G1(x):
        x = np.abs(x)
        frac = xi / phi
        term1 = -np.exp(np.log(x) / 2 - x / 8 - np.log(2 * np.pi) / 2)

        term2a = phi / xi * (phi + 2 * xi) / (phi + xi)
        term2b = np.exp((frac * (1 + frac) * x / 2) + norm.logcdf(-(0.5 + frac) * np.sqrt(x)))
        term2 = term2a * term2b

        term3a = np.log(x / 2 - 2 + ((phi + 2 * xi)**2) / ((phi + xi) * xi))
        term3b = norm.logcdf(-np.sqrt(x)/2)
        term3 = np.exp(term3a + term3b)

        rval = term1 - term2 + term3
        return rval

    def G2(x):
        x = np.abs(x)
        frac = xi**2 / phi
        term1 = 1 + np.sqrt(frac) * np.exp(np.log(x)/2 - (frac * x) / 8  - np.log(2 * np.pi) / 2)

        term2a = xi/phi * (2*phi + xi)/(phi + xi)
        term2b = np.exp(((phi + xi) * x/2) + norm.logcdf(-(phi + xi/2)/np.sqrt(phi) * np.sqrt(x)))
        term2 = term2a * term2b

        term3a = np.log(((2 * phi + xi)**2) / ((phi + xi) * phi) - 2 + frac * x / 2)
        term3b = norm.logcdf(-np.sqrt(frac) * np.sqrt(x) / 2)
        term3 = np.exp(term3a + term3b)

        rval = term1 + term2 - term3
        return rval

    if x < 0:
        return G1(x)
    else:
        return G2(x)


if __name__ == "__main__":
    x = -5.0
    print(pargmaxV(x))
