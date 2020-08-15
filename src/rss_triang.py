import numpy as np

from recresid import recresid

def rss_triang(n, h, X, y, k, intercept_only):
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

    return np.array([RSSi(i) for i in np.arange(n-h+1).astype(int)], dtype=object)
