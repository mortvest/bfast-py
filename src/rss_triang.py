import logging
import multiprocessing as mp
from functools import partial

import numpy as np

from recresid import recresid


logger = logging.getLogger(__name__)


def rss_triang(n, h, X, y, k, intercept_only, use_mp=True):
    """
    Calculates the upper triangular matrix of squared residuals
    """
    fun = rss_triang_par if use_mp else rss_triang_seq
    return fun(n, h, X, y, k, intercept_only)


def RSSi(i, n, h, X, y, k, intercept_only):
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


def rss_triang_seq(n, h, X, y, k, intercept_only):
    """
    sequential version
    """
    logger.info("sequential version of rss_triang chosen")
    my_RSSi = partial(RSSi, n=n, h=h, X=X, y=y, k=k, intercept_only=intercept_only)
    return np.array([my_RSSi(i) for i in np.arange(n-h+1).astype(int)], dtype=object)


def rss_triang_par(n, h, X, y, k, intercept_only):
    """
    parallel version
    """
    logger.info("mp-enabled version of rss_triang chosen")
    my_RSSi = partial(RSSi, n=n, h=h, X=X, y=y, k=k, intercept_only=intercept_only)
    pool = mp.Pool(mp.cpu_count())
    indexes = np.arange(n - h + 1).astype(int)
    rval = pool.map(my_RSSi, indexes)
    rval = np.array(rval, dtype=object)
    return rval
