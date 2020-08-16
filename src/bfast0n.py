import logging

import numpy as np

import datasets
from breakpoints import Breakpoints
from setup import logging_setup
from utils import omit_nans


logger = logging.getLogger(__name__)


def bfast0n(X, y, frequency, stl="none", period=None, order=3, use_mp=True):
    """
    Light-weight detection of multiple breaks in a time series
    """
    def stl_adjust(x):
        seasonal, trend, _ = stl(x, periodic=True).time_series
        if stl == "trend":
            return x - trend
        elif stl == "seasonal":
            return x - seasonal
        elif stl == "both":
            return x - trend - seasonal
        else:
            raise ValueError("Unknown STL type:", stl)

    if stl != "none":
        logger.info("Applying STL")
        if period is None:
            raise ValueError("Provide a period")
        if X.ndim > 1:
            for i in range(X.shape[1]):
                X[:, i] = stl_adjust(X[:, i])

        else:
            X = stl_adjust(X)
    else:
        logger.info("STL ommited")


    nrow, ncol = np.shape(X)

    ## set up harmonic trend matrix as well
    order = min(frequency, order)

    logger.info("Calculating harmon matrix")
    harmon = np.outer(2 * np.pi * X, np.arange(1, order + 1))
    harmon = np.column_stack((np.cos(harmon), np.sin(harmon)))

    if 2 * order == freq:
        harmon = np.delete(harmon, 2 * order - 1, axis=1)

    trend = np.arange(1, y.shape[0] + 1)
    intercept = np.ones(y.shape[0])
    X = np.column_stack((intercept, trend, harmon))

    logger.info("Removing nans")
    X, y = omit_nans(X, y)

    logger.info("Esimating breakpoints")
    bp = Breakpoints(X, y, use_mp=use_mp).breakpoints
    return bp


if __name__ == "__main__":
    logging_setup()
    y = datasets.ndvi
    x = datasets.ndvi_dates
    freq = datasets.ndvi_freqency

    print("Running bfast0n on NDVI")
    v = bfast0n(x, y, freq, "none")
    print(v)
