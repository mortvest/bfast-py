import logging

import numpy as np

import datasets
from breakpoints import Breakpoints
from setup import logging_setup


logger = logging.getLogger(__name__)

def omit_nans(x, y):
    x_index = ~np.isnan(x).any(axis=1)
    if y is None:
        return x[index]
    else:
        x = x[x_index]
        y = y[x_index]
        y_index = ~np.isnan(y)
        return x[y_index], y[y_index]


def bfast0n(X, y, frequency, stl="none", period=None, order=3):
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
        logging.info("Applying STL")
        if period is None:
            raise ValueError("Provide a period")
        if X.ndim > 1:
            for i in range(X.shape[1]):
                X[:, i] = stl_adjust(X[:, i])

        else:
            X = stl_adjust(X)
    else:
        logging.info("STL ommited")


    nrow, ncol = np.shape(X)

    ## set up harmonic trend matrix as well
    order = min(frequency, order)

    logging.info("Calculating harmon matrix")
    harmon = np.outer(2 * np.pi * X, np.arange(1, order + 1))
    harmon = np.column_stack((np.cos(harmon), np.sin(harmon)))

    if 2 * order == freq:
        harmon = np.delete(harmon, 2 * order - 1, axis=1)

    trend = np.arange(1, y.shape[0] + 1)
    intercept = np.ones(y.shape[0])
    X = np.column_stack((intercept, trend, harmon))

    logging.info("Removing nans")
    X, y = omit_nans(X, y)

    logging.info("Esimating breakpoints")
    bp = Breakpoints(X, y).breakpoints
    logging.info("Breakpoints finished")
    return bp


if __name__ == "__main__":
    logging_setup()
    y = datasets.ndvi
    x = datasets.ndvi_dates
    freq = datasets.ndvi_freqency

    v = bfast0n(x, y, freq, "none")
    print(v)
