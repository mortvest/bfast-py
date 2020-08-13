import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.datasets import co2


def seasonal_average(x, period):
    n = x.shape[0]

    # if n % period != 0:
    #     raise ValueError("Sesonal is not complete")

    # n_periods = int(n_samples / period)
    # mat = seasonal

    use_pad = int(n / period) * period != n
    if use_pad:
        periodic = np.zeros(period)
        n_periods = int(np.ceil(n_samples / period))
        full_len = period * n_periods
        zeros = np.zeros(full_len - n_samples)
        mat = np.concatenate((x, zeros))
    else:
        n_periods = int(np.ceil(n_samples / period))
        mat = seasonal

    periodic = np.mean(mat.reshape(n_periods, period), axis=0)
    retval = np.tile(periodic, n_periods)
    return retval


def stl(y, period, periodic=True):
    if period:
        stl = STL(y, period=period)
    else:
        stl = STL(y)
    res = stl.fit()
    seasonal = res.seasonal
    trend = res.trend
    residual = res.resid
    if periodic:
        seasonal = seasonal_average(seasonal, period)
    return seasonal, trend, residual


if __name__ == "__main__":
    n_samples = 49 * 7

    min_x = 0
    max_x = 40
    x = np.linspace(min_x, max_x, n_samples)

    noise = np.random.normal(0, 0.25, n_samples)
    y_scale = 3

    axis = x**0.75
    y = axis + y_scale * np.sin(x) + noise
    period = int(n_samples / (max_x / (2*np.pi)))


    seasonal, trend, residual = stl(y, period, periodic=True)

    plt.plot(np.arange(seasonal.shape[0]), seasonal)
    # plt.plot(np.arange(n_samples), seasonal)
    plt.show()

    # res = STL(y, period=period).fit()
    # fig = res.plot()
    # plt.savefig("../report/imgs/stl1.png")
    # plt.show()
