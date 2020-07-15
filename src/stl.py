import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.datasets import co2


if __name__ == "__main__":
    n_samples = 1000

    min_x = 0
    max_x = 40
    x = np.linspace(min_x, max_x, n_samples)

    noise = np.random.normal(0, 0.25, n_samples)
    y_scale = 3

    axis = x**0.75
    y = axis + y_scale * np.sin(x) + noise

    plt.scatter(x, noise)
    plt.show()

    # data = co2.load(True).data
    # data = data.resample('M').mean().ffill()
    # print(data)

    period = int(n_samples / (max_x / (2*np.pi)))

    # stl = STL(y, period=period)
    # res = stl.fit()
    # fig = res.plot()
    # plt.savefig("../report/imgs/stl1.png")
    # plt.show()
