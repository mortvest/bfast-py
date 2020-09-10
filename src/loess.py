import numpy as np
import matplotlib.pyplot as plt


def find_weights(x, i, q, n):
    dists = np.abs(x - x[i])
    dists_sorted = np.sort(dists)
    qth_dist = dists_sorted[q-1] if q < n else int(dists_sorted[-1] * (q / n))
    # qth_dist = dists_sorted[-q] if q < n else int(dists_sorted[1] * (q / n))
    u = dists / qth_dist
    v = (1 - u**3)**3
    v[u >= 1] = 0
    return v


def loess(x, y, q, d, rhos=None):
    if (x.shape[0] != y.shape[0]):
        raise ValueError("dimensions of x, y must match")
    if rhos is not None and (x.shape[0] != rhos.shape[0]):
        raise ValueError("dimensions of x and rhos must match")
    n = x.shape[0]
    y_f = np.zeros(n)
    for i in range(n):
        v = find_weights(x, i, q, n)
        if rhos is not None:
            v *= rhos
        z = np.polyfit(x, y, d, w=v)
        p = np.poly1d(z)
        y_f[i] = p(x[i])
    return y_f


if __name__ == "__main__":
    img_dir = "../report/imgs/"
    def plot(d):
        fig = plt.figure(figsize=(15,13))
        plt.rc('font', size=16)

        for i, q in enumerate(qs):
            ax = fig.add_subplot(len(qs), 1, i+1)
            y_f = loess(x, y, q, d)
            plt.plot(x, y, label=r"$f(x)$")
            # plt.plot(x, y_f, label=r"LOESS with $q={}$ and $d={}$".format(q, d), color="red")
            plt.plot(x, y_f, label="LOESS", color="red")
            plt.title(r"$q={}$".format(q))
            plt.legend()
        plt.subplots_adjust(hspace=0.45)

        # plt.savefig(img_dir + "loess_t_{}.png".format(d), bbox_inches ="tight")
        plt.savefig(img_dir + "loess{}.png".format(d), bbox_inches ="tight")
        # plt.show()
        plt.clf()

    n_samples = 100
    min_x = 0
    max_x = 20
    x = np.linspace(min_x, max_x, n_samples)

    noise = np.random.normal(0, 0.25, n_samples)
    y_scale = 2
    y = y_scale * np.sin(x) + noise

    # qs = [3, 5, 10, 50]
    qs = [5, 50, 100, 1000]
    # for d in [1]:
    for d in [1,2]:
        plot(d)
