import numpy as np
# import matplotlib.pyplot as plt
import rpy2.robjects as robjects


def rda_to_npy(file_name, save=False):
    robjects.r["load"](file_name + ".rda")
    matrix = robjects.r[file_name]
    a = np.array(matrix)
    if save:
        np.save(file_name, a)
    return a


if __name__ == "__main__":
    rda_to_npy("ndvi", save=True)
    rda_to_npy("simts", save=True)


    # file_name = "USIncExp"
    # mat = rda_to_npy(file_name, save=True)
    # print(mat.shape)

    # mat_loaded = np.load(file_name + ".npy")
    # print(mat_loaded.shape)

    # for i in range(mat.shape[0]):
    #     print("[{}, {}]".format(mat[i,0], mat[i,1]))

    # income = a[:,0]
    # expanditure = a[:,1]
    # xs = np.arange(income.shape[0])
    # plt.plot(xs, income, label="income")
    # plt.plot(xs, expanditure, label="expanditure")
    # plt.legend()
    # plt.show()

