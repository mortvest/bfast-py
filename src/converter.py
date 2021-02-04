import numpy as np
# working R installation is required
import rpy2.robjects as robjects


def rda_to_npy(file_name, save=False):
    """
    Convert an .rda object from the R programming language
    to a .npy object for use with numpy
    """
    robjects.r["load"](file_name + ".rda")
    matrix = robjects.r[file_name]
    a = np.array(matrix)
    if save:
        np.save(file_name, a)
    return a


if __name__ == "__main__":
    # rda_to_npy("ndvi", save=True)
    # rda_to_npy("simts", save=True)
    # rda_to_npy("harvest", save=True)
    rda_to_npy("som", save=True)

    mat_loaded = np.load("som.npy")
    print(mat_loaded)
