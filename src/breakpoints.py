import numpy as np
from scipy.stats import norm


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
