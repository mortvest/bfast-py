import numpy as np


def _no_nans(arr):
    return not np.isnan(arr).any()


def _Xinv0(x):
    """
    Calculate (X'X)^-1 using QR decomposition
    """
    ncol = np.shape(x)[1]
    r = np.linalg.qr(x)[1]
    qr_rank = np.linalg.matrix_rank(r)

    r = r[:qr_rank, :qr_rank]

    rval = np.zeros((ncol, ncol))
    rval[:qr_rank, :qr_rank] = np.linalg.inv(r.T @ r)
    return rval


def recresid(x, y, start=None, end=None, tol=None, deg=1):
    """
    Function for computing the recursive residuals (standardized one step
    prediction errors) of a linear regression model.
    """
    if np.ndim(x) == 1:
        ncol = 1
        nrow = x.shape
    else:
        nrow, ncol = x.shape

    if start is None:
        start = ncol + 1
    if end is None:
        end = nrow
    if tol is None:
        tol = np.sqrt(np.finfo(float).eps / ncol)

    # checks and data dimensions
    assert start > ncol and start <= nrow
    assert end >= start and end <= nrow

    n = end
    q = start - 1
    k = ncol
    rval = np.zeros(n - q)

    # print("q =", q)
    # initialize recursion
    y1 = y[:q]

    x_q = x[:q]
    # fm = lm.fit(x_q, y1)
    coeffs = np.polyfit(x_q.flatten(), y1, deg=deg)
    # print("coeffs = ", coeffs)

    X1 = _Xinv0(x_q)
    # betar = np.nan_to_num(fm.coefficients)
    betar = np.nan_to_num(coeffs)

    xr = x[q]
    fr = 1 + (xr @ X1 @ xr)
    # print("xr =", xr)
    # print("betar =", betar)

    rval[0] = (y[q] - xr @ betar) / np.sqrt(fr)

    # check recursion against full QR decomposition?
    check = True

    if (q + 1) < n:
        for r in range(q + 1, n):
            # check for NAs in coefficients
            nona = _no_nans(coeffs)

            # recursion formula
            X1 = X1 - (X1 @ np.outer(xr, xr) @ X1)/fr
            # print("X1", X1)
            betar += X1 @ xr * rval[r-q-1] * np.sqrt(fr)
            # print("betar", betar)

            # full QR decomposition
            if check:
                # print("r", r)
                y1 = y[:(r-1)]
                # print("check_y1", y1)
                x_i = x[:(r-1)]
                # print("check_x_i", x_i)
                # fm = lm.fit(x_i, y1)
                coeffs = np.polyfit(x_i.flatten(), y1, deg=deg)
                # print("check_coeffs", coeffs)
                nona = nona and _no_nans(betar) and _no_nans(coeffs)
                # print("check_nona", nona)

                # keep checking?
                check = not (nona and np.allclose(coeffs, betar, atol=tol))
                X1 = _Xinv0(x_i)
                # print("check_X1", X1)
                betar = np.nan_to_num(coeffs)

            # residual
            xr = x[r]
            # print("xr", xr)
            fr = 1 + xr @ X1 @ xr
            # print("fr:", fr)
            val = np.nan_to_num(xr * betar)
            rval[r-q] = (y[r] - np.sum(val)) / np.sqrt(fr)
            # print("rval", rval)

    return rval


if __name__ == "__main__":
    x = np.arange(1, 21)
    y = 2 * x
    y[10:] += 10
    x = x.reshape((20,1))
    rec_res = recresid(x, y, deg=0)
    print(rec_res)
