import numpy as np


def _no_nans(arr):
    return not np.isnan(arr).any()


def _Xinv0(obj):
    qr = obj.qr
    k = obj.coefficients.shape[0]
    rval = np.zeros((k, k))
    wi = qr.pivot[1:qr.rank]
    rval[wi, wi] = chol2inv(qr.qr[1:qr.rank, 1:qr.rank, drop=False])
    return rval


def recresid(x, y, start=None, end=None, tol=None):
    """
    Function for computing the recursive residuals (standardized one step prediction errors)
    of a linear regression model.
    """
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
    rval = np.repeat(0, n - q)

    ## initialize recursion
    y1 = y[:q]
    fm = lm.fit(x[:q], y1)
    X1 = _Xinv0(fm)
    betar = np.nan_to_num(fm.coefficients)

    xr = x[q]
    fr = 1 + (xr @ X1 @ xr)
    rval[0] = (y[q] - xr @ betar) / np.sqrt(fr)

    ## check recursion against full QR decomposition?
    check = True

    if (q + 1) < n:
        for r in range(q + 1, n):
            ## check for NAs in coefficients
            nona = _no_nans(fm.coefficients)

            ## recursion formula
            X1 = X1 - (X1 @ np.outer(xr, xr) @ X1)/fr
            betar += X1 @ xr * rval[r-q-2] * sqrt(fr)

            ## full QR decomposition
            if check:
                y1 = y[:(r-1)]
                fm = lm.fit(x[:(r-1)], y1)
                nona = nona and _no_nans(betar) and _no_nans(fm.coefficients)

                ## keep checking?
                check = not (nona and np.allclose(fm.coefficients, betar, atol=tol))
                X1 = _Xinv0(fm)
                betar = np.nan_to_num(fm.coefficients)

        ## residual
        xr = x[r,]
        fr = 1 + xr @ X1 @ xr
        val = np.nan_to_num(xr * betar)
        rval[r-q] = (y[r] - np.sum(val)) / np.sqrt(fr)

    return rval


if __name__ == "__main__":
    recresid()
