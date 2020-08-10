from breakpoints import Breakpoints


# def _bfast_pp(X, y, stl="none", order=3, sbins=1):
#     """
#     STL-based pre-processing to try to adjust for trend or season
#     """
#     def stl_adjust(x):
#         x_stl = stl(x, s_window="periodic").time_series
#         if stl == "trend":
#             return x - x_stl.trend
#         elif stl == "seasonal":
#             return x - x_stl.seasonal
#         elif stl == "both":
#             return x - x_stl.trend - x_stl.seasonal
#         else:
#             raise ValueError("Unknown STL type:", stl)

#     if stl != "none":
#         # if data.shape[1] > 1:
#         if X.ndim > 1:
#             for i in range(X.shape[1]):
#                 X[:, i] = stl_adjust(X[:, i])
#         else:
#             X = stl_adjust(X)

#     nrow, ncol = np.shape(X)
#     ## check for covariates
#     if data.shape[1] > 1:
#         x = coredata(data)[, -1L]
#         y = data[, 1L]
#     else:
#         x = None
#         y = data

#     ## data with trend and season factor
#     rval <- data.frame(
#         time = as.numeric(time(y)),
#         response = y,
#         trend = 1:y.shape[0]
#         season = cut(cycle(y), if (sbins > 1) sbins else frequency(y)*sbins, ordered_result = TRUE)
#     )

#     ## set up harmonic trend matrix as well
#     freq = frequency(y)
#     order = min(freq, order)
#     harmon = outer(2 * pi * as.vector(time(y)), 1:order)
#     harmon = cbind(apply(harmon, 2, cos), apply(harmon, 2, sin))
#     colnames(harmon) <- if(order == 1) {
#         c("cos", "sin")
#     } else {
#         c(paste("cos", 1:order, sep = ""), paste("sin", 1:order, sep = ""))
#     }
#     if((2 * order) == freq) harmon <- harmon[, -(2 * order)]
#     rval$harmon <- harmon


#     ## omit missing values
#     rval <- na.action(rval)

#     ## return everything
#     return rval


def bfast0n(X, y, stl="none"):
    # formula = response ~ trend + harmon
    def stl_adjust(x):
        x_stl = stl(x, s_window="periodic").time_series
        if stl == "trend":
            return x - x_stl.trend
        elif stl == "seasonal":
            return x - x_stl.seasonal
        elif stl == "both":
            return x - x_stl.trend - x_stl.seasonal
        else:
            raise ValueError("Unknown STL type:", stl)

    if stl != "none":
        # if data.shape[1] > 1:
        if X.ndim > 1:
            for i in range(X.shape[1]):
                X[:, i] = stl_adjust(X[:, i])
        else:
            X = stl_adjust(X)

    bp = Breakpoints(X, y)
    return bp
