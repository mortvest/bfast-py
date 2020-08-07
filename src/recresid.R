recresid <- function(x, ...)
{
  UseMethod("recresid")
}

recresid.formula <- function(formula, data = list(), ...)
{
    mf <- model.frame(formula, data = data)
    y <- model.response(mf)
    modelterms <- terms(formula, data = data)
    X <- model.matrix(modelterms, data = data)
    rr <- recresid(X, y, ...)
    return(rr)
}

recresid.lm <- function(x, data = list(), ...)
{
    X <- if(is.matrix(x$x)) x$x else model.matrix(terms(x), model.frame(x))
    y <- if(is.vector(x$y)) x$y else model.response(model.frame(x))
    rr <- recresid(X, y, ...)
    return(rr)
}


## convenience function to replace NAs with 0s in coefs
.coef0 <- function(obj) {
  cf <- obj$coefficients
  ifelse(is.na(cf), 0, cf)
}
.Xinv0 <- function(obj) {
  qr <- obj$qr
  k = length(obj$coefficients)
  rval <- matrix(0, ncol = k, nrow = k)
  wi <- qr$pivot[1:qr$rank]
  rval[wi,wi] <- chol2inv(qr$qr[1:qr$rank, 1:qr$rank, drop = FALSE])
  rval
}


recresid.default <- function(x, y, start = ncol(x) + 1, end = nrow(x),
  tol = sqrt(.Machine$double.eps)/ncol(x), ...)
{
  ## checks and data dimensions
  stopifnot(start > ncol(x) & start <= nrow(x))
  stopifnot(end >= start & end <= nrow(x))
  if (getOption("strucchange.use_armadillo", FALSE))
    return(.sc_cpp_recresid(x,y,start,end,tol, getOption("strucchange.armadillo_rcond_min",sqrt(.Machine$double.eps))))

  n <- end
  q <- start - 1
  k <- ncol(x)
  rval <- rep(0, n - q)


  ## initialize recursion
  y1 <- y[1:q]
  fm <- lm.fit(x[1:q, , drop = FALSE], y1)
  ## cat(fm$coefficients)
  X1 <- .Xinv0(fm)
  betar <- .coef0(fm)
  xr <- as.vector(x[q+1,])

  fr <- as.vector((1 + (t(xr) %*% X1 %*% xr)))
  rval[1] <- (y[q+1] - t(xr) %*% betar)/sqrt(fr)

  ## check recursion agains full QR decomposition?
  check <- TRUE

  if((q+1) < n)
  {
    for(r in ((q+2):n))
    {
      ## check for NAs in coefficients
      nona <- all(!is.na(fm$coefficients))

      ## recursion formula
      X1 <- X1 - (X1 %*% outer(xr, xr) %*% X1)/fr
      ## cat("X1", c(X1), "\n")
      betar <- betar + X1 %*% xr * rval[r-q-1] * sqrt(fr)
      ## cat("betar", c(betar), "\n")

      ## full QR decomposition
      if(check) {
        ## cat("r", c(r), "\n")
        y1 <- y[1:(r-1)]
        ## cat("check_y1", c(y1), "\n")
        fm <- lm.fit(x[1:(r-1), , drop = FALSE], y1)
        ## cat("check_x1", c(x[1:(r-1)]), "\n")
        nona <- nona & all(!is.na(betar)) & all(!is.na(fm$coefficients))
        print(nona)
        ## cat("check_nona", nona, "\n")
        ## keep checking?
        if(nona && isTRUE(all.equal(as.vector(fm$coefficients), as.vector(betar), tol = tol))) {
          check <- FALSE
        }
        X1 <- .Xinv0(fm)
        ## cat("check_X1", X1, "\n")
        betar <- .coef0(fm)
      }

      ## residual
      xr <- as.vector(x[r,])
      ## cat("xr", xr, "\n")
      fr <- as.vector((1 + (t(xr) %*% X1 %*% xr)))
      ## cat("fr", fr, "\n")
      rval[r-q] <- (y[r] - sum(xr * betar, na.rm = TRUE))/sqrt(fr)
      ## cat("rval", rval, "\n")
      ## quit()
    }
  }
  ## quit()
  return(rval)
}

x <- matrix(1:20, nrow=20, ncol=1)
y <- x * 2
y[11:20] <- y[11:20] + 10
print(recresid(x, y))
