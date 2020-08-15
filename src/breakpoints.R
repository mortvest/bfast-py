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
      betar <- betar + X1 %*% xr * rval[r-q-1] * sqrt(fr)

      ## full QR decomposition
      if(check) {
        y1 <- y[1:(r-1)]
        fm <- lm.fit(x[1:(r-1), , drop = FALSE], y1)
        nona <- nona & all(!is.na(betar)) & all(!is.na(fm$coefficients))
        ## print(nona)
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
      fr <- as.vector((1 + (t(xr) %*% X1 %*% xr)))

      v <- (y[r] - sum(xr * betar, na.rm = TRUE))/sqrt(fr)
      rval[r-q] <- v

    }
  }
  ## quit()
  return(rval)
}

breakpoints.breakpointsfull <- function(obj, breaks = NULL, ...)
{
  if(is.null(breaks))
  {
    sbp <- summary(obj)
    ## print(sbp$RSS)
    breaks <- which.min(sbp$RSS["BIC",]) - 1
  }
  if(breaks < 1)
  {
    breakpoints <- NA
    RSS <- obj$RSS(1, obj$nobs)
  } else {
    RSS.tab <- obj$extend.RSS.table(obj$RSS.table, breaks)
    breakpoints <- obj$extract.breaks(RSS.tab, breaks)
    bp <- c(0, breakpoints, obj$nobs)
    RSS <- sum(apply(cbind(bp[-length(bp)]+1,bp[-1]), 1,
                     function(x) obj$RSS(x[1], x[2])))
  }
  RVAL <- list(breakpoints = breakpoints,
               RSS = RSS,
               nobs = obj$nobs,
               nreg = obj$nreg,
               call = match.call(),
               datatsp = obj$datatsp)
  class(RVAL) <- "breakpoints"
  return(RVAL)
}

summary.breakpointsfull <- function(object, breaks = NULL, sort = TRUE, format.times = NULL, ...)
{
  if(is.null(format.times)) format.times <- ((object$datatsp[3] > 1) & (object$datatsp[3] < object$nobs))
  if(is.null(breaks)) breaks <- ncol(object$RSS.table)/2
  n <- object$nobs
  RSS <- c(object$RSS(1, n), rep(NA, breaks))
  BIC <- c(n * (log(RSS[1]) + 1 - log(n) + log(2*pi)) + log(n) * (object$nreg + 1),
           rep(NA, breaks))
  names(RSS) <- as.character(0:breaks)
  bp <- breakpoints(object, breaks = breaks)
  RSS[breaks + 1] <- bp$RSS
  BIC[breaks + 1] <- AIC(bp, k = log(n))

  if(breaks > 1) {
    for(m in (breaks-1):1)
    {
      bpm <- breakpoints(object, breaks = m)
      RSS[m+1] <- bpm$RSS
      BIC[m+1] <- AIC(bpm, k = log(n))
    }
  }
  RSS <- rbind(RSS, BIC)
  rownames(RSS) <- c("RSS", "BIC")
  RVAL <- list(RSS = RSS)
  class(RVAL) <- "summary.breakpointsfull"
  return(RVAL)
}


breakpoints.formula <- function(formula, h = 0.15, breaks = NULL,
                                data = list(), hpc = c("none", "foreach"), ...)
{
  mf <- model.frame(formula, data = data)
  y <- model.response(mf)
  modelterms <- terms(formula, data = data)
  X <- model.matrix(modelterms, data = data)

  n <- nrow(X)
  k <- ncol(X)
  intercept_only <- isTRUE(all.equal(as.vector(X), rep(1L, n)))
  if(is.null(h)) h <- k + 1
  if(h < 1) h <- floor(n*h)
  if(h <= k)
    stop("minimum segment size must be greater than the number of regressors")
  if(h > floor(n/2))
    stop("minimum segment size must be smaller than half the number of observations")
  if(is.null(breaks)) {
    breaks <- ceiling(n/h) - 2
  } else {
    if(breaks > ceiling(n/h) - 2) {
      breaks0 <- breaks
      breaks <- ceiling(n/h) - 2
      warning(sprintf("requested number of breaks = %i too large, changed to %i", breaks0, breaks))
    }
  }

  hpc <- match.arg(hpc)
  if(hpc == "foreach" && !requireNamespace("foreach")) {
    warning("High perfomance computing (hpc) support with 'foreach' package is not available, foreach is not installed.")
    hpc <- "none"
  }

  ## compute ith row of the RSS diagonal matrix, i.e,
  ## the recursive residuals for segments starting at i = 1:(n-h+1)


  if (getOption("strucchange.use_armadillo", FALSE)) {
    res = .sc_cpp_construct_rss_table(y,X,n,h,breaks,intercept_only,sqrt(.Machine$double.eps)/ncol(X), getOption("strucchange.armadillo_rcond_min",sqrt(.Machine$double.eps)))
    RSS.table = res$RSS.table
    dimnames(RSS.table) = list(as.character(h:(n-h)),
                               as.vector(rbind(paste("break", 1:breaks, sep = ""),paste("RSS", 1:breaks, sep = ""))))
    RSS.triang = res$RSS.triang
    RSS <- function(i, j) .sc_cpp_rss(RSS.triang, i, j)
    extend.RSS.table <- function(RSS.table, breaks) {
      if (2*breaks > ncol(RSS.table)) {
        RSS.table = .sc_cpp_extend_rss_table(rss_table = RSS.table, rss_triang = RSS.triang, n = n, h=h, breaks = breaks)
        dimnames(RSS.table) = list(as.character(h:(n-h)), as.vector(rbind(paste("break", 1:breaks, sep = ""),paste("RSS", 1:breaks, sep = ""))))
      }
      RSS.table
    }
  }
  else {

    RSSi <- function(i)
    {
      ssr <- if(intercept_only) {
        (y[i:n] - cumsum(y[i:n])/(1L:(n-i+1L)))[-1L] * sqrt(1L + 1L/(1L:(n-i)))
      } else {
        recresid(X[i:n,,drop = FALSE],y[i:n])
      }
      c(rep(NA, k), cumsum(ssr^2))
    }

    ## employ HPC support if available/selected
    RSS.triang <- if(hpc == "none") sapply(1:(n-h+1), RSSi) else foreach::foreach(i = 1:(n-h+1)) %dopar% RSSi(i)

    ## function to extract the RSS(i,j) from RSS.triang
    RSS <- function(i,j) RSS.triang[[i]][j - i + 1]

    ## compute optimal previous partner if observation i is the mth break
    ## store results together with RSSs in RSS.table

    ## breaks = 1

    index <- h:(n-h)
    break.RSS <- sapply(index, function(i) RSS(1,i))

    RSS.table <- cbind(index, break.RSS)
    rownames(RSS.table) <- as.character(index)

    ## breaks >= 2

    extend.RSS.table <- function(RSS.table, breaks)
    {
      ## print(RSS.table)
      ## cat("range", (ncol(RSS.table)/2 + 1):breaks, "\n")
      if((breaks*2) > ncol(RSS.table)) {
        for(m in (ncol(RSS.table)/2 + 1):breaks)
        {
          my.index <- (m*h):(n-h)
          my.RSS.table <- RSS.table[,c((m-1)*2 - 1, (m-1)*2)]
          my.RSS.table <- cbind(my.RSS.table, NA, NA)
          for(i in my.index)
          {
            pot.index <- ((m-1)*h):(i - h)
            break.RSS <- sapply(pot.index, function(j) my.RSS.table[as.character(j), 2] + RSS(j+1,i))
            opt <- which.min(break.RSS)
            my.RSS.table[as.character(i), 3:4] <- c(pot.index[opt], break.RSS[opt])
          }
          RSS.table <- cbind(RSS.table, my.RSS.table[,3:4])
        }
        colnames(RSS.table) <- as.vector(rbind(paste("break", 1:breaks, sep = ""),
                                               paste("RSS", 1:breaks, sep = "")))
      }
      return(RSS.table)
    }

    RSS.table <- extend.RSS.table(RSS.table, breaks)
    ## print(RSS.table)
    ## quit()

  }

  ## extract optimal breaks

  extract.breaks <- function(RSS.table, breaks)
  {
    if((breaks*2) > ncol(RSS.table)) stop("compute RSS.table with enough breaks before")
    index <- RSS.table[, 1, drop = TRUE]
    break.RSS <- sapply(index, function(i) RSS.table[as.character(i),breaks*2] + RSS(i + 1, n))
    opt <- index[which.min(break.RSS)]
    if(breaks > 1) {
      for(i in ((breaks:2)*2 - 1))
        opt <- c(RSS.table[as.character(opt[1]),i], opt)
    }
    names(opt) <- NULL
    return(opt)
  }

  ## opt <- extract.breaks(RSS.table, breaks)
  ## print(opt)

  if(is.ts(data)) {
    if(NROW(data) == n) datatsp <- tsp(data)
    else datatsp <- c(1/n, 1, n)
  } else {
    env <- environment(formula)
    if(missing(data)) data <- env
    orig.y <- eval(attr(terms(formula), "variables")[[2]], data, env)
    if(is.ts(orig.y) & (NROW(orig.y) == n)) datatsp <- tsp(orig.y)
    else datatsp <- c(1/n, 1, n)
  }

  RVAL <- list(RSS.table = RSS.table,
               RSS.triang = RSS.triang,
               RSS = RSS,
               extract.breaks = extract.breaks,
               extend.RSS.table = extend.RSS.table,
               nobs = n,
               nreg = k, y = y, X = X,
               call = match.call(),
               datatsp = datatsp)
  class(RVAL) <- c("breakpointsfull", "breakpoints")

  ## RVAL$breakpoints <- breakpoints(RVAL)$breakpoints
  RVAL$breakpoints <- breakpoints.breakpointsfull(RVAL)$breakpoints
  ## x <- breakpoints(RVAL)

  return(RVAL)
}


logLik.breakpoints <- function(object, ...)
{
  n <- object$nobs
  df <- (object$nreg + 1) * (length(object$breakpoints[!is.na(object$breakpoints)]) + 1)
  logL <- -0.5 * n * (log(object$RSS) + 1 - log(n) + log(2 * pi))
  attr(logL, "df") <- df
  class(logL) <- "logLik"
  return(logL)
}

breakpoints <- function(obj, ...)
{
  UseMethod("breakpoints")
}



## data("Nile")
## bp.nile <- breakpoints(Nile ~ 1)
## print(bp.nile$breakpoints)

n <- 50
x <- matrix(1:n, nrow=n, ncol=1)
y <- x
y[15:nrow(y)] <- y[15:nrow(y)] * 0.03
y[35:nrow(y)] <- y[35:nrow(y)] + 10

form <- y ~ 1
## form <- y ~ x
bp <- breakpoints(form)
print(bp$breakpoints)




## ## the BIC also chooses one breakpoint
## plot(bp.nile)
## breakpoints(bp.nile)

## x <- matrix(1:50, nrow=50, ncol=1)
## y <- x * 2
## y[15:nrow(y)] <- y[15:nrow(y)] + 30
## ## y[35:nrow(y)] <- y[35:nrow(y)] + 25
## ## print(y)
## form <- y ~ 1
## bp <- breakpoints(form)
## print(bp$breakpoints)
## ## summary(bp)
