breakpoints.breakpointsfull <- function(obj, breaks = NULL, ...)
{
  cat("breakpoints.breakpointsfull called for breaks =", breaks, "\n")
  if(is.null(breaks))
  {
    sbp <- summary(obj)
    print(sbp$RSS)
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
  cat("summary.breakpointsfull called", "\n")
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
  bp <- bp$breakpoints
  if(breaks > 1) {
    for(m in (breaks-1):1)
    {
      bp <- rbind(NA, bp)
      bpm <- breakpoints(object, breaks = m)
      if(sort) {
        pos <- apply(outer(bpm$breakpoints, bp[nrow(bp),],
                           FUN = function(x,y) abs(x - y)), 1, which.min)
        if(length(pos) > unique(length(pos))) {
          warning("sorting not possible", call. = FALSE)
          sort <- FALSE
        }
      }
      if(!sort) pos <- 1:m
      bp[1,pos] <- bpm$breakpoints
      RSS[m+1] <- bpm$RSS
      BIC[m+1] <- AIC(bpm, k = log(n))
    }
  } else {
    bp <- as.matrix(bp)
  }
  rownames(bp) <- as.character(1:breaks)
  colnames(bp) <- rep("", breaks)
  RSS <- rbind(RSS, BIC)
  rownames(RSS) <- c("RSS", "BIC")
  RVAL <- list(breakpoints = bp,
               RSS = RSS,
               call = object$call)
  class(RVAL) <- "summary.breakpointsfull"
  cat("summary.breakpointsfull is finished\n")
  return(RVAL)
}


breakpoints.formula <- function(formula, h = 0.15, breaks = NULL,
                                data = list(), hpc = c("none", "foreach"), ...)
{
  print("breakpoints.formula called")
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



data("Nile")
bp.nile <- breakpoints(Nile ~ 1)
## print(bp.nile)
## summary(bp.nile)

## ## the BIC also chooses one breakpoint
## plot(bp.nile)
## breakpoints(bp.nile)
