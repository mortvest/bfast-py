breakpoints.formula <- function(formula, h = 0.15, breaks = NULL, data = list(), ...)
{
  mf <- model.frame(formula, data = data)
  y <- model.response(mf)
  modelterms <- terms(formula, data = data)
  X <- model.matrix(modelterms, data = data)

  n <- nrow(X)
  k <- ncol(X)

  intercept_only <- isTRUE(all.equal(as.vector(X), rep(1L, n)))

  h <- floor(n*h)

  if(is.null(breaks)) {
    breaks <- ceiling(n/h) - 2
  } else {
    if(breaks > ceiling(n/h) - 2) {
      breaks0 <- breaks
      breaks <- ceiling(n/h) - 2
      warning(sprintf("requested number of breaks = %i too large, changed to %i", breaks0, breaks))
    }
  }

  ## compute ith row of the RSS diagonal matrix, i.e,
  ## the recursive residuals for segments starting at i = 1:(n-h+1)
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
  RSS.triang <- sapply(1:(n-h+1), RSSi)

  ## function to extract the RSS(i,j) from RSS.triang
  RSS <- function(i,j) {
    RSS.triang[[i]][j - i + 1]
  }

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
      for(m in (ncol(RSS.table)/2 + 1):breaks) {
        my.index <- (m*h):(n-h)
        my.RSS.table <- RSS.table[,c((m-1)*2 - 1, (m-1)*2)]
        my.RSS.table <- cbind(my.RSS.table, NA, NA)
        for(i in my.index) {
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

  ## extract optimal breaks
  extract.breaks <- function(RSS.table, breaks)
  {
    if((breaks*2) > ncol(RSS.table)) {
      stop("compute RSS.table with enough breaks before")
    }
    index <- RSS.table[, 1, drop = TRUE]
    break.RSS <- sapply(index, function(i) RSS.table[as.character(i),breaks*2] + RSS(i + 1, n))
    opt <- index[which.min(break.RSS)]
    if(breaks > 1) {
      for(i in ((breaks:2)*2 - 1)) {
        opt <- c(RSS.table[as.character(opt[1]),i], opt)
      }
    }
    names(opt) <- NULL
    return(opt)
  }

  opt <- extract.breaks(RSS.table, breaks)

  if(is.ts(data)) {
    if(NROW(data) == n) {
      datatsp <- tsp(data)
    } else {
      datatsp <- c(1/n, 1, n)
    }
  } else {
    env <- environment(formula)
    if(missing(data)) {
      data <- env
    }
    orig.y <- eval(attr(terms(formula), "variables")[[2]], data, env)
    if(is.ts(orig.y) & (NROW(orig.y) == n)) {
      datatsp <- tsp(orig.y)
    } else {
      datatsp <- c(1/n, 1, n)
    }
  }

  RVAL <- list(breakpoints = opt,
               RSS.table = RSS.table,
               RSS.triang = RSS.triang,
               RSS = RSS,
               extract.breaks = extract.breaks,
               extend.RSS.table = extend.RSS.table,
               nobs = n,
               nreg = k,
               y = y,
               X = X,
               call = match.call(),
               datatsp = datatsp)
  class(RVAL) <- c("breakpointsfull", "breakpoints")
  RVAL$breakpoints <- breakpoints(RVAL)$breakpoints
  return(RVAL)
}


breakfactor <- function(obj, breaks = NULL, labels = NULL, ...)
{
  if("breakpointsfull" %in% class(obj)) {
    obj <- breakpoints(obj, breaks = breaks)
  }
  breaks <- obj$breakpoints
  if(all(is.na(breaks))) {
    return(factor(rep("segment1", obj$nobs)))
  }
  nbreaks <- length(breaks)
  fac <- rep(1:(nbreaks + 1), c(breaks[1], diff(c(breaks, obj$nobs))))

  if(is.null(labels)){
    labels <- paste("segment", 1:(nbreaks+1), sep = "")
  }
  fac <- factor(fac, labels = labels, ...)
  return(fac)
}


pargmaxV <- function(x, xi = 1, phi1 = 1, phi2 = 1)
{
  phi <- xi * (phi2/phi1)^2

  G1 <- function(x, xi = 1, phi = 1)
  {
    x <- abs(x)
    frac <- xi/phi
    rval <- - exp(log(x)/2 - x/8 - log(2*pi)/2) -
      (phi/xi * (phi + 2*xi)/(phi+xi)) * exp((frac * (1 + frac) * x/2) + pnorm(-(0.5 + frac) * sqrt(x), log.p = TRUE)) +
      exp(log(x/2 - 2 + ((phi + 2 * xi)^2)/((phi + xi)*xi)) + pnorm(-sqrt(x)/2, log.p = TRUE))
    rval
  }

  G2 <- function(x, xi = 1, phi = 1)
  {
    x <- abs(x)
    frac <- xi^2/phi
    rval <- 1 + sqrt(frac) * exp(log(x)/2 - (frac*x)/8  - log(2*pi)/2) +
      (xi/phi * (2*phi + xi)/(phi + xi)) * exp(((phi + xi) * x/2) + pnorm(-(phi + xi/2)/sqrt(phi) * sqrt(x), log.p = TRUE)) -
      exp(log(((2*phi + xi)^2)/((phi+xi)*phi) - 2 + frac*x/2) + pnorm(-sqrt(frac) * sqrt(x)/2 , log.p = TRUE))
    rval
  }

  ifelse(x < 0, G1(x, xi = xi, phi = phi), G2(x, xi = xi, phi = phi))
}


confint.breakpointsfull <- function(object, level = 0.95, breaks = NULL, ...)
{
  X <- object$X
  y <- object$y
  n <- object$nobs
  a2 <- (1 - level)/2

  myfun <- function(x, level = 0.975, xi = 1, phi1 = 1, phi2 = 1)
    (pargmaxV(x, xi = xi, phi1 = phi1, phi2 = phi2) - level)

  myprod <- function(delta, mat) as.vector(crossprod(delta, mat) %*% delta)

  bp <- breakpoints(object, breaks = breaks)$breakpoints

  if(any(is.na(bp))) {
    stop("cannot compute confidence interval when `breaks = 0'")
  }

  nbp <- length(bp)
  upper <- rep(0, nbp)
  lower <- rep(0, nbp)
  bp <- c(0, bp, n)

  res <- residuals(object, breaks = breaks)
  sigma1 <- sigma2 <- sum(res^2)/n
  Q1 <- Q2 <- crossprod(X)/n

  Omega1 <- Omega2 <- sigma1 * Q1

  xi <- 1

  X2 <- X[(bp[1]+1):bp[2],,drop = FALSE]
  y2 <- y[(bp[1]+1):bp[2]]
  fm2 <- lm(y2 ~ 0+ X2)
  beta2 <- coef(fm2)
  Q2 <- crossprod(X2)/nrow(X2)

  for(i in 2:(nbp+1)) {
    X1 <- X2
    y1 <- y2
    beta1 <- beta2
    sigma1 <- sigma2
    Q1 <- Q2
    Omega1 <- Omega2

    X2 <- X[(bp[i]+1):bp[i+1],,drop = FALSE]
    y2 <- y[(bp[i]+1):bp[i+1]]
    fm2 <- lm(y2 ~ 0 + X2)
    beta2 <- coef(fm2)
    delta <- beta2 - beta1

    Q2 <- crossprod(X2)/nrow(X2)

    Oprod1 <- myprod(delta, Omega1)
    Oprod2 <- myprod(delta, Omega2)
    Qprod1 <- myprod(delta, Q1)
    Qprod2 <- myprod(delta, Q2)

    xi <- Qprod2/Qprod1
    phi1 <- sqrt(sigma1)
    phi2 <- sqrt(sigma2)

    p0 <- pargmaxV(0, phi1 = phi1, phi2 = phi2, xi = xi)
    if(is.nan(p0) || p0 < a2 || p0 > (1-a2)) {
      warning(paste("Confidence interval", as.integer(i-1),
                    "cannot be computed: P(argmax V <= 0) =", round(p0, digits = 4)))
      upper[i-1] <- NA
      lower[i-1] <- NA
    } else {
      ub <- lb <- 0
      while(pargmaxV(ub, phi1 = phi1, phi2 = phi2, xi = xi) < (1 - a2)) {
        ub <- ub + 1000
      }
      while(pargmaxV(lb, phi1 = phi1, phi2 = phi2, xi = xi) > a2) {
        lb <- lb - 1000
      }

      upper[i-1] <- uniroot(myfun, c(0, ub), level = (1-a2), xi = xi, phi1 = phi1, phi2 = phi2)$root
      lower[i-1] <- uniroot(myfun, c(lb, 0), level = a2, xi = xi, phi1 = phi1, phi2 = phi2)$root

      upper[i-1] <- upper[i-1] * phi1^2 / Qprod1
      lower[i-1] <- lower[i-1] * phi1^2 / Qprod1
    }
  }
  bp <- bp[-c(1, nbp+2)]
  bp <- cbind(bp - ceiling(upper), bp, bp - floor(lower))
  a2 <- round(a2 * 100, digits = 1)
  colnames(bp) <- c(paste(a2, "%"), "breakpoints", paste(100 - a2, "%"))
  rownames(bp) <- 1:nbp

  RVAL <- list(confint = bp,
               nobs = object$nobs,
               nreg = object$nreg,
               call = match.call(),
               datatsp = object$datatsp)
  class(RVAL) <- "confint.breakpoints"
  return(RVAL)
}


residuals.breakpointsfull <- function(object, breaks = NULL, ...)
{
  X <- object$X
  y <- object$y
  n <- object$nobs
  bp <- obp <- breakpoints(object, breaks = breaks)$breakpoints

  if(any(is.na(bp))) {
    nbp <- 0
    bp <- c(0, n)
  } else {
    nbp <- length(bp)
    bp <- c(0, bp, n)
  }
  rval <- NULL

  for(i in 1:(nbp+1))
  {
    X2 <- X[(bp[i]+1):bp[i+1],,drop = FALSE]
    y2 <- y[(bp[i]+1):bp[i+1]]
    rval <- c(rval, lm.fit(X2, y2)$residuals)
  }
  rval <- ts(as.vector(rval))
  tsp(rval) <- object$datatsp

  return(rval)
}

