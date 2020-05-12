efp <- function(X, ...)
{
  UseMethod("efp")
}

efp.formula <- function(formula, data = list(),
                type = c("Rec-CUSUM", "OLS-CUSUM", "Rec-MOSUM", "OLS-MOSUM",
                "RE", "ME", "Score-CUSUM", "Score-MOSUM", "fluctuation"),
                h = 0.15, dynamic = FALSE, rescale = TRUE, ...)
{
  if(!inherits(formula, "formula")) {
    X <- if(is.matrix(formula$x))
      formula$x
    else model.matrix(terms(formula), model.frame(formula))
    y <- if(is.vector(formula$y))
      formula$y
    else model.response(model.frame(formula))
  } else {
    mf <- model.frame(formula, data = data)
    y <- model.response(mf)
    X <- model.matrix(formula, data = data)
  }

  n <- nrow(X)

  if (h > 1) {
    if (h %% 1 != 0) {
      warning("h has non-integer value > 1; will be rounded")
      h = round(h)
    }
    h = h / n
  }

  if(dynamic)
  {
    Xnames <- colnames(X)
    X <- cbind(y[1:(n-1)],X[2:n,])
    colnames(X) <- c("lag", Xnames)
    y <- y[-1]
    n <- n-1
  }
  k <- ncol(X)
  type <- match.arg(type)

  if(type == "fluctuation") type <- "RE"

  retval <- list(process = NULL,
                 type = type,
                 nreg = k,
                 nobs = n,
                 call = match.call(),
                 formula = formula,
                 par = NULL,
                 type.name = NULL,
                 lim.process = NULL,
                 coef = NULL,
                 Q12 = NULL,
                 datatsp = NULL,
                 rescale = rescale)

  orig.y <- NULL

  fm <- lm.fit(X,y)
  e <- fm$residuals
  sigma <- sqrt(sum(e^2)/fm$df.residual)
  nh <- floor(n*h)
  process <- cumsum(c(0,e))
  process <- process[-(1:nh)] - process[1:(n-nh+1)]
  process <- process/(sigma*sqrt(n))
  if(is.ts(data)) {
      if(NROW(data) == n) process <- ts(process, end = time(data)[(n-floor(0.5 + nh/2))], frequency = frequency(data))
  } else {
      env <- environment(formula)
      if(missing(data)) data <- env
      orig.y <- eval(attr(terms(formula), "variables")[[2]], data, env)
      if(is.ts(orig.y) && (NROW(orig.y) == n)) {
          process <- ts(process, end = time(orig.y)[(n-floor(0.5 + nh/2))],
                        frequency = frequency(orig.y))
      } else {
          process <- ts(process, end = (n-floor(0.5 + nh/2))/n,
                        frequency = n)
      }
  }
  retval$par <- h
  retval$type.name <- "OLS-based MOSUM test"
  retval$lim.process <- "Brownian bridge increments"


  if(!is.ts(process))
    process <- ts(process, start = 0, frequency = (NROW(process)-1))

  retval$process <- process

  if(is.ts(data) & NROW(data) == n)
    retval$datatsp <- tsp(data)
  else if(!is.null(orig.y) && is.ts(orig.y) & NROW(orig.y) == n)
    retval$datatsp <- tsp(orig.y)
  else
    retval$datatsp <- c(0, 1, n)

  m.fit <- lm.fit(X,y)
  retval$coefficients <- coefficients(m.fit)
  retval$sigma <-  sqrt(sum(m.fit$residual^2)/m.fit$df.residual)
  class(retval) <- c("efp")
  return(retval)
}





efp.matrix <- function(X,y,
                       type = c("Rec-CUSUM", "OLS-CUSUM", "Rec-MOSUM", "OLS-MOSUM",
                                "RE", "ME", "Score-CUSUM", "Score-MOSUM", "fluctuation"),
                       h = 0.15, dynamic = FALSE, rescale = TRUE, ...)
{
  n <- nrow(X)
  if(dynamic)
  {
    Xnames <- colnames(X)
    X <- cbind(y[1:(n-1)],X[2:n,])
    colnames(X) <- c("lag", Xnames)
    y <- y[-1]
    n <- n-1
  }
  k <- ncol(X)
  type <- match.arg(type)
  if(type == "fluctuation") type <- "RE"

  retval <- list(process = NULL,
                 type = type,
                 nreg = k,
                 nobs = n,
                 call = match.call(),
                 # formula = formula,
                 par = NULL,
                 type.name = NULL,
                 lim.process = NULL,
                 coef = NULL,
                 Q12 = NULL,
                 datatsp = NULL,
                 rescale = rescale)

  orig.y <- NULL

  fm <- lm.fit(X,y)
  e <- fm$residuals
  sigma <- sqrt(sum(e^2)/fm$df.residual)
  nh <- floor(n*h)
  process <- cumsum(c(0,e))
  process <- process[-(1:nh)] - process[1:(n-nh+1)]
  process <- process/(sigma*sqrt(n))
  if(is.ts(y)) {
      if(NROW(y) == n) process <- ts(process, end = time(y)[(n-floor(0.5 + nh/2))], frequency = frequency(y))
  }
  retval$par <- h
  retval$type.name <- "OLS-based MOSUM test"
  retval$lim.process <- "Brownian bridge increments"


  if(!is.ts(process))
    process <- ts(process, start = 0, frequency = (NROW(process)-1))

  retval$process <- process

  if(is.ts(y) & NROW(y) == n)
    retval$datatsp <- tsp(y)
  else if(!is.null(orig.y) && is.ts(orig.y) & NROW(orig.y) == n)
    retval$datatsp <- tsp(orig.y)
  else
    retval$datatsp <- c(0, 1, n)

  m.fit <- lm.fit(X,y)
  retval$coefficients <- coefficients(m.fit)
  retval$sigma <-  sqrt(sum(m.fit$residual^2)/m.fit$df.residual)
  class(retval) <- c("efp")
  return(retval)
}


pvalue.efp <- function(x, lim.process, alt.boundary, functional = "max", h = NULL, k = NULL)
{
  if(k > 6) k <- 6
  crit.table <- get("sc.me")[(((k-1)*10+1):(k*10)), ]
  tablen <- dim(crit.table)[2]
  tableh <- (1:10)*0.05
  tablep <- c(0.1, 0.05, 0.025, 0.01)
  tableipl <- numeric(tablen)
  for(i in (1:tablen)) tableipl[i] <- approx(tableh, crit.table[,i], h, rule = 2)$y
  p <- approx(c(0,tableipl), c(1,tablep), x, rule = 2)$y

  return(p)
}


sctest.formula <- function(formula, type = c("Rec-CUSUM", "OLS-CUSUM",
  "Rec-MOSUM", "OLS-MOSUM", "RE", "ME", "fluctuation", "Score-CUSUM", "Nyblom-Hansen",
  "Chow", "supF", "aveF", "expF"), h = 0.15, alt.boundary = FALSE, functional = c("max", "range",
  "maxL2", "meanL2"), from = 0.15, to = NULL, point = 0.5, asymptotic = FALSE, data = list(), ...)
{
    type <- match.arg(type)
    functional <- match.arg(functional)
    dname <- paste(deparse(substitute(formula)))

    process <- efp(formula, type = type, h = h, data = data, ...)
    RVAL <- sctest(process, alt.boundary = alt.boundary, functional = functional)

    RVAL$data.name <- dname
    return(RVAL)
}


sctest.efp <- function(x, alt.boundary = FALSE, functional = c("max", "range", "maxL2", "meanL2"), ...)
{
    h <- x$par
    type <- x$type
    lim.process <- x$lim.process
    functional <- match.arg(functional)
    dname <- paste(deparse(substitute(x)))
    METHOD <- x$type.name
    x <- as.matrix(x$process)
    n <- nrow(x)
    k <- ncol(x)

    STAT <- max(abs(x))
    names(STAT) <- "M0"
    PVAL <- pvalue.efp(STAT, lim.process, alt.boundary, functional = functional, h = h, k = k)
    RVAL <- list(statistic = STAT, p.value = PVAL, method = METHOD, data.name = dname)
    class(RVAL) <- "htest"
    return(RVAL)
}

boundary.efp <- function(x, alpha = 0.05, alt.boundary = FALSE, functional = "max", ...)
{
    h <- x$par
    type <- x$type
    lim.process <- x$lim.process
    functional <- match.arg(functional, c("max", "range", "maxL2", "meanL2"))
    proc <- as.matrix(x$process)
    n <- nrow(proc)
    k <- ncol(proc)

    bound <- uniroot(function(y) {pvalue.efp(y, lim.process, alt.boundary = alt.boundary,
                                             functional = functional, h = h, k = k) - alpha}, c(0,20))$root
    if(functional == "max") {
        bound <- rep(bound, n)
    } else {
        stop("only boundaries for Brownian bridge increments with max functional available")
    }
    if(alt.boundary) warning("no alternative boundaries available for Brownian bridge increments")

    bound <- ts(bound, end = end(x$process), frequency = frequency(x$process))
    return(bound)
}

