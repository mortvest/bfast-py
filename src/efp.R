## efp.formula <- function(formula, data = list(), type = c("OLS-MOSUM"),
##                         h = 0.15, dynamic = FALSE, rescale = TRUE, ...)
efp.matrix <- function(X, y, type = c("OLS-MOSUM"), h = 0.15, dynamic = FALSE, rescale = TRUE, ...)
{
  n <- nrow(X)
  k <- ncol(X)

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

  fm <- lm.fit(X,y)
  ## residual: \hat{y} - y
  e <- fm$residuals
  ## df.residual: degrees of freedom of residuals (n-k)?
  sigma <- sqrt(sum(e^2)/fm$df.residual)
  nh <- floor(n*h)

  process <- cumsum(c(0,e))
  # negative index specifies values to be EXCLUDED
  process <- process[-(1:nh)] - process[1:(n-nh+1)]
  process <- process/(sigma*sqrt(n))


  process <- ts(process, end = time(y)[(n-floor(0.5 + nh/2))], frequency = frequency(y))

  retval$process <- process
  retval$par <- h
  retval$datatsp <- tsp(y)
  retval$coefficients <- coefficients(fm)
  retval$sigma <- sigma

  class(retval) <- c("efp")
  return(retval)
}


pvalue.efp <- function(x, lim.process, alt.boundary, functional = "max", h = NULL, k = NULL)
{
    if(k > 6){
        k <- 6
    }
    ## sc.me = array(60,4)
    crit.table <- get("sc.me")[(((k-1)*10+1):(k*10)), ] ## dim = 10 x 4

    tablen <- dim(crit.table)[2] ## 4
    tableh <- (1:10) * 0.05 ## 0.05, 0.10, ... ,0.5
    tablep <- c(0.1, 0.05, 0.025, 0.01)
    tableipl <- numeric(tablen) ## np.zeros(4)

    for(i in (1:tablen)){ # for i in [1,2,3,4]:
        ## interpolate
        tableipl[i] <- approx(tableh, crit.table[,i], h, rule = 2)$y
    }
    p <- approx(c(0,tableipl), c(1,tablep), x, rule = 2)$y

    return(p)
}


sctest.efp <- function(x, alt.boundary = FALSE, functional = "max", ...)
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
