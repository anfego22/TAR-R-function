est.tar <- function(y.ori, p1 = 2, p2 = 2, pi0 = .15,
                    type = c("l", "i", "e"), tresh = "Self",
                    critic = c(19.77, 11.04, 7.69), c0 = c(0, 12),
                    est.method = "ols", delta = 1e-04,
                    doParallel = TRUE, cluster = 2,
                    method = c("Nelder-Mead", "BFGS", "CG", "L-BFGS-B",
                        "SANN", "Brent"), control = ls()){
    source("C:/Users/Andres/SkyDrive/Documents/Documentos/FOREX/Source/TAR_predict.R")
    source("C:/Users/Andres/SkyDrive/Documents/Documentos/FOREX/Source/TAR_trans_funct.R")
    library(foreach)
    library(doParallel)
            
    ##  Estimate the Treshold Autoregresive model (TAR)  for
    ##  two regimes, with logistic, indicator or exponencial transition function.
    ##  The model TAR:
    ##
    ##      y_t=phi_1*y1_t*(1 - G(gamma,d,c)) + phi_2*y2_t*G(gamma,d,c) + e_t
    ##
    ##  Where:
    ##      y_t:   An observation in time t
    ##      phi_1: A row vector of unknown parameter for the first regime.
    ##             The first element is the constant term.
    ##      y1_t:  A column vector with the significant lags of y_t
    ##             in first regime
    ##      phi_2: A row vector of unknown parameter for the second regime
    ##             The first element is the constant term.
    ##      y2_t:  A column vector with the significant lags of y_t in
    ##             second regime
    ##      G():   The transition function of the TAR
    ##      gamma: The scale parameter of the transition function
    ##      c:     The treshold value
    ##      d:     Delay parameter
    ##      e_t:   A random normal variable
    ##
    ##  The optimization process used is sequencial least squares for the
    ##  identical transition function. The process consist in chosing
    ##  {gamma,d,c,phi_1,phi_2} that minimizing the sum of square residuals.
    ##
    ##
    ##  Args:
    ##     y:      A univarieta time series
    ##     p1:     The number of lags in the first regimen
    ##     p2:     The number of lags in the second regimen
    ##     pi0:    A number betwen 0 and 1 that specify the lenght
    ##             of the sample for computing the residuals
    ##     miu:    Error term mean
    ##     type:   The type of the transicion function
    ##                l, (logistic function),
    ##                i, (indicator function)
    ##                e, (exponencial function)
    ##     tresh:  Default self, which estimate the Self Existing Treshold
    ##             Auto regressive, or a vector with the same length of y.ori
    ##     critic: The critical value to construct the confidence
    ##             interval (99%, 95%, 90% respectivly)
    ##     c0:     A vector of thwo elements. First element must be
    ##             the starting value for c, second for gamma and
    ##             the last one for d.
    ## est.method: Default ols, which estimates the coefficient by te
    ##             least squares. other estimation method available is
    ##             "ridge" which estimates ridge regression.
    ##     method: The method of the optimization algorithm.
    ##             only necessary for logistic and exponencial type.
    ##             See optim for details of the algorithm.

    ## Estandarizing and creating the dising matrix y1_t y2_t
    if(tresh[1] == "Self") tresh <- y.ori
    tresh.o <- tresh
    tresh <- embed((tresh - mean(tresh))/sd(tresh),
                   (max(p1, p2) + 1))
    x <- embed((y.ori - mean(y.ori))/sd(y.ori),
               max(p1, p2) + 1)
    std <- sd(y.ori)
    y <- x[, 1]
    x <- cbind(1, x[, 2:(max(p1, p2) + 1)])
    x1 <- x[, 1:(p1 + 1)]
    x2 <- x[, 1:(p2 + 1)]
    T <- length(y)
    ##
    ## Indicator transition function
    ##
    ## Algorithm:
    ## Sequencial least square
    ##
    if (type == "i"){
        sigma <- function(c, gamma, d, y, x, x1, x2, tresh, est.method, delta, type){
            g <- transi.funct(y = tresh[, (d + 1)], gamma, c, type)
            xc <- cbind(x1*(1 - g), x2*g)
            phi <- switch(est.method,
                          "ols" = chol2inv(chol(t(xc)%*%xc))%*%t(xc)%*%y,
                          "ridge" = chol2inv(chol(t(xc)%*%xc
                              + delta*diag(ncol(xc))))%*%t(xc)%*%y)
            resid <- y - xc%*%phi
            sigma <- (1/T)*t(resid)%*%resid
        }
        ## Making the set C to find the treshold value
        ## and the delay parameter d.
        C <- sort(tresh, decreasing = TRUE)
        C <- C[round(T*pi0):round((1 - pi0)*T)]
        cval <- matrix(0, nc = max(p1, p2), nr = length(C))

        ##for(i in 1:max(p1, p2)){
        if(doParallel==TRUE){
            makeCluster(cluster)->cl
            registerDoParallel(cl)
            foreach(i = 1:max(p1, p2))%dopar%{
                source("C:/Users/Andres/SkyDrive/Documents/Documentos/FOREX/Source/TAR_predict.R")
                source("C:/Users/Andres/SkyDrive/Documents/Documentos/FOREX/Source/TAR_trans_funct.R")
                cval[, i] <- apply(matrix(C, nc = 1), MARGIN = 1, FUN = sigma,
                                   gamma = 0, d = i, y = y, x = x, tresh = tresh,
                                   est.method = est.method, delta = delta,
                                   x1 = x1, x2 = x2, type = type)
            }
        }
        ## Determining the c and d for which cval in minimum
        l <- arrayInd(which.min(cval), .dim = dim(cval))
        c <- C[l[1]]; d <- l[2]
        ## Making the empirical confidence interval for c
        LRint <- T*((cval[, d] - min(cval[, d]))/min(cval[, d]))
        int <- LRint[LRint < critic[1]]
        a <- match(int, LRint)
        Int <- range(C[a])
        ## Results
        g <- transi.funct(tresh[, (d + 1)], c, gamma = 0, type)
        xc <- cbind(x1*(1 - g), x2*g)

        phi <- switch(est.method,
                      "ols" = chol2inv(chol(t(xc)%*%xc))%*%t(xc)%*%y,
                      "ridge" = chol2inv(chol(t(xc)%*%xc
                          + delta*diag(ncol(xc))))%*%t(xc)%*%y)
        resid <- y - xc%*%phi
        yest <- xc%*%phi*std + mean(y.ori)
        phi1 <- phi[1:(p1 + 1)]; phi2 <- phi[(p1 + 2):length(phi)]
        m3 <- max(length(phi1), length(phi2))
        length(phi1) <- m3; length(phi2) <- m3;
        phit <- rbind(phi1, phi2)
        y.est <- list(y.ori = y.ori, resid = resid, tresh = tresh.o,
                      p1 = p1, p2 = p2, phi = phi, type = type)
        resultados <- list(resid = resid, phit = phit,
                           c = (c*sd(tresh.o) + mean(tresh.o)),
                           d = d, y.est = y.est, xc = xc,
                           yest = yest, Int = Int, gamma = 0)
    }
    ## When The transition function is the logistic or exponencial
    ##
    ## Algorithm:
    if (type == "l" || type == "e")  {
        sigma <- function(param, d, y, x, x1, x2, tresh,
                          est.method, type, delta){
            g <- transi.funct(y = tresh[, (d + 1)],
                              gamma = param[2],
                              c = param[1], type = type)
            xc <- cbind(x1*(1 - g), x2*g)
            phi <- switch(est.method,
                          "ols" = chol2inv(chol(t(xc)%*%xc))%*%t(xc)%*%y,
                          "ridge" = chol2inv(chol(t(xc)%*%xc
                              + delta*diag(ncol(xc))))%*%t(xc)%*%y)
            resid<-y - xc%*%phi
            sigma<-(1/T)*t(resid)%*%resid
        }
        val<-list()
        for(j in 1:max(p1, p2)){
            v <- optim(c0, fn = sigma, gr = NULL, d = j, y = y, x = x,
                       x1 = x1, x2 = x2, tresh = tresh, type = type,
                            est.method = est.method, delta = delta,
                       method = method, control = control)
            val<-cbind(v,val)
        }
        d <- apply(matrix(unlist(val[2, ]), nc = max(p1, p2)),
                   1, which.min)
        val.algr <- val[, d]
        c <- val.algr$par[1]; gamma <- val.algr$par[2]
        g <- transi.funct(tresh[, (d + 1)], c = c,
                          gamma = gamma, type = type)
        xc <- cbind(x1*(1 - g), x2*g)
        phi <- chol2inv(chol(t(xc)%*%xc))%*%t(xc)%*%y
        phi1 <- phi[1:(p1 + 1)]; phi2 <- phi[(p1 + 2):length(phi)]
        m3 <- max(length(phi1), length(phi2))
        length(phi1) <- m3; length(phi2) <- m3;
        phit <- rbind(phi1, phi2)
        resid <- y - xc%*%phi
        yest <- xc%*%phi*std + mean(y.ori)
        y.est <- list(y.ori = y.ori, resid = resid, p1 = p1, p2 = p2,
                      type = type, phi = phi, tresh = tresh.o)
        ## Creating confidence interval, in other ocasion

        resultados<-list(phi = phi, c = (c*sd(tresh.o) + mean(tresh.o)),
                         d = d, gamma = gamma, y.est = y.est,
                         val.algr = val.algr, val = val,
                         yest = yest, phit = phit)
    }
    return(resultados)
}
## Author:
## Andres Felipe Gonzalez














