predict.tar <- function(object, n.ahead = 5){
    y.o <- object$y.est$y.ori
    y <- (y.o - mean(y.o))/sd(y.o)
    d <- object$d; c <- (object$c + mean(y))/sd(y)
    tresh <- object$y.est$tresh
    gamma <- object$gamma
    type <- object$y.est$type
    phi <- object$y.est$phi
    p1 <- object$y.est$p1; p2 <- object$y.est$p2
    predic <- c()
    if(sum(tresh - y.o)==0){
        for (i in 1:n.ahead){
            n <- length(y)
            x1 <- matrix(c(1, y[(n - p1 + 1):n]), nr = 1)
            x2 <- matrix(c(1, y[(n - p2 + 1):n]), nr = 1)
            g <- transi.funct(y[n - d], c, gamma, type)
            xc <- cbind(x1*(1 - g), x2*g)
            predic[i] <- xc%*%phi
            y <- c(y, predic[i])
        }
    }
    if(sum(tresh - y.o)!=0){
        for (i in 1:n.ahead){
            n <- length(y)
            x1 <- matrix(c(1, y[(n - p1 + 1):n]), nr = 1)
            x2 <- matrix(c(1, y[(n - p2 + 1):n]), nr = 1)
            g <- transi.funct(tresh[n - d], c, gamma, type)
            xc <- cbind(x1*(1 - g), x2*g)
            predic[i] <- xc%*%phi
            y <- c(y, predic[i])
        }
    }
    result <- list(y = (y*sd(y.o) + mean(y.o)),
                   predic = (predic*sd(y.o) + mean(y.o)))
    return(result)
}

