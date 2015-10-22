transi.funct <- function(y, gamma, c, type = c("l","e","i")){
    ## Compute the transition function of TAR
    ##
    ## Arg:
    ##    y:     One number vector
    ##    c:     A parameter containing the treshold value
    ##    gamma: A parameter indicating the transition smoothness
    ##    type:  A word indicating the function to use:
    ##           l, (logistic function),
    ##           i, (indicator function)
    ##           e, (exponencial function).
    transi.funct <- switch(type,
                           "l" = 1/(1 + exp(-gamma*(y - c))),
                           "i" = ifelse(y >= c,0,1),
                           "e" = (1 - exp(-gamma*(y - c)^2)))
    return(transi.funct)
}
