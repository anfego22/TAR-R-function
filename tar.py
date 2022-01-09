"""Treshold Auto Regresive Model.

  Estimate the Treshold Autoregresive model (TAR)  for
  two regimes, with logistic, indicator or exponencial transition function.
  The model TAR:
           y_t=phi_1*y1_t*(1 - G(gamma,d,c)) + phi_2*y2_t*G(gamma,d,c) + e_t
  Where:
  y_t:   An observation in time t
  phi_1: A row vector of unknown parameter for the first regime.
         The first element is the constant term.
  y1_t:  A column vector with the significant lags of y_t
         in first regime
  phi_2: A row vector of unknown parameter for the second regime
         The first element is the constant term.
  y2_t:  A column vector with the significant lags of y_t in
         second regime
  G():   The transition function of the TAR
  gamma: The scale parameter of the transition function
  c:     The treshold value
  d:     Delay parameter
  e_t:   A random normal variable
"""
import numpy as np
from sklearn.preprocessing import StandarScaler


def logistic(gamma, y, c):
    return 1/(1 + np.exp(-gamma*(y - c)))

def indicator(y: np.array, c: int):
    return (y > c).astype(int)

def exponential(gamma: float, y: np.array, c: int) -> np.array:
    return (1 - exp(-gamma*(y - c)^2))


TRANSITION_FUN = {'l': logistic, 'i': indicator, 'e': exponential)

class star():
    """Calculate the Self Threshold Autoregresive Model."""

    def __init__(lags_1: int, lags_2: int, type: str = 'l') -> None:
        """Class initizizer.

        Args:
        lags_1:  Number of lags for regimen one
        lags_2:  Number of lags for regimen two
        pi0:     A number betwen 0 and 1 that specify the lenght
                 of the sample for computing the residuals
        type:    The type of the transicion function:
                    l, (logistic function),
                    i, (indicator function)
                    e, (exponencial function)
        """
        self.lags_1 = lags_1
        self.lags_2 = lags_2
        self.pi0 = pi0
        self.type = type

    def design_matrix(self, X: np.array) -> (np.array, np.array):
        """Create the design matrix."""
        max_lag = max(self.lags_1, self.lags_2)
        intercept = np.ones((len(X), 1))
        lagged_vars = [np.roll(X, p) for p in range(max_lag)]
        X = np.concatenate([intercept] + lagged_vars, axis=1)[max_lag:, :]
        self.scaler = StandarScaler()
        X = self.scaler.fit_transform(X)
        return (X[:, 0], X[:, 1:])

    def fit(self, X: np.array) -> None:
        """Fit the model.

        The optimization process used is sequencial least squares for the
        identical transition function. The process consist in chosing
        {gamma,d,c,phi_1,phi_2} that minimizing the sum of square residuals.

        Args:
        X: Numpy array of shape (N, 1) where N is the number of samples
        """
        X = self.design_matrix(X)

    def solution_indicator():
        """Returns the solution optimal parameters given a threshold.

        Args:
        

        """

            g <- transi.funct(y = tresh[, (d + 1)], gamma, c, type)
            xc <- cbind(x1*(1 - g), x2*g)
            phi <- switch(est.method,
                          "ols" = chol2inv(chol(t(xc)%*%xc))%*%t(xc)%*%y,
                          "ridge" = chol2inv(chol(t(xc)%*%xc
                              + delta*diag(ncol(xc))))%*%t(xc)%*%y)
            resid <- y - xc%*%phi
            sigma <- (1/T)*t(resid)%*%resid
        }
