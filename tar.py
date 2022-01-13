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
from sklearn.preprocessing import StandardScaler


def logistic(gamma, y, c):
    """Logistic function."""
    return 1 / (1 + np.exp(-gamma * (y - c)))


def indicator(y: np.array, c: int):
    """Indicator function."""
    return (y > c).astype(int)


def exponential(gamma: float, y: np.array, c: int) -> np.array:
    """Exponential function."""
    return (1 - np.exp(-gamma * (y - c) ^ 2))


TRANSITION_FUN = {'l': logistic, 'i': indicator, 'e': exponential}


class star():
    """Calculate the Self Existing Threshold Autoregresive Model.

    The SETAR model is a non linear model define by the formula:
    y_t = phi_1 * y_{t-1} * G(y_{t-d}, c, g) + phi_2 * y_{t-1} * G(y_{t-d}, c, g)

    Where G(y_{t-d}, c, g) is a non linear function depending on the lagged
    variable y_{t-d}, a threshold variable c and other variables related to the
    G() function. The G() function could be the indicator function
    y_{t-d} > c: 1 or 0 otherwise. Or could be another function, logistic or exponential
    which result in the LSTAR or ESTAR model.
    """

    def __init__(self, lags_1: int, lags_2: int, pi0: float, type: str = 'l') -> None:
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
        self.lag_max = max(lags_1, lags_2)

    def design_matrix(self, X: np.array) -> tuple:
        """Concatenate the matrix of lagged variables for both regimes."""
        intercept = np.ones((len(X), 1))
        lagged_vars = [np.roll(X, p) for p in range(self.lag_max)]
        X = np.concatenate([intercept] + lagged_vars, axis=1)[self.lag_max:, :]
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        return (X[:, 0], X[:, 1:])

    def threshold_matrix(self, y: np.array) -> None
        """Compute the matrix of possible lagged variables."""
        thres = [np.roll(y, p) for p in range(self.lag_max)]
        self.thres = np.concatenate(thres)[self.lag_max:, :]

    def solution_indicator(self, X: np.array, y: np.array) -> tuple:
        """Returns the solution optimal parameters given a threshold.

        Args:
        X: Design matrix
        y: Vector with dependant variables

        Returns:
        Tuple containing the parameters and sigma^2
        """
        XX = np.matmul(np.transpose(X), X)
        XX_inv = np.linalg.inv(XX)
        params = np.matmul(np.matmul(XX_inv, np.transpose(X)), y)

        residuals = y - np.matmul(X, params)
        sigma = np.matmul(np.transpose(residuals), residuals))
        return (params, sigma)

    def sequential_least_square(self, X: np.array, y: np.array) -> tuple:
        """Find the values for c and the delay parameter.

        Use a greedy approach, to search for all posible values of C and all
        posible values of d.

        Args:
        X: Design matrix
        y: Vector with dependant variables

        Returns:
        Tuple containing the parameters and sigma^2
        """
        posible_c=np.sort(thres[:, 0], kind = "quicksort")

        for i in range(self.lag_max):


    def fit(self, X: np.array) -> None:
        """Fit the model.

        The optimization process used is sequencial least squares for the
        identical transition function. The process consist in chosing
        {gamma,d,c,phi_1,phi_2} that minimizing the sum of square residuals.

        Args:
        X: Numpy array of shape (N, 1) where N is the number of samples
        """
        X=self.design_matrix(X)
