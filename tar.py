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


def indicator(y: np.ndarray, c: float) -> np.ndarray:
    """Indicator function, 1 i y_i > c."""
    return (y > c).astype(int)


def logistic(gamma, y, c):
    """Logistic function."""
    return 1 / (1 + np.exp(-gamma * (y - c)))


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

    def __init__(self, lags: list, intercept: list = [True, True],
                 pi0: float = .3) -> None:
        """Class initizizer.

        Args:
        lags:       A list of lists with the lags of each regime ex: [[1, 2, 3], [1, 2]]
        intercept: A list of bools that indicate whether to fit an intercept in that regime.
        pi0:       Controls the number of candidates for the threshold value.
                   (a value of 1 search in all values of y)
        type:      The type of the transicion function:
                   i, (indicator function)
        """
        self.lag_1 = lags[0]
        self.lag_2 = lags[1]
        self.fit_intercept = intercept
        self.pi0 = pi0
        self.delta = 0
        self.lag_max = max([max(lags[0]), max(lags[1])])

    def design_matrix(self, y: np.ndarray) -> tuple:
        """Concatenate the matrix of lagged variables for both regimes."""
        self.threshold_matrix(y)
        X = np.copy(self.lagged_matrix)
        y = X[:, 0]
        intercept = np.ones((X.shape[0], 1))
        X = [X[:, self.lag_1], X[:, self.lag_2]]
        for i, with_intercept in enumerate(self.fit_intercept):
            if with_intercept:
                X[i] = np.concatenate([intercept, X[i]], axis=1)
        return (y, X)

    def threshold_matrix(self, y: np.ndarray) -> None:
        """Compute the matrix of possible lagged variables."""
        thres = np.array([np.roll(y, p) for p in range(self.lag_max + 1)]).transpose()
        self.lagged_matrix = thres[self.lag_max:, :]

    def ordinal_least_square(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Returns the solution optimal parameters given a threshold.

        Args:
        X: Design matrix
        y: Vector with dependant variables

        Returns:
        Tuple containing the parameters and sigma^2
        """
        XX = np.matmul(np.transpose(X), X)
        # regul = np.zeros(XX.shape)
        # Add regularization to the diagonal
        # np.fill_diagonal(regul, self.delta)
        params = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - np.matmul(X, params)
        sigma = np.matmul(np.transpose(residuals), residuals)
        return (params, sigma)

    def sequential_least_square(self, X: list, y: np.ndarray) -> dict:
        """Find the values for c and the delay parameter.

        Use a greedy approach, to search for all posible values of C and all
        posible values of d.

        Args:
        X: Design matrix a list of with two np.ndarray, the result from design_matrix
        y: Vect or with dependant variables

        Returns:
        Tuple containing the parameters and sigma^2
        """
        min_sigma = np.inf
        res = {}
        for lag_d in range(1, self.lag_max+1):
            for c in self.thre_sorted:
                g = indicator(self.lagged_matrix[:, lag_d], c).reshape(-1, 1)
                X_all = np.concatenate([X[0]*g, X[1]*(1-g)], axis=1)
                params, metric = self.ordinal_least_square(X_all, y)
                if metric < min_sigma:
                    min_sigma = metric
                    res['params'] = params
                    res['metric'] = metric
                    res['d'] = lag_d
                    res['c'] = c
        return res

    def fit(self, X: np.ndarray) -> None:
        """Fit the model.

        The optimization process used is sequencial least squares for the
        identical transition function. The process consist in chosing
        {gamma,d,c,phi_1,phi_2} that minimizing the sum of square residuals.

        Args:
        X:      Numpy array of shape (N, 1) where N is the number of samples
        """
        y, X = self.design_matrix(X)
        self.thre_sorted = np.sort(y, kind="mergesort")
        self.params = self.sequential_least_square(X, y)
        return None
