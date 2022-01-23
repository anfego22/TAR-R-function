import torch


def design_matrix(y: torch.HalfTensor, lag_list: list, fit_intercept: list) -> tuple:
    """Concatenate the matrix of lagged variables for both regimes.

    Args
    y:             An array with shape (T, ) 
    lag_list:      A list with the lag variables for each regime ex
    fit_intercept: A list of bools indicating if we fit an intercept or not
    """
    lag_max = max([max(lag_list[0]), max(lag_list[1])])
    X = threshold_matrix(y, lag_max)
    y = X[:, 0]
    intercept = torch.ones((X.shape[0], 1), dtype=torch.int8)
    X = [X[:, lag_list[0]], X[:, lag_list[1]]]
    for i, with_intercept in enumerate(fit_intercept):
        if with_intercept:
            X[i] = torch.cat([intercept, X[i]], dim=1)
    return (y, X)


def threshold_matrix(y: torch.HalfTensor, lag_max: int) -> None:
    """Compute the matrix of possible lagged variables.

    Args
    y:       An array with shape (T, )
    lag_max: An integer with the maximum lag taking into account for all regimes
    """
    thres = torch.cat([y.roll(p)[:, None] for p in range(lag_max + 1)], dim=1)
    return thres[lag_max:, :]
