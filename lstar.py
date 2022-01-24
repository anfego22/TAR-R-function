"""SETAR model with logistic transition."""
import torch
import utils as ut


class lstar():
    """Create the lstart.

    The LSTAR model is given by:
    y_t = phi_1 * y_{t-1} * G(y_{t-d}, c, g) + phi_2 * y_{t-1} * G(y_{t-d}, c, g)
    Where G(y_{t-d}, c, g) is the logistic function
    """

    def __init__(self, lag_list: list, fit_intercept: list = [True, True]) -> None:
        """Initialize parameters."""
        self.lag_list = lag_list
        self.lag_1 = lag_list[0]
        self.lag_2 = lag_list[1]
        self.max_lag = max(self.lag_1 + self.lag_2)
        self.loss = torch.nn.MSELoss()
        self.fit_intercept = fit_intercept
        self.parameters()

    def parameters(self) -> None:
        """Set parameters of the model."""
        param_shape = sum(self.fit_intercept) + self.max_lag
        c = torch.tensor(3., requires_grad=True)
        gamma = torch.tensor(.9, requires_grad=True)
        w = torch.randn((param_shape, 1), requires_grad=True)
        self.params = {'gamma': gamma, 'c': c, 'w': w}
        self.optimizer = torch.optim.Adam(list(self.params.values()))

    def forward(self, y: torch.tensor, X: list, y_lagged_d: torch.tensor):
        """Forward pass given the lagg variable."""
        g = torch.sigmoid(self.params['gamma']*(y_lagged_d - self.params['c']))[:, None]
        Xc = torch.cat([X[0]*g, X[1]*(1-g)], dim=1)
        y_hat = torch.mm(Xc, self.params['w'])
        return self.loss(y_hat, y[:, None])

    def fit(self, y: torch.tensor) -> None:
        """Parameter estimation."""
        lagged_vars = ut.threshold_matrix(y, self.max_lag)
        y, X = ut.design_matrix(y, self.lag_list, self.fit_intercept)
        for d in range(1, self.max_lag + 1):
            self.optimizer.zero_grad()
            loss = self.forward(y, X, lagged_vars[:, d])
            loss.backward()
            self.optimizer.step()
            print(self.params)
