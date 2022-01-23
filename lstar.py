"""Calculate the logistict self existing threshold autoregresive."""
import torch
import utils as ut
import torch.nn.functional as F


class lstar():
	 """Calculate SETAR with logistic transition function.

	The LSTAR model is given by:
	y_t = phi_1 * y_{t-1} * G(y_{t-d}, c, g) + phi_2 * y_{t-1} * G(y_{t-d}, c, g)

	Where G(y_{t-d}, c, g) is the logistic function
	"""
	def __init__(self, lag_list: list, fit_intercept:list=[True, True]) -> None:
		"""Start initial parameter."""
		self.lag_list = lag_list
		self.lag_1 = lags[0]
		self.lag_2 = lags[1]
		self.fit_intercept = fit_intercept
		self.lag_max = max([max(lags[0]), max(lags[1])])

	def forward(self, y: torch.tensor, X: list, y_lag_d: torch.tensor) -> None:
		"""Forward pass of the LSTAR."""
		gamma = torch.tensor(0.9, requires_grad=True)
		c = torch.tensor(3., requires_grad=True)
		g = torch.sigmoid(gamma*(y_lag_d - c))
		Xc = torch.cat([X[0]*g[:, None], X[1]*(1-g[:, None])], dim=1)
		w = torch.randn((Xc.shape[1], 1), requires_grad=True)
		y_hat = torch.mm(Xc, w)
		loss = torch.nn.MSELoss(y_hat, y)
        return loss
		
	def fit(self, y: torch.tensor) -> None:
		"""Calculate the forward pass."""
		y, X = ut.design_matrix(y, self.lag_list, self.fit_intercept)
		lagged_vars = ut.threshold_matrix(y, self.lag_max)
		for d in range(1, self.lag_max + 1):
			  loss = self.forward(y, X, lagged_vars[:, d])
              return None
