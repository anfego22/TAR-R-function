import torch
from lstar import lstar


def test_forward() -> None:
    """Forward test."""
    torch.manual_seed(0)
    X1 = torch.tensor([
        [2, 3, 4]
    ], dtype=torch.int16).transpose(0, 1)
    X2 = torch.tensor([
        [1, 1, 1],
        [3, 4, 5],
        [1, 2, 3]
    ], dtype=torch.int16).transpose(0, 1)
    g = torch.tensor([0.5, 0.7109495, 0.85814894])[:, None]
    Xc = torch.cat([X1*g, X2*(1-g)], dim=1)
    y = torch.tensor([4, 5, 6], dtype=torch.int16)
    y_lagged = torch.tensor([3, 4, 5], dtype=torch.int16)
    w = torch.tensor([[1.5410],
                      [-0.2934],
                      [-2.1788],
                      [0.5684]])
    y_hat = torch.mm(Xc, w)
    loss = torch.nn.MSELoss()
    expected_loss = loss(y_hat, y[:, None])
    model = lstar([[2], [1, 3]], [False, True])
    loss = model.forward(y, [X1, X2], y_lagged)
    if loss - expected_loss > 0.001:
        print(f"Forward pass test failed! loss was {loss:.4f} expected was {expected_loss:.4f}")
        raise
    else:
        print("Forward pass test pass!")
        return


def test_fit():
    lag_list = [[2], [1, 3]]
    fit_intercept = [False, True]
    y = torch.tensor([1., 2., 3., 4., 5., 6.])
    torch.manual_seed(0)
    model = lstar(lag_list, fit_intercept)
    # loss = model.forward(y, [X1, X2], y_lagged)
    # expected_loss = 17.1262
    # if loss - expeced_loss > 0.001:
    #     print(
    #         f"Forward integrated test failed! loss was {loss:.4f} expected was {expected_loss:.4f}")
    #     return
    # else:
    #     print("Forward integrated test pass!")
    model.fit(y)


test_forward()
test_fit()
