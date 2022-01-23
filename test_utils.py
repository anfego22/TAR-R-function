import torch
import utils as ut


def test_threshold_matrix() -> None:
    """Test for threshold matrix."""
    y = torch.tensor([0.26256947, 1.39972713, -1.11509627, -0.21995397, 0.0564498])
    """
    x_t    = 0.26256947, 1.39972713,-1.11509627, -0.21995397, 0.0564498
    x_{t-1}=      0    , 0.26256947, 1.39972713, -1.11509627,-0.21995397,
    x_{t-2}=      0    ,     0     , 0.26256947, 1.39972713, -1.11509627
    """
    expected_out = torch.tensor([
        [-1.11509627, -0.21995397, 0.0564498],
        [1.39972713, -1.11509627, -0.21995397],
        [0.26256947, 1.39972713, -1.11509627]
    ])
    expected_out = torch.transpose(expected_out, 0, 1)
    outcome = ut.threshold_matrix(y, 2)
    # The expe
    if not torch.equal(expected_out, outcome):
        print(f"Error, se esperaba {expected_out} pero el resultado fue {thres}")
        print("Error, threshold_matrix test 1 failed")
        raise
    else:
        print("Success, threshold_matrix test 1 pass")
    return


def test_design_matrix() -> None:
    """Test the design matrix."""
    lag_list = [[2], [1, 3]]
    fit_intercept = [False, True]
    y = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int16)
    X1_expected = torch.tensor([
        [2, 3, 4]
    ], dtype=torch.int16).transpose(0, 1)
    X2_expected = torch.tensor([
        [1, 1, 1],
        [3, 4, 5],
        [1, 2, 3]
    ], dtype=torch.int16).transpose(0, 1)
    y_expect = torch.tensor([4, 5, 6], dtype=torch.int16)
    y, X = ut.design_matrix(y, lag_list, fit_intercept)
    if torch.equal(y, y_expect):
        print("Dependent variable test pass!")
    else:
        print("Dependent variable test fail!")
        raise
    if torch.equal(X[0], X1_expected):
        print("First regime test 1 pass, design_matrix")
    else:
        print("First regime test 1 failed, design_matrix")
        print(X[0])
        print(X1_expected)
        raise
    if torch.equal(X[1], X2_expected):
        print("Second regime test 1 pass, design_matrix")
        return
    else:
        print("Second regime test 1 failed, design_matrix")
        print(X[1])
        print(X2_expected)
        raise


test_design_matrix()
test_threshold_matrix()
