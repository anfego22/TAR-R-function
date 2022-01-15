"""Test for TAR functions."""
import numpy as np
import tar as tar


def test_indicator() -> None:
    """Test indicator function."""
    y = np.array([
        0.26256947, 1.39972713, -1.11509627, -0.21995397, 0.0564498,
        -1.0148332, 0.24226806, -0.42289193, 1.29083007, -0.44132848
    ])
    res = tar.indicator(y, .5)
    expected = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0])

    if not np.equal(res, expected):
        print(f"Error, se esperaba {exptected} pero la funcion arrojo {res}")
        raise
    return


def test_threshold_matrix() -> None:
    """Test for threshold matrix."""
    lag_1 = 1
    lag_2 = 2
    pi0 = 0.5
    y = np.array([0.26256947, 1.39972713, -1.11509627, -0.21995397, 0.0564498])
    """
    x_t    = 0.26256947, 1.39972713,-1.11509627, -0.21995397, 0.0564498
    x_{t-1}=      0    , 0.26256947, 1.39972713, -1.11509627,-0.21995397, 
    x_{t-2}=      0    ,     0     , 0.26256947, 1.39972713, -1.11509627

    """
    expected_out = np.array([
        [-1.11509627, -0.21995397, 0.0564498],
        [1.39972713, -1.11509627, -0.21995397],
        [0.26256947, 1.39972713, -1.11509627]
    ]).transpose()
    # The expe
    tarObj = tar.star(lag_1, lag_2, pi0)
    tarObj.threshold_matrix(y)
    if not np.all(np.equal(tarObj.thres, expected_out)):
        print(f"Error, se esperaba {expected_out} pero el resultado fue {tarObj.thres}")
        raise
    else:
        print("Test pass!")
    return
