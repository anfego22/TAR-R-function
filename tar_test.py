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


def test_logistic() -> None:
    """Test logistic function."""
    y = np.array([
        0.26256947, 1.39972713, -1.11509627, -0.21995397, 0.0564498,
        -1.0148332, 0.24226806, -0.42289193, 1.29083007, -0.44132848
    ])

    res = tar.logistic(1, y, -4.05)
    expected = np.array([
        0.02215191, 0.06597219, 0.00568007, 0.01378961, 0.01810049, 0.00627534,
        0.0217164, 0.01128544, 0.05957085, 0.01108157
    ])

    return 1 / (1 + exp(-gamma * (y - c)))
