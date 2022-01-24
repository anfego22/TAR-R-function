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
    if not np.all(np.equal(res, expected)):
        print(f"Error, se esperaba {expected} pero la funcion arrojo {res}")
        raise
    else:
        print("Indicator test 1 pass!")
    return


def test_threshold_matrix() -> None:
    """Test for threshold matrix."""
    lags = [[1], [1, 2]]
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
    tarObj = tar.star(lags)
    tarObj.threshold_matrix(y)
    if not np.all(np.equal(tarObj.lagged_matrix, expected_out)):
        print(f"Error, se esperaba {expected_out} pero el resultado fue {thres}")
        print("Error, threshold_matrix test 1 failed")
        raise
    else:
        print("Success, threshold_matrix test 1 pass")
    return


def test_design_matrix() -> None:
    """Test the design matrix."""
    lags = [[2], [1, 3]]
    y = np.array([1, 2, 3, 4, 5, 6])
    X1_expected = np.array([
        [1, 1, 1],
        [2, 3, 4]
<< << << < HEAD
    ]).transpose()
    X2_expected = np.array([
        [1, 1, 1],
        [3, 4, 5],
        [1, 2, 3]
    ]).transpose()


== == == =
    ]).transpose()
    X2_expected=np.array([
        [1, 1, 1],
        [3, 4, 5],
        [1, 2, 3]
    ]).transpose()
>> >>>> > eef4745(Fix bug. Design test pass)
    y_expect=np.array([4, 5, 6])
    tarObj=tar.star(lags)
    y, X=tarObj.design_matrix(y)
    if np.all(np.equal(y, y_expect)):
        print("Dependent variable test pass!")
    else:
        print("Dependent variable test fail!")
        raise
    if np.all(np.equal(X[0], X1_expected)):
        print("First regime test 1 pass, design_matrix")
        return
    else:
        print("First regime test 1 failed, design_matrix")
        print(X[0])
        print(X1_expected)
        raise
    if np.all(np.equal(X[1], X2_expected)):
        print("Second regime test 1 pass, design_matrix")
        return
    else:
        print("Second regime test 1 failed, design_matrix")
        print(X[1])
        print(X2_expected)
        raise


def syntetic_data(d: int= 2, c: float = .5, T: int = 10000) -> np.ndarray:
    """Create data with a specify dynamics.

    Syntetic data is created with parameters
    phi_1 = [.7, 0.4], phi_2 = [.2, 0.2, 0.3]
    d = 2 and c = .5
    """
    x=[0]*T
    for t in range(1, T-1):
        if x[t-d] > c:
            x[t+1]=0.7 + 0.4 * x[t] + np.random.randn(1)[0]
        else:
            x[t+1]=.2 + 0.2 * x[t] + 0.3 * x[t-1] + np.random.randn(1)[0]
    return np.array(x)




def test_tar_syntetic(T = 10000):
    y=syntetic_data(T)
    objTar=tar.star([[1], [1, 2]])
    objTar.fit(y)
    coeff_expected=[.7, .4, .2, .2, .3]
    c_expected=.5
    d_expected=2
    coeff_result=objTar.params['params']
    c_result=objTar.params['c']
    d_result=objTar.params['d']
    print(f"Expected coefficients are {coeff_expected} and the result was {coeff_result}\n")
    print(f"Expected threshold is {c_expected} and the result was {c_result}\n")
    print(f"Expected lag is {d_expected} and the result was {d_result}\n")
    return None


def test_tar_tsay():
    y = np.genfromtxt("Data/m-unrate.txt", skip_header=True)[:, 3]
    print(y.shape)
    y = np.diff(y, 1)
    print(y.shape)
    print(y[0:10])
    lag_r1 = [2, 3, 4, 12]
    lag_r2 = [2, 3, 12]
    objTar = tar.star([lag_r1, lag_r2], [False, False])
    objTar.fit(y)
    print(objTar.params)


test_indicator()
test_threshold_matrix()
test_design_matrix()
test_tar_syntetic(10000)
test_tar_tsay()
