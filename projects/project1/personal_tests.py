import numpy as np
from grading_tests.conftest import ATOL, RTOL
from implementations import mean_squared_error_gd, mean_squared_error_sgd, ridge_regression

MAX_ITERS = 2
GAMMA = 0.1

BATCH_SIZE = 1
MSE_FACTOR = 0.5

def initial_w():
    return np.array([0.5, 1.0])

def y():
    return np.array([0.1, 0.3, 0.5])

def tx():
    return np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])

def test_mean_squared_error_sgd(y, tx, initial_w):
    # n=1 to avoid stochasticity
    w, loss = mean_squared_error_sgd(
        y[:1], tx[:1], initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 0.844595
    expected_w = np.array([0.063058, 0.39208])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape
    print("Test passed")

def test_ridge_regression_lambda0(y, tx):
    lambda_ = 0.0
    w, loss = ridge_regression(y, tx, lambda_)

    expected_loss = 0.026942
    expected_w = np.array([0.218786, -0.053837])

    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape
    print("Test passed")

def test_ridge_regression_lambda1(y, tx):
    lambda_ = 1.0
    w, loss = ridge_regression(y, tx, lambda_)

    expected_loss = 0.03175
    expected_w = np.array([0.054303, 0.042713])

    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)

    assert loss.ndim == 0
    assert w.shape == expected_w.shape
    print("Test passed")

def test_mean_squared_error_gd_0_step(y, tx):
    expected_w = np.array([0.413044, 0.875757])
    w, loss = mean_squared_error_gd(y, tx, expected_w, 0, GAMMA)

    expected_w = np.array([0.413044, 0.875757])
    expected_loss = 2.959836

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape
    print("Test passed")

def test_mean_squared_error_gd(y, tx, initial_w):
    w, loss = mean_squared_error_gd(
        y, tx, initial_w, MAX_ITERS, GAMMA
    )

    expected_w = np.array([-0.050586, 0.203718])
    expected_loss = 0.051534

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape
    print("Test passed")

test_mean_squared_error_gd_0_step(y(), tx())
test_mean_squared_error_gd(y(), tx(), initial_w())
test_mean_squared_error_sgd(y(), tx(), initial_w())
test_ridge_regression_lambda0(y(), tx())
test_ridge_regression_lambda1(y(), tx())