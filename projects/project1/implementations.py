import numpy as np

def compute_loss_MSE(y, tx, w):
    """
        Compute the Mean Squared Error (MSE) loss.
        A factor 0.5 is used to be consistent with the lecture notes.

        Args:
            y: labels
            tx: features
            w: weights vector
    """
    MSE_FACTOR = 0.5
    N = len(y)
    e = y - tx.dot(w)
    return MSE_FACTOR * (1 /  N) * np.sum(e ** 2)

def compute_gradient_MSE(y, tx, w):
    """
        Compute the gradient of the Mean Squared Error (MSE) loss.

        Args:
            y: labels
            tx: features
            w: weights vector
    """
    N = len(y)
    e = y - tx.dot(w)
    return -1 / N * tx.T.dot(e)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
        Gradient descent algorithm with MSE loss.

        Args:
            y: labels
            tx: features
            initial_w: initial weights vector
            max_iters: number of iterations
            gamma: step-size
    """
    
    w = initial_w
    gradient = compute_gradient_MSE(y, tx, w)
    loss = compute_loss_MSE(y, tx, w)
    for _ in range(max_iters):
        gradient = compute_gradient_MSE(y, tx, w)
        w -= gamma * gradient
        loss = compute_loss_MSE(y, tx, w)
    return w, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
        Generate a minibatch iterator for a dataset.

        Args:
            y: labels
            tx: features
            batch_size: number of samples in a minibatch
            num_batches: number of batches
            shuffle: shuffle the data before creating minibatches
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index >= end_index:
            break
        yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stochastic_gradient(y, tx, w):
    """
        Compute the gradient of the Mean Squared Error (MSE) loss.

        Args:
            y: labels
            tx: features
            w: weights vector
    """
    e = y - tx.dot(w)
    return -1 / len(y) * tx.T.dot(e)

def mean_squared_error_sgd (y, tx, initial_w, max_iters, gamma):
    """
        Stochastic gradient descent algorithm with MSE loss.

        Args: 
            y: labels
            tx: features
            initial_w: initial weights vector
            max_iters: number of iterations
            gamma: step-size
    """
    if max_iters == 0:
        return initial_w, compute_loss_MSE(y, tx, initial_w)
    w = initial_w
    for _ in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            gradient = compute_stochastic_gradient(y_batch, tx_batch, w)
            w -= gamma * gradient
        loss = compute_loss_MSE(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
        Ridge regression using normal equations.
        Normal equations: w = (X^T X + lambda' I)^-1 X^T y

        Args:
            y: labels
            tx: features
            lambda_: regularization parameter
    """
    N = len(y)
    lambda_ = lambda_ * 2 * N
    XTX = tx.T.dot(tx)
    I = np.eye(tx.shape[1])
    w = np.linalg.inv(XTX + lambda_ * I).dot(tx.T).dot(y)
    loss = compute_loss_MSE(y, tx, w)

    return w, loss