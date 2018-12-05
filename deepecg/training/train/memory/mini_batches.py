"""
mini_batches.py
---------------
This module provides classes and functions for generating mini-batches.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np


def mini_batch_generator(x, y, mini_batch_size, seed, gpu_count):
    """Creates a list of random mini-batches from (x, y)"""
    # Increase mini-batch size to accommodate more than 1 GPU
    mini_batch_size = mini_batch_size_update(mini_batch_size=mini_batch_size, gpu_count=gpu_count)

    # Setup variables
    np.random.seed(seed)  # Set random state for mini-batches
    m = x.shape[0]        # Number of training examples
    n_y = y.shape[1]      # Number of classes
    mini_batches = []     # List of mini-batches

    # Shuffle (x, y)
    permutation = list(np.random.permutation(m))
    shuffled_x = x[permutation]
    shuffled_y = y[permutation].reshape((m, n_y))

    # Compute number of complete mini batches
    num_complete_minibatches = int(np.floor(m / mini_batch_size))

    # Partition (shuffled_x, shuffled_y)
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_x[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    # Handling the end case
    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[num_complete_minibatches * mini_batch_size:]
        mini_batch_y = shuffled_y[num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def mini_batch_size_update(mini_batch_size, gpu_count):
    """Increase mini-batch size to accommodate more than 1 GPU."""
    if gpu_count <= 1:
        return mini_batch_size
    elif gpu_count > 1:
        return mini_batch_size * gpu_count
