"""
labels.py
---------
This module provides a functions for converting labels to one-hot.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import tensorflow as tf


def one_hot_encoding(labels, classes):
    """Converts labels to one-hot labels."""
    # Create a tf.constant equal to classes (depth)
    classes = tf.constant(value=classes, name='classes')

    # Use create one-hot matrix
    one_hot_matrix = tf.one_hot(indices=labels, depth=classes, axis=1)

    # Create tf session
    sess = tf.Session()

    # Run tf session
    one_hot = sess.run(one_hot_matrix)

    # Close tf session
    sess.close()

    return one_hot
