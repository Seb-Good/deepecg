"""
layers.py
---------
This module provides functions for building basic neural network layers with summaries in tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import tensorflow as tf


def fc_layer(input_layer, neurons, activation, use_bias, name, seed):

    # Create fully connected layer
    fc = tf.layers.dense(
        inputs=input_layer,
        units=neurons,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        bias_initializer=tf.zeros_initializer(),
        name=name
    )

    return fc


def conv_layer(input_layer, kernel_size, strides, dilation_rate, filters, padding, activation,
               use_bias, name, seed):

    # Create convolutional layer
    conv = tf.layers.conv1d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        bias_initializer=tf.zeros_initializer(),
        name=name,
        activation=activation,
        use_bias=use_bias
    )

    return conv


def batch_norm_layer(input_layer, training, name):

    # Create batch normalization layer
    batchnorm = tf.layers.batch_normalization(
        inputs=input_layer,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        training=training,
        name=name
    )

    return batchnorm


def max_pool_layer(input_layer, pool_size, strides, padding, name):

    # Create maxpool layer
    maxpool = tf.layers.max_pooling1d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        name=name
    )

    return maxpool


def dropout_layer(input_layer, drop_rate, seed, training, name):

    # Create dropout layer
    dropout = tf.layers.dropout(
        inputs=input_layer,
        rate=drop_rate,
        seed=seed,
        training=training,
        name=name
    )

    return dropout
