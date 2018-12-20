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


def print_output_shape(layer_name, net, print_shape):
    """Print output shape of a layer."""
    if print_shape:
        print(layer_name + ' | shape = ' + str(net.shape))


def fc_layer(input_layer, neurons, activation, use_bias, name, seed):
    """Create fully connected layer."""
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


def conv_layer(input_layer, kernel_size, strides, dilation_rate, filters, padding, activation, use_bias, name, seed):
    """Create convolutional layer."""
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
    """Create batch normalization layer."""
    batchnorm = tf.layers.batch_normalization(
        inputs=input_layer,
        momentum=0.9,
        epsilon=0.001,
        center=True,
        scale=True,
        training=training,
        name=name
    )

    return batchnorm


def max_pool_layer(input_layer, pool_size, strides, padding, name):
    """Create maxpool layer."""
    maxpool = tf.layers.max_pooling1d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        name=name
    )

    return maxpool


def avg_pool_layer(input_layer, pool_size, strides, padding, name):
    """Create average pooling."""
    avgpool = tf.layers.average_pooling1d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        name=name
    )

    return avgpool


def dropout_layer(input_layer, drop_rate, seed, training, name):
    """Create dropout layer."""
    dropout = tf.layers.dropout(
        inputs=input_layer,
        rate=drop_rate,
        seed=seed,
        training=training,
        name=name
    )

    return dropout


def temporal_padding(inputs, paddings):
    paddings = [[0, 0], [paddings[0], paddings[1]], [0, 0]]

    return tf.pad(inputs, paddings)


def conv1d(inputs,
           filters,
           kernel_size,
           strides=1,
           padding='same',
           data_format='channels_last',
           dilation_rate=1,
           activation=None,
           use_bias=False,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           bias_initializer=tf.zeros_initializer(),
           name=None):
    """A general wrapper of tf.layers.conv1d() supporting
       1. 'causal' padding method used for WaveNet.
       2. batch normalization when use_bias is False for accuracy.
    """

    if padding == 'causal':
        left_pad = dilation_rate * (kernel_size - 1)
        inputs = temporal_padding(inputs, (left_pad, 0))
        padding = 'valid'

    outputs = tf.layers.conv1d(inputs,
                               filters,
                               kernel_size,
                               strides=strides,
                               padding=padding,
                               data_format=data_format,
                               dilation_rate=dilation_rate,
                               activation=activation,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer)

    if not use_bias:
        axis = -1 if data_format == 'channels_last' else 1
        outputs = tf.layers.batch_normalization(outputs,
                                                axis=axis,
                                                name='{}-batch_normalization'.format(name))

    return outputs
