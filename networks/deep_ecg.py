"""
deep_ecg.py
-----------
This module provides a class and methods for building a 13 layer convolutional neural network with tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import tensorflow as tf

# Local imports
from networks.layers import fc_layer, conv_layer, max_pool_layer, batch_norm_layer, dropout_layer


class DeepECG(object):

    """Build the forward propagation computational graph from a 13 layer convolutional neural network."""

    def __init__(self, length, channels, classes, seed=0):

        # Set input parameters
        self.length = length
        self.channels = channels
        self.classes = classes
        self.seed = seed

    def inference(self, input_layer, is_training):
        """Forward propagation of computational graph."""
        # Check input layer dimensions
        assert input_layer.shape[1] == self.length
        assert input_layer.shape[2] == self.channels

        # Define a scope for reusing the variables
        with tf.variable_scope('DeepECG', reuse=tf.AUTO_REUSE):

            ###############################################################
            # ------------------ Convolutional Layer 1 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=input_layer, kernel_size=24, strides=1, filters=320, padding='SAME',
                             activation=None, use_bias=True, name='conv1', seed=self.seed, collections=['train_full'],
                             dilation_rate=1)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm1')

            # Activation
            with tf.variable_scope('relu1') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Max Pool
            net = max_pool_layer(input_layer=net, pool_size=2, strides=2, padding='SAME', name='maxpool1')

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout1')

            ###############################################################
            # ------------------ Convolutional Layer 2 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=16, strides=1, filters=256, padding='SAME',
                             activation=None, use_bias=True, name='conv2', seed=self.seed, collections=['train_full'],
                             dilation_rate=2)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm2')

            # Activation
            with tf.variable_scope('relu2') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout2')

            ###############################################################
            # ------------------ Convolutional Layer 3 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=16, strides=1, filters=256, padding='SAME',
                             activation=None, use_bias=True, name='conv3', seed=self.seed, collections=['train_full'],
                             dilation_rate=4)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm3')

            # Activation
            with tf.variable_scope('relu3') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout3')

            ###############################################################
            # ------------------ Convolutional Layer 4 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=16, strides=1, filters=256, padding='SAME',
                             activation=None, use_bias=True, name='conv4', seed=self.seed, collections=['train_full'],
                             dilation_rate=4)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm4')

            # Activation
            with tf.variable_scope('relu4') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout4')

            ###############################################################
            # ------------------ Convolutional Layer 5 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=16, strides=1, filters=256, padding='SAME',
                             activation=None, use_bias=True, name='conv5', seed=self.seed, collections=['train_full'],
                             dilation_rate=4)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm5')

            # Activation
            with tf.variable_scope('relu5') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout5')

            ###############################################################
            # ------------------ Convolutional Layer 6 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=8, strides=1, filters=128, padding='SAME',
                             activation=None, use_bias=True, name='conv6', seed=self.seed, collections=['train_full'],
                             dilation_rate=4)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm6')

            # Activation
            with tf.variable_scope('relu6') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Max Pool
            net = max_pool_layer(input_layer=net, pool_size=2, strides=2, padding='SAME', name='maxpool6')

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout6')

            ###############################################################
            # ------------------ Convolutional Layer 7 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=8, strides=1, filters=128, padding='SAME',
                             activation=None, use_bias=True, name='conv7', seed=self.seed, collections=['train_full'],
                             dilation_rate=6)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm7')

            # Activation
            with tf.variable_scope('relu7') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout7')

            ###############################################################
            # ------------------ Convolutional Layer 8 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=8, strides=1, filters=128, padding='SAME',
                             activation=None, use_bias=True, name='conv8', seed=self.seed, collections=['train_full'],
                             dilation_rate=6)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm8')

            # Activation
            with tf.variable_scope('relu8') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout8')

            ###############################################################
            # ------------------ Convolutional Layer 9 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=8, strides=1, filters=128, padding='SAME',
                             activation=None, use_bias=True, name='conv9', seed=self.seed, collections=['train_full'],
                             dilation_rate=6)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm9')

            # Activation
            with tf.variable_scope('relu9') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout9')

            ###############################################################
            # ------------------ Convolutional Layer 10 ----------------- #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=8, strides=1, filters=128, padding='SAME',
                             activation=None, use_bias=True, name='conv10', seed=self.seed,
                             collections=['train_full'], dilation_rate=6)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm10')

            # Activation
            with tf.variable_scope('relu10') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout10')

            ###############################################################
            # ------------------ Convolutional Layer 11 ----------------- #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=8, strides=1, filters=128, padding='SAME',
                             activation=None, use_bias=True, name='conv11', seed=self.seed,
                             collections=['train_full'], dilation_rate=8)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm11')

            # Activation
            with tf.variable_scope('relu11') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Max Pool
            net = max_pool_layer(input_layer=net, pool_size=2, strides=2, padding='SAME', name='maxpool11')

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout11')

            ###############################################################
            # ------------------ Convolutional Layer 12 ----------------- #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=8, strides=1, filters=64, padding='SAME',
                             activation=None, use_bias=True, name='conv12', seed=self.seed,
                             collections=['train_full'], dilation_rate=8)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm12')

            # Activation
            with tf.variable_scope('relu12') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout12')

            ###############################################################
            # ------------------ Convolutional Layer 13 ----------------- #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=net, kernel_size=8, strides=1, filters=64, padding='SAME',
                             activation=None, use_bias=True, name='conv13', seed=self.seed,
                             collections=['train_full'], dilation_rate=8)

            # Batch Norm
            net = batch_norm_layer(input_layer=net, training=is_training, name='batchnorm13')

            # Activation
            with tf.variable_scope('relu13') as scope:
                net = tf.nn.relu(net, name=scope.name)

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training, name='dropout13')

            ###############################################################
            # -------------- Global Average Pooling Layer --------------- #
            ###############################################################

            # Global average pooling layer
            with tf.variable_scope('GAP'):

                # Reduce mean along dimension 1
                gap = tf.reduce_mean(input_tensor=net, axis=1)

            ###############################################################
            # ------------------------- Softmax ------------------------- #
            ###############################################################

            # Activation
            logits = fc_layer(input_layer=gap, neurons=self.classes, activation=None, use_bias=False, name='logits',
                              seed=self.seed, collections=['train_full'])

        return logits, net

    def create_placeholders(self):
        """Creates place holders: x, and y."""
        with tf.variable_scope('x') as scope:
            x = tf.placeholder(dtype=tf.float32, shape=[None, self.length, self.channels], name=scope.name)

        with tf.variable_scope('y') as scope:
            y = tf.placeholder(dtype=tf.float32, shape=[None, self.classes], name=scope.name)

        return x, y

    @staticmethod
    def compute_accuracy(logits, labels):
        """Computes the model accuracy for set of logits (predicted) and labels (true)."""
        with tf.variable_scope('accuracy'):
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), 'float'))

    @staticmethod
    def get_var_list():
        """Get list of parameters trainable parameters."""
        return tf.trainable_variables()
