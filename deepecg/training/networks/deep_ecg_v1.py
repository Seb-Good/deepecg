"""
deep_ecg_v1.py
--------------
This module provides a class and methods for building a 13 layer convolutional neural network with tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

# Local imports
from deepecg.training.train.disc.data_generator import DataGenerator
from deepecg.training.networks.layers import fc_layer, conv_layer, max_pool_layer, batch_norm_layer, dropout_layer


class DeepECGV1(object):

    """
    Build the forward propagation computational graph from a 13 layer convolutional neural network.

    Reference:
    Goodfellow, S. D., A. Goodwin, R. Greer, P. C. Laussen, M. Mazwi, and D. Eytan, Towards understanding ECG
    rhythm classification using convolutional neural networks and attention mappings, Proceedings of Machine
    Learning for Healthcare 2018 JMLR W&C Track Volume 85, Aug 17–18, 2018, Stanford, California, USA.
    """

    def __init__(self, length, channels, classes, seed=0):

        # Set input parameters
        self.length = length
        self.channels = channels
        self.classes = classes
        self.seed = seed

    def inference(self, input_layer, reuse, is_training, name, print_shape=True):
        """Forward propagation of computational graph."""
        # Check input layer dimensions
        assert input_layer.shape[1] == self.length
        assert input_layer.shape[2] == self.channels

        # Define a scope for reusing the variables
        with tf.variable_scope(name, reuse=reuse):

            ###############################################################
            # ------------------ Convolutional Layer 1 ------------------ #
            ###############################################################

            # Convolution
            net = conv_layer(input_layer=input_layer, kernel_size=24, strides=1, filters=320, padding='SAME',
                             activation=None, use_bias=True, name='conv1', seed=self.seed, dilation_rate=1)

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
                             activation=None, use_bias=True, name='conv2', seed=self.seed, dilation_rate=2)

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
                             activation=None, use_bias=True, name='conv3', seed=self.seed, dilation_rate=4)

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
                             activation=None, use_bias=True, name='conv4', seed=self.seed, dilation_rate=4)

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
                             activation=None, use_bias=True, name='conv5', seed=self.seed, dilation_rate=4)

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
                             activation=None, use_bias=True, name='conv6', seed=self.seed, dilation_rate=4)

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
                             activation=None, use_bias=True, name='conv7', seed=self.seed, dilation_rate=6)

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
                             activation=None, use_bias=True, name='conv8', seed=self.seed, dilation_rate=6)

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
                             activation=None, use_bias=True, name='conv9', seed=self.seed, dilation_rate=6)

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
                             activation=None, use_bias=True, name='conv10', seed=self.seed, dilation_rate=6)

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
                             activation=None, use_bias=True, name='conv11', seed=self.seed, dilation_rate=8)

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
                             activation=None, use_bias=True, name='conv12', seed=self.seed, dilation_rate=8)

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
                             activation=None, use_bias=True, name='conv13', seed=self.seed, dilation_rate=8)

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
            logits = fc_layer(input_layer=gap, neurons=self.classes, activation=None, use_bias=False,
                              name='logits', seed=self.seed)

            # Compute Class Activation Maps
            cams = self._get_cams(net=net, is_training=is_training)

        return logits, net, cams

    def _get_cams(self, net, is_training):
        """Collect class activation maps (CAMs)."""
        # Empty list for class activation maps
        cams = dict()

        # Compute class activation map
        if is_training is not None:
            for label in range(self.classes):
                cams[label] = self._compute_cam(net=net, label=label)

        return cams

    def _compute_cam(self, net, label):
        """Compute class activation map (CAM) for specified label."""
        # Compute logits weights
        weights = self._get_logit_weights(net=net, label=label)

        # Compute class activation map
        cam = tf.matmul(net, weights)

        return cam

    def _get_logit_weights(self, net, label):
        """Get logits weights for specified label."""
        # Get number of filters in the final output
        num_filters = int(net.shape[-1])

        with tf.variable_scope('logits', reuse=True):
            weights = tf.gather(tf.transpose(tf.get_variable('kernel')), label)
            weights = tf.reshape(weights, [-1, num_filters, 1])

        # Reshape weights
        weights = self._reshape_logit_weights(net=net, weights=weights)

        return weights

    @staticmethod
    def _reshape_logit_weights(net, weights):
        """Reshape logits shapes to batch size for multiplication with net output."""
        return tf.tile(input=weights, multiples=[tf.shape(net)[0], 1, 1])

    def create_placeholders(self):
        """Creates place holders: waveform and label."""
        with tf.variable_scope('waveform') as scope:
            waveform = tf.placeholder(dtype=tf.float32, shape=[None, self.length, self.channels], name=scope.name)

        with tf.variable_scope('label') as scope:
            label = tf.placeholder(dtype=tf.int32, shape=[None], name=scope.name)

        return waveform, label

    def create_generator(self, path, mode, batch_size):
        """Create data generator graph operation."""
        return DataGenerator(path=path, mode=mode, shape=[self.length, self.channels],
                             batch_size=batch_size, prefetch_buffer=200, seed=0, num_parallel_calls=32)

    @staticmethod
    def compute_accuracy(logits, labels):
        """Computes the model accuracy for set of logits (predicted) and labels (true)."""
        with tf.variable_scope('accuracy'):
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, tf.int64)), 'float'))

    def compute_f1(self, logits, labels):
        """Computes the model f1 score for set of logits and labels."""
        with tf.variable_scope('f1'):

            # Get prediction
            predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)

            # Get label
            labels = tf.cast(labels, tf.int32)

            return tf.py_func(func=self._compute_f1, inp=[predictions, labels], Tout=[tf.float64])

    @staticmethod
    def _compute_f1(predictions, labels):
        """Compute the mean f1 score."""
        return np.mean(f1_score(labels, predictions, labels=[0, 1, 2, 3], average=None)[0:3])
