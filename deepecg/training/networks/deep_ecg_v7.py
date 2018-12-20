"""
deep_ecg_v7.py
--------------
This module provides a class and methods for building a convolutional neural network with tensorflow.
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
from deepecg.training.networks.layers import fc_layer, conv_layer, max_pool_layer, avg_pool_layer, \
                                             batch_norm_layer, dropout_layer, print_output_shape


class DeepECGV7(object):

    """
    Build the forward propagation computational graph for an Inception-V4 and ResNet inspired deep neural network.

    Szegedy, C., Ioffe, S., Vanhoucke, V. (2016) Inception-v4, inception-resnet and the impact of residual
    connections on learning (2016). arXiv:1602.07261
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

            # Set variables
            kernel_size = 2
            conv_filts = 128
            res_filts = 32
            skip_filts = 128
            skips = list()

            # Print shape
            print_output_shape(layer_name='input', net=input_layer, print_shape=print_shape)

            """Block Series 1"""
            # --- Layer 1 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_1'

            # Set layer scope
            with tf.variable_scope(layer_name):

                # Convolution
                res = conv_layer(input_layer=input_layer, kernel_size=kernel_size, strides=1, dilation_rate=1,
                                 filters=res_filts, padding='SAME', activation=None, use_bias=False,
                                 name=layer_name + '_conv', seed=self.seed)

            # Print shape
            print_output_shape(layer_name=layer_name, net=res, print_shape=print_shape)

            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_2'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=2)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_3'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=4)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_4'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=8)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_5'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=16)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_6'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=32)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_7'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=64)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_8'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=128)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_9'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=256)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #
            #
            # # Set name
            # layer_name = 'layer_10'
            #
            # # Compute block
            # res, skip = self._residual_block(input_layer=res, kernel_size=kernel_size, layer_name=layer_name,
            #                                  conv_filts=conv_filts, res_filts=res_filts, skip_filts=skip_filts,
            #                                  dilation_rate=512)
            #
            # # Collect skip
            # skips.append(skip)
            #
            # # Print shape
            # print_output_shape(layer_name=layer_name + '_res', net=res, print_shape=print_shape)
            # print_output_shape(layer_name=layer_name + '_skip', net=skip, print_shape=print_shape)
            #
            # # Add all skips to res output
            # output = tf.zeros(shape=tf.shape(skip))
            # for skip in skips[0:-1]:
            #     output = tf.add(x=output, y=skip)
            #
            # # Print shape
            # print_output_shape(layer_name='output_skip_addition', net=output, print_shape=print_shape)
            #
            # Activation
            with tf.variable_scope('relu1') as scope:
                output = tf.nn.relu(res, name=scope.name)

            # Convolution
            output = conv_layer(input_layer=output, kernel_size=1, strides=1, dilation_rate=1,
                                filters=256, padding='SAME', activation=None, use_bias=False,
                                name='conv1', seed=self.seed)

            # Print shape
            print_output_shape(layer_name='output_conv1', net=output, print_shape=print_shape)

            # Activation
            with tf.variable_scope('relu2') as scope:
                output = tf.nn.relu(output, name=scope.name)

            # Convolution
            output = conv_layer(input_layer=output, kernel_size=1, strides=1, dilation_rate=1,
                                filters=512, padding='SAME', activation=None, use_bias=False,
                                name='conv2', seed=self.seed)

            # Print shape
            print_output_shape(layer_name='output_conv2', net=output, print_shape=print_shape)

            """Network Output"""
            # --- Global Average Pooling Layer ----------------------------------------------------------------------- #

            # Set name
            layer_name = 'gap'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Reduce mean along dimension 1
                gap = tf.reduce_mean(input_tensor=output, axis=1)

            # Print shape
            print_output_shape(layer_name=layer_name, net=gap, print_shape=print_shape)

            # --- Softmax Layer -------------------------------------------------------------------------------------- #

            # Set name
            layer_name = 'logits'

            # Softmax activation
            logits = fc_layer(input_layer=gap, neurons=self.classes, activation=None, use_bias=False,
                              name=layer_name, seed=self.seed)

            # Print shape
            print_output_shape(layer_name=layer_name, net=logits, print_shape=print_shape)

            # Compute Class Activation Maps
            # cams = self._get_cams(net=res, is_training=is_training)

        return logits, [], []

    def _residual_block(self, input_layer, kernel_size, layer_name, conv_filts, res_filts, skip_filts, dilation_rate):
        """Wavenet residual block."""

        # Set layer scope
        with tf.variable_scope(layer_name):

            # Set identity
            identity = tf.identity(input=input_layer, name=layer_name + '_identity')

            # Convolution
            tanh = conv_layer(input_layer=input_layer, kernel_size=kernel_size, strides=1,
                              dilation_rate=dilation_rate, filters=conv_filts, padding='SAME', activation=None,
                              use_bias=False, name=layer_name + '_conv_tanh', seed=self.seed)

            # Tanh activation
            tanh = tf.nn.tanh(tanh, name='tanh')

            # Convolution
            sigmoid = conv_layer(input_layer=input_layer, kernel_size=kernel_size, strides=1,
                                 dilation_rate=dilation_rate, filters=conv_filts, padding='SAME', activation=None,
                                 use_bias=False, name=layer_name + '_conv_sigmoid', seed=self.seed)

            # Sigmoid activation
            sigmoid = tf.nn.sigmoid(sigmoid, name='sigmoid')

            # Combine activations
            activation = tf.multiply(tanh, sigmoid, name='gate_activation')

            # Residual Convolution
            res_out = conv_layer(input_layer=activation, kernel_size=1, strides=1, dilation_rate=dilation_rate,
                                 filters=res_filts, padding='SAME', activation=None, use_bias=False,
                                 name=layer_name + '_conv_res', seed=self.seed)

            # Add identity
            res_out = tf.add(res_out, identity, name=layer_name + '_add_identity')

            # Skip Convolution
            skip_out = conv_layer(input_layer=activation, kernel_size=1, strides=1, dilation_rate=dilation_rate,
                                  filters=skip_filts, padding='SAME', activation=None, use_bias=False,
                                  name=layer_name + '_conv_skip', seed=self.seed)

            return res_out, skip_out

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
                             batch_size=batch_size, prefetch_buffer=1500, seed=0, num_parallel_calls=32)

    @staticmethod
    def compute_accuracy(logits, labels):
        """Computes the model accuracy for set of logits and labels."""
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

