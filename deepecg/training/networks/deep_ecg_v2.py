"""
deep_ecg_v2.py
--------------
This module provides a class and methods for building a convolutional neural network with tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import tensorflow as tf

# Local imports
from deepecg.training.train.disc.data_generator import DataGenerator
from deepecg.training.networks.layers import fc_layer, conv_layer, max_pool_layer, avg_pool_layer, \
                                             batch_norm_layer, dropout_layer, print_output_shape


class DeepECGV2(object):

    """
    Build the forward propagation computational graph for an Inception-V4 inspired deep neural network.

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

            """Network Stem"""
            # --- Layer 1 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_1'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Convolution
                net = conv_layer(input_layer=input_layer, kernel_size=3, strides=2, dilation_rate=1,
                                 filters=32, padding='VALID', activation=tf.nn.relu, use_bias=True,
                                 name=layer_name + '_conv_ks3_dr1', seed=self.seed)

                # Batch Norm
                net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Dropout
                net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                    name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_2'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Convolution
                net = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                 filters=32, padding='VALID', activation=tf.nn.relu, use_bias=True,
                                 name=layer_name + '_conv_ks3_dr1', seed=self.seed)

                # Batch Norm
                net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Dropout
                net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                    name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 3 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_3'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Convolution
                net = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                 filters=64, padding='VALID', activation=tf.nn.relu, use_bias=True,
                                 name=layer_name + '_conv_ks3_dr1', seed=self.seed)

                # Batch Norm
                net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Dropout
                net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                    name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 4 (Mixed - Convolution) ---------------------------------------------------------------------- #

            # Set name
            layer_name = 'layer_4'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Max pool
                    branch_0 = max_pool_layer(input_layer=net, pool_size=3, strides=2, padding='VALID',
                                              name=layer_name + '_branch_0_a_maxpool_ps3')

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=3, strides=2, dilation_rate=1,
                                          filters=96, padding='VALID', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks3_dr1', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1], axis=2)

                # Batch Norm
                net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Dropout
                net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                    name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 5 (Mixed - Convolution) ---------------------------------------------------------------------- #

            # Set name
            layer_name = 'layer_5'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=1, strides=1, dilation_rate=1,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks1_dr1', seed=self.seed)
                    branch_0 = conv_layer(input_layer=branch_0, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=96, padding='VALID', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_b_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=1, strides=1, dilation_rate=1,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks1_dr1', seed=self.seed)
                    branch_1 = conv_layer(input_layer=branch_1, kernel_size=7, strides=1, dilation_rate=2,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_b_conv_ks7_dr2', seed=self.seed)
                    branch_1 = conv_layer(input_layer=branch_1, kernel_size=7, strides=1, dilation_rate=4,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_c_conv_ks7_dr4', seed=self.seed)
                    branch_1 = conv_layer(input_layer=branch_1, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=96, padding='VALID', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_d_conv_ks3_dr1', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1], axis=2)

                # Batch Norm
                net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Dropout
                net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                    name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 6 (Mixed - Convolution) ---------------------------------------------------------------------- #

            # Set name
            layer_name = 'layer_6'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=2, dilation_rate=1,
                                          filters=192, padding='VALID', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Max pool
                    branch_1 = max_pool_layer(input_layer=net, pool_size=3, strides=2, padding='VALID',
                                              name=layer_name + '_branch_0_a_maxpool_ps3')

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1], axis=2)

                # Batch Norm
                net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Dropout
                net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                    name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            """Inception-A (V4)"""
            # Set number of Inception layers and previous layer
            num_layers = 2
            previous_layer = 6

            for layer_id in np.arange(num_layers) + (previous_layer + 1):

                # Set name
                layer_name = 'layer_{}'.format(layer_id)

                # Set layer scope
                with tf.variable_scope(layer_name):
                    # Inception-A (V4)
                    net = self._inception_a(inputs=net, layer_name=layer_name)

                    # Batch Norm
                    net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Print shape
                print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            """Reduction-A"""

            """Inception-B"""

            """Reduction-B"""

            """Inception-C"""

            """Network Output"""
            # --- Global Average Pooling Layer ----------------------------------------------------------------------- #

            # Set layer scope
            with tf.variable_scope('GAP'):
                # Reduce mean along dimension 1
                gap = tf.reduce_mean(input_tensor=net, axis=1)

            # --- Softmax Layer -------------------------------------------------------------------------------------- #

            # Softmax activation
            logits = fc_layer(input_layer=gap, neurons=self.classes, activation=None, use_bias=False,
                              name='logits', seed=self.seed)

        return logits, net

    def _inception_a(self, inputs, layer_name):
        """Builds Inception-A block (V4)."""
        # Branch 0
        with tf.variable_scope('branch_0'):
            # Convolution
            branch_0 = conv_layer(input_layer=inputs, kernel_size=1, strides=1, dilation_rate=1,
                                  filters=96, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                  name=layer_name + '_branch_0_a_conv_ks1_dr1', seed=self.seed)

        # Branch 1
        with tf.variable_scope('branch_1'):
            # Convolution
            branch_1 = conv_layer(input_layer=inputs, kernel_size=1, strides=1, dilation_rate=1,
                                  filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                  name=layer_name + '_branch_1_a_conv_ks1_dr1', seed=self.seed)
            branch_1 = conv_layer(input_layer=branch_1, kernel_size=3, strides=1, dilation_rate=1,
                                  filters=96, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                  name=layer_name + '_branch_1_b_conv_ks3_dr1', seed=self.seed)

        # Branch 2
        with tf.variable_scope('branch_2'):
            # Convolution
            branch_2 = conv_layer(input_layer=inputs, kernel_size=1, strides=1, dilation_rate=1,
                                  filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                  name=layer_name + '_branch_2_a_conv_ks1_dr1', seed=self.seed)
            branch_2 = conv_layer(input_layer=branch_2, kernel_size=3, strides=1, dilation_rate=3,
                                  filters=96, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                  name=layer_name + '_branch_2_b_conv_ks3_dr1', seed=self.seed)
            branch_2 = conv_layer(input_layer=branch_2, kernel_size=3, strides=1, dilation_rate=6,
                                  filters=96, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                  name=layer_name + '_branch_2_c_conv_ks3_dr1', seed=self.seed)

        # Branch 3
        with tf.variable_scope('branch_3'):
            # Convolution
            branch_3 = conv_layer(input_layer=inputs, kernel_size=3, strides=1, dilation_rate=1,
                                  filters=96, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                  name=layer_name + '_branch_3_a_conv_ks3_dr1', seed=self.seed)
            branch_3 = conv_layer(input_layer=branch_3, kernel_size=1, strides=1, dilation_rate=1,
                                  filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                  name=layer_name + '_branch_3_b_conv_ks1_dr1', seed=self.seed)

        return tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=2)

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
        """Computes the model accuracy for set of logits (predicted) and labels (true)."""
        with tf.variable_scope('accuracy'):
            return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, tf.int64)), 'float'))
