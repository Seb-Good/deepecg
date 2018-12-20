"""
deep_ecg_v3.py
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


class DeepECGV3(object):

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

            """Block Series 1"""
            # --- Layer 1 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_1'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Convolution
                net = conv_layer(input_layer=input_layer, kernel_size=12, strides=1, dilation_rate=1,
                                 filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                 name=layer_name + '_conv_ks3_dr1', seed=self.seed)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Max pool
                net = max_pool_layer(input_layer=net, pool_size=3, strides=2, padding='SAME',
                                     name=layer_name + '_maxpool_ps3')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 2 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_2'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Convolution
                net = conv_layer(input_layer=net, kernel_size=12, strides=1, dilation_rate=1,
                                 filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                 name=layer_name + '_conv_ks3_dr1', seed=self.seed)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 3 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_3'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Convolution
                net = conv_layer(input_layer=net, kernel_size=12, strides=1, dilation_rate=1,
                                 filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                 name=layer_name + '_conv_ks3_dr1', seed=self.seed)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

                # Max pool
                net = max_pool_layer(input_layer=net, pool_size=3, strides=2, padding='SAME',
                                     name=layer_name + '_maxpool_ps3')

                # Dropout
                net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                    name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            """Block Series 2"""
            # --- Layer 4 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_4'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=6,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=6,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 5 (Convolution) ------------------------------------------------------------------------------ #

            # Set identity
            identity = tf.identity(input=net, name='identity_5_6')

            # Set name
            layer_name = 'layer_5'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=6,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=6,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 6 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_6'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=6,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=6,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Add identity
            net = tf.add(net, identity, name='add_identity_5_6')

            # Max pool
            net = max_pool_layer(input_layer=net, pool_size=3, strides=2, padding='SAME',
                                 name=layer_name + '_maxpool_ps3')

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            """Block Series 3"""
            # --- Layer 7 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_7'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Branch 4
                with tf.variable_scope('branch_4'):
                    # Convolution
                    branch_4 = conv_layer(input_layer=net, kernel_size=1, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_4_a_conv_ks1_dr1', seed=self.seed)

                # Branch 5
                with tf.variable_scope('branch_5'):
                    # Convolution
                    branch_5 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_a_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_b_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_c_conv_ks9_dr2', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3, branch_4, branch_5], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 8 (Convolution) ------------------------------------------------------------------------------ #

            # Set identity
            identity = tf.identity(input=net, name='identity_8_9')

            # Set name
            layer_name = 'layer_8'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Branch 4
                with tf.variable_scope('branch_4'):
                    # Convolution
                    branch_4 = conv_layer(input_layer=net, kernel_size=1, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_4_a_conv_ks1_dr1', seed=self.seed)

                # Branch 5
                with tf.variable_scope('branch_5'):
                    # Convolution
                    branch_5 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_a_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_b_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_c_conv_ks9_dr2', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3, branch_4, branch_5], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 9 (Convolution) ------------------------------------------------------------------------------ #

            # Set name
            layer_name = 'layer_9'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Branch 4
                with tf.variable_scope('branch_4'):
                    # Convolution
                    branch_4 = conv_layer(input_layer=net, kernel_size=1, strides=1, dilation_rate=1,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_4_a_conv_ks1_dr1', seed=self.seed)

                # Branch 5
                with tf.variable_scope('branch_5'):
                    # Convolution
                    branch_5 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_a_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_b_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=16, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_c_conv_ks9_dr2', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3, branch_4, branch_5], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Add identity
            net = tf.add(net, identity, name='add_identity_8_9')

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            """Block Series 4"""
            # --- Layer 10 (Convolution) ----------------------------------------------------------------------------- #

            # Set name
            layer_name = 'layer_10'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Branch 4
                with tf.variable_scope('branch_4'):
                    # Convolution
                    branch_4 = conv_layer(input_layer=net, kernel_size=1, strides=1, dilation_rate=1,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_4_a_conv_ks1_dr1', seed=self.seed)

                # Branch 5
                with tf.variable_scope('branch_5'):
                    # Convolution
                    branch_5 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=256, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_a_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_b_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_c_conv_ks9_dr2', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3, branch_4, branch_5], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 11 (Convolution) ----------------------------------------------------------------------------- #

            # Set identity
            identity = tf.identity(input=net, name='identity_11_12')

            # Set name
            layer_name = 'layer_11'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Branch 4
                with tf.variable_scope('branch_4'):
                    # Convolution
                    branch_4 = conv_layer(input_layer=net, kernel_size=1, strides=1, dilation_rate=1,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_4_a_conv_ks1_dr1', seed=self.seed)

                # Branch 5
                with tf.variable_scope('branch_5'):
                    # Convolution
                    branch_5 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=256, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_a_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_b_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_c_conv_ks9_dr2', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3, branch_4, branch_5], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            # --- Layer 12 (Convolution) ----------------------------------------------------------------------------- #

            # Set name
            layer_name = 'layer_12'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Branch 0
                with tf.variable_scope('branch_0'):
                    # Convolution
                    branch_0 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=1,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_0_a_conv_ks3_dr1', seed=self.seed)

                # Branch 1
                with tf.variable_scope('branch_1'):
                    # Convolution
                    branch_1 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=1,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_1_a_conv_ks9_dr1', seed=self.seed)

                # Branch 2
                with tf.variable_scope('branch_2'):
                    # Convolution
                    branch_2 = conv_layer(input_layer=net, kernel_size=3, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_2_a_conv_ks3_dr6', seed=self.seed)

                # Branch 3
                with tf.variable_scope('branch_3'):
                    # Convolution
                    branch_3 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_3_a_conv_ks9_dr6', seed=self.seed)

                # Branch 4
                with tf.variable_scope('branch_4'):
                    # Convolution
                    branch_4 = conv_layer(input_layer=net, kernel_size=1, strides=1, dilation_rate=1,
                                          filters=32, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_4_a_conv_ks1_dr1', seed=self.seed)

                # Branch 5
                with tf.variable_scope('branch_5'):
                    # Convolution
                    branch_5 = conv_layer(input_layer=net, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=256, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_a_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=128, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_b_conv_ks9_dr2', seed=self.seed)
                    branch_5 = conv_layer(input_layer=branch_5, kernel_size=9, strides=1, dilation_rate=2,
                                          filters=64, padding='SAME', activation=tf.nn.relu, use_bias=True,
                                          name=layer_name + '_branch_5_c_conv_ks9_dr2', seed=self.seed)

                # Merge branches
                net = tf.concat(values=[branch_0, branch_1, branch_2, branch_3, branch_4, branch_5], axis=2)

                # Batch Norm
                # net = batch_norm_layer(input_layer=net, training=is_training, name=layer_name + '_batchnorm')

            # Add identity
            net = tf.add(net, identity, name='add_identity_11_12')

            # Dropout
            net = dropout_layer(input_layer=net, drop_rate=0.3, seed=self.seed, training=is_training,
                                name=layer_name + '_dropout')

            # Print shape
            print_output_shape(layer_name=layer_name, net=net, print_shape=print_shape)

            """Network Output"""
            # --- Global Average Pooling Layer ----------------------------------------------------------------------- #

            # Set name
            layer_name = 'gap'

            # Set layer scope
            with tf.variable_scope(layer_name):
                # Reduce mean along dimension 1
                gap = tf.reduce_mean(input_tensor=net, axis=1)

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

