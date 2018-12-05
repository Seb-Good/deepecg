"""
state.py
--------
This module includes a model state class.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import time
import numpy as np
import tensorflow as tf
from datetime import datetime


class State(object):

    def __init__(self, sess, graph, save_path, learning_rate, batch_size, num_gpus):

        # Set input parameters
        self.sess = sess
        self.graph = graph
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        # Set attributes
        self.train_loss = None
        self.val_loss = None
        self.train_accuracy = None
        self.val_accuracy = None
        self.time = time.time()
        self.global_step = self._get_global_step()
        self.datetime = str(datetime.utcnow())
        self.num_train_batches = self._get_num_train_batches()
        self.num_val_batches = self._get_num_val_batches()
        self.train_steps_per_epoch = int(np.ceil(self.num_train_batches / self.num_gpus))
        self.val_steps_per_epoch = int(np.ceil(self.num_val_batches / self.num_gpus))
        self.epoch = self._get_epoch()

        # Compute training and validation metrics
        self._compute_metrics()

    def _compute_metrics(self):

        # Training metrics
        self.train_loss, self.train_accuracy = self._compute_train_metrics()

        # Validation metrics
        self.val_loss, self.val_accuracy = self._compute_val_metrics()

    def _get_num_train_batches(self):
        """Number of batches for training Dataset."""
        return self.graph.generator_train.num_batches.eval(feed_dict={self.graph.batch_size: self.batch_size})

    def _get_num_val_batches(self):
        """Number of batches for validation Dataset."""
        return self.graph.generator_val.num_batches.eval(feed_dict={self.graph.batch_size: self.batch_size})

    def _get_global_step(self):
        return tf.train.global_step(self.sess, self.graph.global_step)

    def _get_epoch(self):
        return int(self.global_step / self.train_steps_per_epoch)

    def _compute_train_metrics(self):
        """Get training metrics."""
        if self.epoch > 0:
            # If metrics have been computed during a training epoch
            metrics_op = {key: val[0] for key, val in self.graph.metrics.items()}
            metrics = self.sess.run(metrics_op)

            return metrics['loss'], metrics['accuracy']

        else:
            # Get train handle
            handle_train = self.sess.run(self.graph.generator_train.iterator.string_handle())

            # Initialize train iterator
            self.sess.run(fetches=[self.graph.generator_train.iterator.initializer],
                          feed_dict={self.graph.batch_size: self.batch_size})

            # Initialize metrics
            self.sess.run(fetches=[self.graph.init_metrics_op])

            # Loop through train batches
            for batch in range(self.train_steps_per_epoch):

                # Run metric update operation
                self.sess.run(fetches=[self.graph.update_metrics_op],
                              feed_dict={self.graph.batch_size: self.batch_size, self.graph.is_training: True,
                                         self.graph.mode_handle: handle_train})

            # Get metrics
            metrics_op = {key: val[0] for key, val in self.graph.metrics.items()}
            metrics = self.sess.run(metrics_op)

            return metrics['loss'], metrics['accuracy']

    def _compute_val_metrics(self):
        """Get validation metrics."""
        # Get val handle
        handle_val = self.sess.run(self.graph.generator_val.iterator.string_handle())

        # Initialize val iterator
        self.sess.run(fetches=[self.graph.generator_val.iterator.initializer],
                      feed_dict={self.graph.batch_size: self.batch_size})

        # Initialize metrics
        self.sess.run(fetches=[self.graph.init_metrics_op])

        # Loop through val batches
        for batch in range(self.val_steps_per_epoch):

            # Run metric update operation
            self.sess.run(fetches=[self.graph.update_metrics_op],
                          feed_dict={self.graph.batch_size: self.batch_size, self.graph.is_training: False,
                                     self.graph.mode_handle: handle_val})

        # Get metrics
        metrics_op = {key: val[0] for key, val in self.graph.metrics.items()}
        metrics = self.sess.run(metrics_op)

        return metrics['loss'], metrics['accuracy']
