"""
state.py
--------
This module provides a model state class.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import f1_score


class State(object):

    def __init__(self, sess, graph, save_path, learning_rate, num_mini_batches,
                 mini_batches_train, mini_batches_val):

        # Set input parameters
        self.sess = sess
        self.graph = graph
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.num_mini_batches = num_mini_batches
        self.mini_batches_train = mini_batches_train
        self.mini_batches_val = mini_batches_val

        # Set attributes
        self.time = time.time()
        self.datetime = str(datetime.utcnow())
        self.training_time = None
        self.epoch = None
        self.global_step = None
        self.train_loss = None
        self.val_loss = None
        self.train_accuracy = None
        self.val_accuracy = None
        self.train_f1 = None
        self.val_f1 = None

        # Set model state
        self._get_model_state()

    def _get_model_state(self):

        # Global step
        self.global_step = self._get_global_step()

        # Epoch
        self.epoch = self._get_epoch()

        # Training metrics
        self.train_loss, self.train_accuracy, self.train_f1 = self._get_train_metrics()

        # Validation metrics
        self.val_loss, self.val_accuracy, self.val_f1 = self._get_val_metrics()

    def _get_global_step(self):
        return tf.train.global_step(self.sess, self.graph.global_step)

    def _get_epoch(self):
        return self.global_step / self.num_mini_batches

    def _get_train_metrics(self):
        return self._compute_metrics(mini_batches=self.mini_batches_train)

    def _get_val_metrics(self):
        return self._compute_metrics(mini_batches=self.mini_batches_val)

    def _compute_batch_loss(self, x, y):
        return self.graph.loss.eval(feed_dict={self.graph.x: x, self.graph.y: y, self.graph.is_training: False})

    def _compute_batch_accuracy(self, x, y):
        return self.graph.accuracy.eval(feed_dict={self.graph.x: x, self.graph.y: y, self.graph.is_training: False})

    def _compute_batch_f1(self, x, y):

        # Get logits
        logits = self.graph.logits.eval(feed_dict={self.graph.x: x, self.graph.y: y, self.graph.is_training: False})

        # Compute f1 score
        f1_scores = f1_score(np.argmax(y, axis=1), np.argmax(logits, axis=1), average=None)

        return np.mean(f1_scores)

    def _compute_metrics(self, mini_batches):

        # Set starting loss, accuracy, and f1
        loss = 0
        accuracy = 0
        f1 = 0

        # Run mini_batch loop
        for mini_batch in mini_batches:

            # Select mini_batch x and y
            (mini_batch_x, mini_batch_y) = mini_batch

            # Sum losses, accuracies, and f1
            loss += self._compute_batch_loss(x=mini_batch_x, y=mini_batch_y)
            accuracy += self._compute_batch_accuracy(x=mini_batch_x, y=mini_batch_y)
            f1 += self._compute_batch_f1(x=mini_batch_x, y=mini_batch_y)

        # Get average loss, accuracy, and f1 across all mini-batches
        loss /= len(mini_batches)
        accuracy /= len(mini_batches)
        f1 /= len(mini_batches)

        return loss, accuracy, f1
