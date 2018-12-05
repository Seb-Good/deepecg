"""
monitor.py
----------
This module provides a class to monitor the model state during training.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import copy
import numpy as np

# Local imports
from train.state import State
from train.mini_batches import mini_batch_generator


class Monitor(object):

    def __init__(self, sess, graph, x_train, y_train, x_val, y_val, learning_rate,
                 mini_batch_size, save_path, save_epoch, seed, gpu_count):

        # Set parameters
        self.sess = sess
        self.graph = graph
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.save_path = save_path
        self.save_epoch = save_epoch
        self.seed = seed
        self.gpu_count = gpu_count

        # Set attributes
        self.num_mini_batches = int(np.ceil(self.x_train.shape[0] / self.mini_batch_size))
        self.mini_batches_train = None
        self.mini_batches_val = None
        self.current_state = None
        self.best_state = None

        # Get training and validation mini-batches
        self._get_mini_batches()

        # Set previous and current model state
        self.current_state = self._get_state()
        self.best_state = copy.copy(self.current_state)

    def _get_mini_batches(self):
        """Get training and validation mini-batches for evaluation metrics."""
        self.mini_batches_train = mini_batch_generator(x=self.x_train, y=self.y_train,
                                                       mini_batch_size=self.mini_batch_size, seed=self.seed,
                                                       gpu_count=self.gpu_count)
        self.mini_batches_val = mini_batch_generator(x=self.x_val, y=self.y_val, mini_batch_size=self.mini_batch_size,
                                                     seed=self.seed, gpu_count=self.gpu_count)

    def _get_state(self):
        """Get current model state."""
        return State(sess=self.sess, graph=self.graph, save_path=self.save_path,
                     learning_rate=self.learning_rate, num_mini_batches=self.num_mini_batches,
                     mini_batches_train=self.mini_batches_train, mini_batches_val=self.mini_batches_val)

    def update_state(self, learning_rate):
        """Update model state and log improvements."""
        # Update learning rate
        self.learning_rate = learning_rate

        # Set current model state
        self.current_state = self._get_state()

        # Check for improvement
        self._improvement_check()

        # Save checkpoint if improvement listed
        self._save_checkpoint()

    def _improvement_check(self):
        """Check for improvement in F1 score on the validation dataset."""
        if self.current_state.val_f1 > self.best_state.val_f1:

            # Set best state
            self.best_state = copy.copy(self.current_state)

    def _save_checkpoint(self):
        """Save checkpoint if improvement in F1 score is observed or if scheduled."""
        # Check for improvement in validation accuracy
        if self.current_state.val_f1 == self.best_state.val_f1 or self.current_state.epoch % self.save_epoch == 0:

            # Save checkpoint
            self.graph.saver.save(sess=self.sess, save_path=os.path.join(self.save_path, 'checkpoints', 'model'),
                                  global_step=self.graph.global_step)

    def end_monitoring(self):
        """Save checkpoint when monitoring is ended."""
        self.graph.saver.save(sess=self.sess, save_path=os.path.join(self.save_path, 'checkpoints', 'model'),
                              global_step=self.graph.global_step)
