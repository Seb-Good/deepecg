"""
monitor.py
----------
This module provides a class for monitoring the model state during training.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import copy

# Local imports
from deepecg.training.train.disc.state import State


class Monitor(object):

    def __init__(self, sess, graph, learning_rate, batch_size, save_path, num_gpus):

        # Set parameters
        self.sess = sess
        self.graph = graph
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_path = save_path
        self.num_gpus = num_gpus

        # Set attributes
        self.current_state = self._get_model_state()
        self.best_state = copy.copy(self.current_state)

    def _get_model_state(self):
        """Get model state at current learning step."""
        return State(sess=self.sess, graph=self.graph, save_path=self.save_path, learning_rate=self.learning_rate,
                     batch_size=self.batch_size, num_gpus=self.num_gpus)

    def update_model_state(self, learning_rate):
        """Update model state and check for improvements."""
        # Update learning rate
        self.learning_rate = learning_rate

        # Set current model state
        self.current_state = self._get_model_state()

        # Check for improvement
        self._improvement_check()

        # Save checkpoint if validation accuracy improvement
        self._save_checkpoint()

    def _improvement_check(self):
        """Check for improvement in validation accuracy."""
        if self.current_state.val_f1 > self.best_state.val_f1:
            self.best_state = copy.copy(self.current_state)

    def _save_checkpoint(self):
        """Check for improvement in validation accuracy."""
        if self.current_state.val_accuracy == self.best_state.val_accuracy:
            self.graph.saver.save(sess=self.sess, save_path=os.path.join(self.save_path, 'checkpoints', 'model'),
                                  global_step=self.graph.global_step)

    def end_monitoring(self):
        """Save checkpoint."""
        self.graph.saver.save(sess=self.sess, save_path=os.path.join(self.save_path, 'checkpoints', 'model'),
                              global_step=self.graph.global_step)
