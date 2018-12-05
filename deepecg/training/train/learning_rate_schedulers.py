"""
learning_rate_schedulers.py
---------------------------
This module provide classes and functions for managing learning rate schedules.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np


class AnnealingRestartScheduler(object):

    """
    Cyclical learning rate decay with warm restarts and cosine annealing.
    Reference: https://arxiv.org/pdf/1608.03983.pdf
    """

    def __init__(self, lr_min, lr_max, steps_per_epoch, lr_max_decay, epochs_per_cycle, cycle_length_factor):

        # Set parameters
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.steps_per_epoch = steps_per_epoch
        self.lr_max_decay = lr_max_decay
        self.epochs_per_cycle = epochs_per_cycle
        self.cycle_length_factor = cycle_length_factor

        # Set attributes
        self.lr = self.lr_max
        self.steps_since_restart = 0
        self.next_restart = self.epochs_per_cycle

    def on_batch_end_update(self):
        """Update at the end of each mini-batch."""
        # Update steps since restart
        self.steps_since_restart += 1

        # Update learning rate
        self.lr = self._compute_cosine_learning_rate()

    def on_epoch_end_update(self, epoch):
        """Check for end of current cycle, apply restarts when necessary."""
        if epoch + 1 == self.next_restart:
            self.steps_since_restart = 0
            self.epochs_per_cycle = np.ceil(self.epochs_per_cycle * self.cycle_length_factor)
            self.next_restart += self.epochs_per_cycle
            self.lr_max *= self.lr_max_decay

    def _compute_cosine_learning_rate(self):
        """Compute cosine learning rate decay."""
        # Compute the cycle completion factor
        fraction_complete = self._compute_fraction_complete()

        # Compute learning rate
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(fraction_complete * np.pi))

    def _compute_fraction_complete(self):
        """Compute the fraction of the cycle that is completed."""
        return self.steps_since_restart / (self.steps_per_epoch * self.epochs_per_cycle)


def exponential_step_decay(decay_epochs, decay_rate, initial_learning_rate, epoch):
    """Compute exponential learning rate step decay."""
    return initial_learning_rate * np.power(decay_rate, np.floor((epoch / decay_epochs)))
