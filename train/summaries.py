"""
summaries.py
------------
This module provides a class and methods for writing training and validation summaries.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import tensorflow as tf


class Summaries(object):

    """Writes training and validation summaries."""

    def __init__(self, sess, graph, path):

        # Set input parameters
        self.sess = sess
        self.graph = graph
        self.path = path

        # Set summary paths
        self.train_summary_path = os.path.join(self.path, 'training')
        self.val_summary_path = os.path.join(self.path, 'validation')

        # Initialize Tensorboard writers
        self.train_summary_writer = tf.summary.FileWriter(logdir=self.train_summary_path, graph=self.sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(logdir=self.val_summary_path)

    def write_train_summary_ops(self, summary, global_step):
        """Add training summary from graph summary operation."""
        self.train_summary_writer.add_summary(summary=summary, global_step=global_step)

    def write_val_summaries(self, monitor):
        """Write validation summary from monitor."""
        # Create value summary
        summary = tf.Summary()
        summary.value.add(tag='loss/loss', simple_value=monitor.current_state.val_loss)
        summary.value.add(tag='accuracy/accuracy', simple_value=monitor.current_state.val_accuracy)
        summary.value.add(tag='f1/f1', simple_value=monitor.current_state.val_f1)

        # Write validation summary
        self.val_summary_writer.add_summary(summary=summary, global_step=monitor.current_state.global_step)

        # Flush summary writer
        self.val_summary_writer.flush()

    def write_train_summaries(self, monitor):
        """Write training summary from monitor."""
        # Create value summary
        summary = tf.Summary()
        summary.value.add(tag='loss/loss', simple_value=monitor.current_state.train_loss)
        summary.value.add(tag='accuracy/accuracy', simple_value=monitor.current_state.train_accuracy)
        summary.value.add(tag='f1/f1', simple_value=monitor.current_state.train_f1)

        # Write training summary
        self.train_summary_writer.add_summary(summary=summary, global_step=monitor.current_state.global_step)

        # Flush summary writer
        self.train_summary_writer.flush()

    def close_summaries(self):
        """Close summary writers."""
        self.train_summary_writer.close()
        self.val_summary_writer.close()
