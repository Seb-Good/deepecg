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
        self.train_summary_path = os.path.join(self.path, 'train')
        self.val_summary_path = os.path.join(self.path, 'val')

        # Initialize Tensorboard writers
        self.train_summary_writer = tf.summary.FileWriter(logdir=self.train_summary_path, graph=self.sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(logdir=self.val_summary_path)

    def log_train_summaries(self, summary, global_step):
        """Add training summary."""
        self.train_summary_writer.add_summary(summary=summary, global_step=global_step)

    def log_scalar_summaries(self, monitor):
        # Get training summary
        self.log_train_scalar_summaries(monitor=monitor)

        # Get validation summary
        self.log_val_scalar_summaries(monitor=monitor)

    def log_train_scalar_summaries(self, monitor):
        # Create value summary
        summary = tf.Summary()
        summary.value.add(tag='loss/loss', simple_value=monitor.current_state.train_loss)
        summary.value.add(tag='accuracy/accuracy', simple_value=monitor.current_state.train_accuracy)
        summary.value.add(tag='f1/f1', simple_value=monitor.current_state.train_f1)

        # Get validation summary
        self.train_summary_writer.add_summary(summary=summary, global_step=monitor.current_state.global_step)

        # Flush summary writer
        self.train_summary_writer.flush()

    def log_val_scalar_summaries(self, monitor):
        # Create value summary
        summary = tf.Summary()
        summary.value.add(tag='loss/loss', simple_value=monitor.current_state.val_loss)
        summary.value.add(tag='accuracy/accuracy', simple_value=monitor.current_state.val_accuracy)
        summary.value.add(tag='f1/f1', simple_value=monitor.current_state.val_f1)

        # Get validation summary
        self.val_summary_writer.add_summary(summary=summary, global_step=monitor.current_state.global_step)

        # Flush summary writer
        self.val_summary_writer.flush()

    def log_val_cam_plots_summaries(self, monitor):
        """Generate class activation map plot summaries."""
        if monitor.current_state.val_f1 == monitor.best_state.val_f1:

            # Get validation cam plots as numpy array
            val_cam_plots = self.sess.run([monitor.current_state.val_cam_plots])[0]

            # Get summary
            summary = self.sess.run(fetches=[self.graph.val_cam_plots_summary_op],
                                    feed_dict={self.graph.val_cam_plots: val_cam_plots})

            # Write summary
            self.val_summary_writer.add_summary(summary=summary[0], global_step=monitor.current_state.global_step)

            # Flush summary writer
            self.val_summary_writer.flush()

    def close_summaries(self):
        """Close summary writers."""
        self.train_summary_writer.close()
        self.val_summary_writer.close()
