"""
train.py
--------
This module provides classes and functions for training a deep neural network in tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np

# Local imports
from train.logger import Logger
from train.monitor import Monitor
from train.summaries import Summaries
from train.mini_batches import mini_batch_generator
from utils.devices.device_check import get_device_count


def train(model, x_train, y_train, x_val, y_val, learning_rate, epochs, mini_batch_size):

    """Trains a tensorflow computational graph given some training and validation data."""

    # Get number of GPUs
    gpu_count = get_device_count(device_type='GPU')

    # Set random state for mini_batch
    mini_batch_seed = 0

    # Calculate number of mini-batches
    num_mini_batches = int(np.ceil(x_train.shape[0] / mini_batch_size))

    # Start Tensorflow session context
    with model.sess as sess:

        # Initialize model monitor
        monitor = Monitor(sess=sess, graph=model.graph, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                          learning_rate=learning_rate, mini_batch_size=mini_batch_size, save_path=model.save_path,
                          save_epoch=100, seed=mini_batch_seed, gpu_count=gpu_count)

        # Initialize logger
        logger = Logger(monitor=monitor, epochs=epochs, save_path=model.save_path, log_epoch=1,
                        num_mini_batches=num_mini_batches, mini_batch_size=mini_batch_size)

        # Initialize summary writers
        summaries = Summaries(sess=sess, graph=model.graph, path=model.save_path)

        # Write training and validation summaries
        summaries.write_train_summaries(monitor=monitor)
        summaries.write_val_summaries(monitor=monitor)

        # Save graph
        model.graph.save_graph(sess=sess)

        # Run the training loop
        for epoch in range(epochs):

            # Update mini_batch random seed
            mini_batch_seed += 1

            # Get random training mini_batches
            mini_batches = mini_batch_generator(x=x_train, y=y_train, mini_batch_size=mini_batch_size,
                                                seed=mini_batch_seed, gpu_count=gpu_count)

            # Initialize metrics operation
            sess.run(fetches=[model.graph.init_metrics_op])

            # Run mini_batch loop
            for mini_batch in mini_batches:

                # Select a mini_batch
                (mini_batch_x, mini_batch_y) = mini_batch

                # Run training step
                _, train_metrics_summary, global_step = sess.run(fetches=[model.graph.train_op,
                                                                          model.graph.train_summary_metrics_op,
                                                                          model.graph.global_step],
                                                                 feed_dict={model.graph.x: mini_batch_x,
                                                                            model.graph.y: mini_batch_y,
                                                                            model.graph.is_training: True,
                                                                            model.graph.learning_rate: learning_rate})

                # Get training parameter summaries
                summaries.write_train_summary_ops(summary=train_metrics_summary, global_step=global_step)

            # Update model monitor
            monitor.update_state(learning_rate=learning_rate)

            # Log training progress
            logger.log_training(monitor=monitor)

            # Write training and validation summaries
            summaries.write_train_summaries(monitor=monitor)
            summaries.write_val_summaries(monitor=monitor)

        # End monitoring
        monitor.end_monitoring()

        # End logging
        logger.end_logging()

        # Close summary writers
        summaries.close_summaries()

    # Close tensorflow session
    model.close_session()
