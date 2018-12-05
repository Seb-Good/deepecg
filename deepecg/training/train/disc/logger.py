"""
logger.py
---------
This module provides a class and methods for logging training performance.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import time
import logging
import pandas as pd
from datetime import datetime


class Logger(object):

    """Log model performance."""

    def __init__(self, monitor, epochs, save_path, log_epoch, batch_size, num_train_batches):

        # Set parameters
        self.monitor = monitor
        self.epochs = epochs
        self.save_path = save_path
        self.log_epoch = log_epoch
        self.batch_size = batch_size
        self.num_train_batches = num_train_batches

        # Set attributes
        self.start_time = time.time()
        self.start_datetime = str(datetime.utcnow())
        self.global_steps = self.epochs * self.num_train_batches
        self.logger = None
        self.best_model = None
        self.saver = None
        self.previous_time = self.start_time

        # Setup logger
        self._setup_logger()

        # Setup CSV
        self._setup_csv()

        # Start logging
        self._start_logging()

    def _setup_logger(self):

        # Logger
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(fmt='%(message)s')

        # Handler
        file_handler = logging.FileHandler(filename=os.path.join(self.save_path, 'logs', 'training.log'))
        file_handler.setFormatter(fmt=formatter)

        # Update logger
        self.logger.addHandler(hdlr=file_handler)

    def _setup_csv(self):

        # Import/create logger CSV
        if os.path.exists(os.path.join(self.save_path, 'logs', 'training.csv')):

            # Import
            self.csv = pd.read_csv(os.path.join(self.save_path, 'logs', 'training.csv'))

        else:
            # Create
            self.csv = pd.DataFrame(data=[], columns=['epoch', 'steps', 'train_time', 'epoch_time', 'lr', 'train_loss',
                                                      'val_loss', 'train_acc', 'val_acc'])

    def _compute_training_time(self):
        """Compute elapsed time from start of training."""
        return time.time() - self.start_time

    def _compute_epoch_time(self):
        """Compute epoch compute time."""
        # Get current time
        current_time = time.time()

        # Compute epoch time
        epoch_time = current_time - self.previous_time

        # Set previous time
        self.previous_time = current_time

        return epoch_time

    def _compute_mean_global_step_time(self):
        """Compute mean step compute time."""
        return self._compute_training_time() / self.global_steps

    def _compute_mean_epoch_time(self):
        """Compute mean epoch compute time."""
        return self._compute_training_time() / self.epochs

    def _log_training_parameters(self):
        self.logger.info('\nTraining Parameters :')
        self.logger.info('Mini-Batch Size : {:d}'.format(self.batch_size))
        self.logger.info('Training-Batches : {:d}'.format(self.num_train_batches))
        self.logger.info('Global Steps : {:.0f}'.format(self.global_steps))
        self.logger.info('Epochs : {}'.format(self.epochs))

    def _log_start_conditions(self):
        self.logger.info('\nStarting Conditions :')
        self._log_conditions()

    def _log_end_conditions(self):
        self.logger.info('\nEnding Conditions :')
        self._log_conditions()

    def _log_conditions(self):
        self.logger.info('Datetime : {:%Y-%m-%d %H:%M}'.format(datetime.utcnow()))
        self.logger.info('Learning Rate : {:.2e}'.format(self.monitor.learning_rate))
        self.logger.info('Global Step : {:.0f}'.format(self.monitor.current_state.global_step))
        self.logger.info('Epoch : {:.0f}'.format(self.monitor.current_state.epoch))
        self.logger.info('Training Loss : {:.6f}'.format(self.monitor.current_state.train_loss))
        self.logger.info('Validation Loss : {:.6f}'.format(self.monitor.current_state.val_loss))
        self.logger.info('Training Accuracy : {:.3f}'.format(self.monitor.current_state.train_accuracy * 100))
        self.logger.info('Validation Accuracy : {:.3f}'.format(self.monitor.current_state.val_accuracy * 100))

    def _log_time_summary(self):
        self.logger.info('\nTime Summary :')
        self.logger.info('Total Time : {:.2f} hours'.format(self._compute_training_time() / 3600.))
        self.logger.info('Time Per Global Step : {:.6f} seconds'.format(self._compute_mean_global_step_time()))
        self.logger.info('Time Per Epoch : {:.6f} seconds'.format(self._compute_mean_epoch_time()))

    def _log_best_model(self):
        self.logger.info('\nBest Model:')
        self.logger.info('Datetime : {}'.format(self.monitor.best_state.datetime))
        self.logger.info('Learning Rate : {:.2e}'.format(self.monitor.best_state.learning_rate))
        self.logger.info('Global Step : {:.0f}'.format(self.monitor.best_state.global_step))
        self.logger.info('Epoch : {:.0f}'.format(self.monitor.best_state.epoch))
        self.logger.info('Training Loss : {:.6f}'.format(self.monitor.best_state.train_loss))
        self.logger.info('Validation Loss : {:.6f}'.format(self.monitor.best_state.val_loss))
        self.logger.info('Training Accuracy : {:.3f}'.format(self.monitor.current_state.train_accuracy * 100))
        self.logger.info('Validation Accuracy : {:.3f}'.format(self.monitor.current_state.val_accuracy * 100))

    def _is_best(self):
        """Check for improvement."""
        if self.monitor.current_state.val_accuracy == self.monitor.best_state.val_accuracy:
            return '*'
        else:
            return ''

    def _get_training_log_string(self):

        # Set log string
        log_string = 'Epoch {0:.0f}, Step {1}, T-Time: {2:.3f} hr, E-Time: {3:.3f} min, lr: {4:.2e}, ' + \
                     'Train Loss: {5:.6f}, Val Loss: {6:.6f}, Train Acc: {7:.3f} %, Val Acc: {8:.3f} % {9}'

        return log_string.format(self.monitor.current_state.epoch, self.monitor.current_state.global_step,
                                 self._compute_training_time() / 3600., self._compute_epoch_time() / 60.,
                                 self.monitor.current_state.learning_rate, self.monitor.current_state.train_loss,
                                 self.monitor.current_state.val_loss, self.monitor.current_state.train_accuracy * 100,
                                 self.monitor.current_state.val_accuracy * 100, self._is_best())

    def log_training(self, monitor):
        """Log training results."""
        if self.monitor.current_state.epoch % self.log_epoch == 0:

            # Update current model state
            self.monitor = monitor

            # Get training log string
            training_log_string = self._get_training_log_string()

            # Log training
            self.logger.info(training_log_string)

            # Log training for CSV
            self._log_csv()

    def _close_logger(self):
        # Loop through handlers
        for handler in self.logger.handlers:

            # Close handler
            handler.close()

            # Remove handler from logger
            self.logger.removeHandler(handler)

        # Save logs CSV
        self._save_csv()

    def _log_csv(self):
        """Append training log to DataFrame."""
        self.csv = self.csv.append(self._get_training_log_csv(), ignore_index=True)

    def _save_csv(self):
        """Save Pandas DataFrame as CSV."""
        self.csv.to_csv(os.path.join(self.save_path, 'logs', 'training.csv'), index=False)

    def _get_training_log_csv(self):
        """Write training log to CSV."""
        return pd.Series(dict(epoch=self.monitor.current_state.epoch,
                              steps=self.monitor.current_state.global_step,
                              train_time=self._compute_training_time() / 60.,
                              epoch_time=self._compute_epoch_time() / 60.,
                              lr=self.monitor.current_state.learning_rate,
                              train_loss=self.monitor.current_state.train_loss,
                              val_loss=self.monitor.current_state.val_loss))

    def _start_logging(self):
        # Log all parameters at beginning of training
        self.logger.info('**************************\n*** Start Training ***\n**************************')

        # Log training parameters
        self._log_training_parameters()

        # Log start conditions
        self._log_start_conditions()

        self.logger.info('\nTraining Logs :')
        self.log_training(monitor=self.monitor)

    def end_logging(self):
        # Log end conditions
        self._log_end_conditions()

        # Log training time summary
        self._log_time_summary()

        # Lof best model
        self._log_best_model()

        self.logger.info('\n************************\n*** End Training ***\n************************\n\n\n')

        # Close logger
        self._close_logger()
