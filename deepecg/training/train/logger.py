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

    """Logs, saves, and prints training progress metrics."""

    def __init__(self, monitor, epochs, save_path, log_epoch, num_mini_batches, mini_batch_size):

        # Set parameters
        self.monitor = monitor
        self.epochs = epochs
        self.save_path = save_path
        self.log_epoch = log_epoch
        self.num_mini_batches = num_mini_batches
        self.mini_batch_size = mini_batch_size

        # Set attributes
        self.start_datetime = str(datetime.utcnow())
        self.start_time = time.time()
        self.previous_time = self.start_time
        self.global_steps = self.epochs * self.num_mini_batches
        self.logger = None
        self.best_model = None

        # Setup CSV
        self._setup_csv()

        # Setup logger
        self._setup_logger()

        # Start logger
        self._start_logger()

    def _setup_csv(self):
        """Import/create DataFrame for saving logs as CSV."""
        if os.path.exists(os.path.join(self.save_path, 'logs', 'training.csv')):
            # Import CSV as DataFrame
            self.csv = pd.read_csv(os.path.join(self.save_path, 'logs', 'training.csv'))
        else:
            # Create DataFrame
            self.csv = pd.DataFrame(data=[], columns=['epoch', 'steps', 'train_time', 'epoch_time', 'lr', 'train_loss',
                                                      'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1'])

    def _setup_logger(self):
        """Setup training logger."""
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

    def _compute_epoch_train_time(self):
        """Compute epoch training time."""
        # Get current time
        current_time = time.time()

        # Compute epoch time
        epoch_time = current_time - self.previous_time

        # Set previous time
        self.previous_time = current_time

        return epoch_time

    def _compute_training_time(self):
        """Compute time spent training to date."""
        return time.time() - self.start_time

    def _log_training_parameters(self):
        self.logger.info('\nTraining Parameters :')
        self.logger.info('Mini-Batch Size : {:d}'.format(self.mini_batch_size))
        self.logger.info('Mini-Batches : {:d}'.format(self.num_mini_batches))
        self.logger.info('Global Steps : {:.0f}'.format(self.global_steps))
        self.logger.info('Epochs : {}'.format(self.epochs))

    def _log_start_state(self):
        self.logger.info('\nStarting State :')
        self._log_state()

    def _log_end_state(self):
        self.logger.info('\nEnding State :')
        self._log_state()

    def _log_state(self):
        self.logger.info('Datetime : {:%Y-%m-%d %H:%M}'.format(datetime.utcnow()))
        self.logger.info('Learning Rate : {}'.format(self.monitor.learning_rate))
        self.logger.info('Global Step : {:.0f}'.format(self.monitor.current_state.global_step))
        self.logger.info('Epoch : {:.0f}'.format(self.monitor.current_state.epoch))
        self.logger.info('Training Loss : {:.6f}'.format(self.monitor.current_state.train_loss))
        self.logger.info('Validation Loss : {:.6f}'.format(self.monitor.current_state.val_loss))
        self.logger.info('Training Accuracy : {:.3f} %'.format(self.monitor.current_state.train_accuracy * 100.))
        self.logger.info('Validation Accuracy : {:.3f} %'.format(self.monitor.current_state.val_accuracy * 100.))
        self.logger.info('Training F1 : {:.3f} %'.format(self.monitor.current_state.train_f1 * 100.))
        self.logger.info('Validation F1 : {:.3f} %'.format(self.monitor.current_state.val_f1 * 100.))

    def _log_best_state(self):
        self.logger.info('\nBest State:')
        self.logger.info('Datetime : {}'.format(self.monitor.best_state.datetime))
        self.logger.info('Learning Rate : {}'.format(self.monitor.best_state.learning_rate))
        self.logger.info('Global Step : {:.0f}'.format(self.monitor.best_state.global_step))
        self.logger.info('Epoch : {:.0f}'.format(self.monitor.best_state.epoch))
        self.logger.info('Training Loss : {:.6f}'.format(self.monitor.best_state.train_loss))
        self.logger.info('Validation Loss : {:.6f}'.format(self.monitor.best_state.val_loss))
        self.logger.info('Training Accuracy : {:.3f} %'.format(self.monitor.best_state.train_accuracy * 100.))
        self.logger.info('Validation Accuracy : {:.3f} %'.format(self.monitor.best_state.val_accuracy * 100.))
        self.logger.info('Training F1 : {:.3f} %'.format(self.monitor.current_state.train_f1 * 100.))
        self.logger.info('Validation F1 : {:.3f} %'.format(self.monitor.current_state.val_f1 * 100.))

    def _get_training_log_string(self):

        # Set log string
        log_string = 'Epoch {0:.0f}, Step {1}, T-Time: {2:.3f} hr, E-Time: {3:.3f} min, lr: {4:.2e}, ' + \
                     'Train Loss: {5:.6f}, Val Loss: {6:.6f}, Train Acc: {7:.3f} %, Val Acc: {8:.3f} %, ' + \
                     'Train F1: {9:.3f} %, Val F1: {10:.3f} % {11}'

        return log_string.format(self.monitor.current_state.epoch, self.monitor.current_state.global_step,
                                 self._compute_training_time() / 3600., self._compute_epoch_train_time() / 60.,
                                 self.monitor.current_state.learning_rate, self.monitor.current_state.train_loss,
                                 self.monitor.current_state.val_loss, self.monitor.current_state.train_accuracy * 100.,
                                 self.monitor.current_state.val_accuracy * 100.,
                                 self.monitor.current_state.train_f1 * 100., self.monitor.current_state.val_f1 * 100.,
                                 self._is_best())

    def _is_best(self):

        # Check for improvement
        if self.monitor.current_state.val_f1 == self.monitor.best_state.val_f1:
            return '*'
        else:
            return ''

    def log_training(self, monitor):

        # Log training results ever log_epoch
        if self.monitor.current_state.epoch % self.log_epoch == 0:

            # Update current model state
            self.monitor = monitor

            # Get training log string
            training_log_string = self._get_training_log_string()

            # Log training
            self.logger.info(training_log_string)

            # Log training to CSV
            self._log_csv()

    def print_training(self):

        # Log training results ever log_epoch
        if self.monitor.current_state.epoch % self.log_epoch == 0:

            # Get training log string
            training_log_string = self._get_training_log_string()

            # print training
            print(training_log_string)

    def _start_logger(self):

        # Log all parameters at beginning of training
        self.logger.info('*** Start Training ***')

        # Log training parameters
        self._log_training_parameters()

        # Log start state
        self._log_start_state()

        self.logger.info('\nTraining Epoch Logs :')
        self.log_training(monitor=self.monitor)

    def end_logging(self):

        # Log best model state
        self._log_best_state()

        # End log
        self.logger.info('\n*** End Training ***\n\n\n')

        # Close logger
        self._close_logger()

    def _close_logger(self):

        # Loop through handlers
        for handler in self.logger.handlers:

            # Close handler
            handler.close()

            # Remove handler from logger
            self.logger.removeHandler(handler)

        # Save training logs as CSV
        self._save_csv()

    def _log_csv(self):
        """Append training epoch log to DataFrame."""
        self.csv = self.csv.append(self._get_training_log_csv(), ignore_index=True)

    def _save_csv(self):
        """Save Pandas DataFrame as CSV."""
        self.csv.to_csv(os.path.join(self.save_path, 'logs', 'training.csv'), index=False)

    def _get_training_log_csv(self):
        return pd.Series(dict(epoch=self.monitor.current_state.epoch,
                              steps=self.monitor.current_state.global_step,
                              train_time=self._compute_training_time(),
                              epoch_time=self._compute_epoch_train_time(),
                              lr=self.monitor.current_state.learning_rate,
                              train_loss=self.monitor.current_state.train_loss,
                              val_loss=self.monitor.current_state.val_loss,
                              train_acc=self.monitor.current_state.train_accuracy * 100.,
                              val_acc=self.monitor.current_state.val_accuracy * 100.,
                              train_f1=self.monitor.current_state.train_f1 * 100.,
                              val_f1=self.monitor.current_state.val_f1 * 100.))
