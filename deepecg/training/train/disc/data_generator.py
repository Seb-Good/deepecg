"""
data_generator.py
-----------------
This module provides a class for generating data batches for training and evaluation.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import json
import numpy as np
import tensorflow as tf


class DataGenerator(object):

    def __init__(self, path, mode, shape, batch_size, prefetch_buffer=1, seed=0, num_parallel_calls=1):

        # Set parameters
        self.path = path
        self.mode = mode
        self.shape = shape
        self.batch_size = batch_size
        self.prefetch_buffer = prefetch_buffer
        self.seed = seed
        self.num_parallel_calls = num_parallel_calls

        # Set attributes
        self.lookup_dict = self._get_lookup_dict()
        self.file_names, self.labels = self._get_file_names_and_labels()
        self.num_samples = len(self.lookup_dict)
        self.file_paths = self._get_file_paths()
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        self.current_seed = 0

        # Get lambda functions
        self.import_waveforms_fn_train = lambda file_path, label: self._import_waveform(file_path=file_path,
                                                                                        label=label, augment=True)
        self.import_waveforms_fn_val = lambda file_path, label: self._import_waveform(file_path=file_path,
                                                                                      label=label, augment=False)
        # Get dataset
        self.dataset = self._get_dataset()

        # Get iterator
        self.iterator = self.dataset.make_initializable_iterator()

    def _get_next_seed(self):
        """update current seed"""
        self.current_seed += 1
        return self.current_seed

    def _get_file_paths(self):
        """Convert file names to full absolute file paths with .npy extension."""
        return [os.path.join(self.path, self.mode, 'waveforms', file_name + '.npy') for file_name in self.file_names]

    def _get_lookup_dict(self):
        """Load lookup dictionary {'file_name': label}."""
        return json.load(open(os.path.join(self.path, self.mode, 'labels', 'labels.json')))

    def _get_file_names_and_labels(self):
        """Get list of waveform npy file paths and labels."""
        # Get waveform file names
        file_names = list(self.lookup_dict.keys())

        # Get labels
        labels = [self.lookup_dict[key] for key in self.lookup_dict.keys()]

        # file_paths and labels should have same length
        assert len(file_names) == len(labels)

        return file_names, labels

    def _import_waveform(self, file_path, label, augment):
        """Import waveform file from file path string."""
        # Load numpy file
        waveform = tf.py_func(self._load_npy_file, [file_path], [tf.float32])

        # Set tensor shape
        waveform = tf.reshape(tensor=waveform, shape=self.shape)

        # Augment waveform
        if augment:
            waveform = self._augment(waveform=waveform)

        return waveform, label

    def _augment(self, waveform):
        """Apply random augmentations."""
        # Random amplitude scale
        waveform = self._random_scale(waveform=waveform, prob=0.5)

        # Random polarity flip
        waveform = self._random_polarity(waveform=waveform, prob=0.5)

        return waveform

    def _random_scale(self, waveform, prob):
        """Apply random multiplication factor."""
        # Get random true or false
        prediction = self._random_true_false(prob=prob)

        # Apply random multiplication factor
        waveform = tf.cond(prediction, lambda: self._scale(waveform=waveform),
                           lambda: self._do_nothing(waveform=waveform))

        return waveform

    @staticmethod
    def _scale(waveform):
        """Apply random multiplication factor."""
        # Get random scale factor
        scale_factor = tf.random_uniform(shape=[], minval=0.5, maxval=2.5, dtype=tf.float32)

        return waveform * scale_factor

    def _random_polarity(self, waveform, prob):
        """Apply random polarity flip."""
        # Get random true or false
        prediction = self._random_true_false(prob=prob)

        # Apply random polarity flip
        waveform = tf.cond(prediction, lambda: self._polarity(waveform=waveform),
                           lambda: self._do_nothing(waveform=waveform))

        return waveform

    @staticmethod
    def _polarity(waveform):
        """Apply random polarity flip."""
        return waveform * -1

    @staticmethod
    def _do_nothing(waveform):
        return waveform

    @staticmethod
    def _random_true_false(prob):
        """Get a random true or false."""
        # Get random probability between 0 and 1
        probability = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        return tf.less(x=probability, y=prob)

    def _get_dataset(self):
        """Retrieve tensorflow Dataset object."""
        if self.mode == 'train':
            return (
                tf.data.Dataset.from_tensor_slices(
                    tensors=(tf.constant(value=self.file_paths),
                             tf.reshape(tensor=tf.constant(self.labels), shape=[-1]))
                )
                .shuffle(buffer_size=self.num_samples, reshuffle_each_iteration=True)
                .map(map_func=self.import_waveforms_fn_train, num_parallel_calls=self.num_parallel_calls)
                .repeat()
                .batch(batch_size=self.batch_size)
                .prefetch(buffer_size=self.prefetch_buffer)
            )
        else:
            return (
                tf.data.Dataset.from_tensor_slices(
                    tensors=(tf.constant(value=self.file_paths),
                             tf.reshape(tensor=tf.constant(self.labels), shape=[-1]))
                )
                .map(map_func=self.import_waveforms_fn_val, num_parallel_calls=self.num_parallel_calls)
                .repeat()
                .batch(batch_size=self.batch_size)
                .prefetch(buffer_size=self.prefetch_buffer)
            )

    @staticmethod
    def _load_npy_file(file_path):
        """Python function for loading a single .npy file as casting the data type as float32."""
        return np.load(file_path.decode()).astype(np.float32)
