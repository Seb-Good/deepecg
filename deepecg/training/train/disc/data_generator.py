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
        self.import_waveforms_fn = lambda file_path, label: self._import_waveforms(file_path=file_path, label=label)

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
        # Get image file names
        file_names = list(self.lookup_dict.keys())

        # Get labels
        labels = [self.lookup_dict[key] for key in self.lookup_dict.keys()]

        # file_paths and labels should have same length
        assert len(file_names) == len(labels)

        return file_names, labels

    def _import_waveforms(self, file_path, label):
        """Import waveform files from file path strings."""
        # Load numpy file
        waveforms = tf.py_func(self._load_npy_file, [file_path], [tf.float32])

        # Set tensor shape
        waveforms = tf.reshape(tensor=waveforms, shape=self.shape)

        return waveforms, label

    def _get_dataset(self):
        """Retrieve tensorflow Dataset object."""
        if self.mode == 'train':
            return (
                tf.data.Dataset.from_tensor_slices(
                    tensors=(tf.constant(value=self.file_paths),
                             tf.reshape(tensor=tf.constant(self.labels), shape=[-1]))
                )
                .shuffle(buffer_size=self.num_samples, reshuffle_each_iteration=True)
                .map(map_func=self.import_waveforms_fn, num_parallel_calls=self.num_parallel_calls)
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
                .map(map_func=self.import_waveforms_fn, num_parallel_calls=self.num_parallel_calls)
                .repeat()
                .batch(batch_size=self.batch_size)
                .prefetch(buffer_size=self.prefetch_buffer)
            )

    @staticmethod
    def _load_npy_file(file_path):
        """Python function for loading a single .npy file as casting the data type as float32."""
        return np.load(file_path.decode()).astype(np.float32)
