"""
waveforms.py
------------
This module provides classes and functions for importing and saving ECG files.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import pickle
import pandas as pd
import scipy.io as sio

# Local imports
from utils.data.ecg_tools.waveform import Waveform


class WaveformDB(object):

    def __init__(self, path_waveforms, path_labels, fs):

        # Set parameters
        self.path_waveforms = path_waveforms
        self.path_labels = path_labels
        self.fs = fs

        # Set attributes
        self.labels = None
        self.waveforms = None

    def load_database(self):

        # Load labels
        self.labels = self._load_labels()

        # Load waveform database
        self.waveforms = self._unpickle()

    def build_database(self):

        # Initialize waveforms dictionary
        self.waveforms = dict()

        # Load labels
        self.labels = self._load_labels()

        # Loop through waveform files
        for file_id in range(self.labels.shape[0]):

            # Get waveform
            waveform = Waveform(
                file_name=self.labels.loc[file_id, 'file_name'],
                label=self.labels.loc[file_id, 'label'],
                waveform=self._load_mat_file(file_name=self.labels.loc[file_id, 'file_name']),
                filter_bands=[3, 45],
                fs=self.fs
            )

            # Get waveform entry
            self.waveforms[waveform.file_name] = waveform.get_dictionary()

        # Pickle waveform database
        self._pickle(variable=self.waveforms, file_name='waveforms.pickle')

    def _load_labels(self):
        return pd.read_csv(os.path.join(self.path_labels, 'labels.csv'))

    def _load_mat_file(self, file_name):
        return sio.loadmat(os.path.join(self.path_waveforms, 'mat', file_name))['val'][0] / 1000.

    def _pickle(self, variable, file_name):

        # Set path
        path = os.path.join(self.path_waveforms, 'pickle')

        # Check for pickle directory
        if not os.path.exists(path):
            os.makedirs(path)

        # Pickle variable
        with open(os.path.join(path, file_name), 'wb') as handle:
            pickle.dump(variable, handle)

    def _unpickle(self):
        with open(os.path.join(self.path_waveforms, 'pickle', 'waveforms.pickle'), "rb") as input_file:
            return pickle.load(input_file)
