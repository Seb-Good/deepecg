"""
training_dataset_disc.py
------------------------
This module provides classes and functions for creating a training dataset that is loaded directly from disc.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

# Local imports
from deepecg.training.data.ecg import ECG


class TrainingDB(object):

    def __init__(self, path_labels, path_save, duration, classes, datasets, augment, fs):

        # Set parameters
        self.path_labels = path_labels
        self.path_save = path_save
        self.duration = duration
        self.classes = classes
        self.datasets = datasets
        self.augment = augment
        self.fs = fs

        # Check inputs
        self._check_datasets_input()

        # Set attributes
        self.length = int(self.duration * self.fs)
        self.labels = self._load_labels()

        # Split dataset
        self._split_dataset()

    def _check_datasets_input(self):
        """Check to ensure train, val, test proportions add to 1."""
        if self.datasets['train'] + self.datasets['val'] + self.datasets['test'] != 1.:
            raise ValueError('train, val, and test proportions must add to 1.')

    def _load_labels(self):
        """Load CSV of rhythm labels."""
        # Load csv as DataFrame
        labels = pd.read_csv(os.path.join(self.path_labels, 'labels.csv'))

        # Select classes
        labels = labels[labels['label'].isin(self.classes)]

        return labels

    def _split_dataset(self):
        """Split dataset into train, val, and test segments."""
        # Get target labels
        labels = self.labels[self.labels['func'] == 'target']

        if self.datasets['test'] > 0.:
            # Get evaluate size
            test_size = self.datasets['val'] + self.datasets['test']

            # Split dataset into train/evaluate
            _, _, train, evaluate = train_test_split(labels, labels, test_size=test_size,
                                                     stratify=labels['label'], random_state=0)

            # Get test size
            test_size = self.datasets['test'] / (self.datasets['val'] + self.datasets['test'])

            # Split evaluate into val/test
            _, _, val, test = train_test_split(evaluate, evaluate, test_size=test_size,
                                               stratify=evaluate['label'], random_state=0)

        else:
            # Split dataset into train/evaluate
            _, _, train, val = train_test_split(labels, labels, test_size=self.datasets['val'],
                                                stratify=labels['label'], random_state=0)

        # Add to dataset split to labels DataFrame
        self._add_dataset_labels(train_index=train.index, val_index=val.index, test_index=[])

    def _add_dataset_labels(self, train_index, val_index, test_index):
        """Add to dataset split to labels DataFrame."""
        self.labels['dataset'] = np.nan
        self.labels.loc[train_index, 'dataset'] = 'train'
        self.labels.loc[val_index, 'dataset'] = 'val'
        self.labels.loc[test_index, 'dataset'] = 'test'
        self.labels['dataset'][self.labels['func'] == 'augment'] = 'train'

    @staticmethod
    def _load_waveform_file(path):
        """Load NumPy (.npy) file."""
        return np.load(path)

    def _process_waveform(self, idx):
        """Process a single waveform and add to training dataset."""
        # Get file name
        file_name = self.labels.loc[idx, 'file_name']

        # Get dataset
        dataset = self.labels.loc[idx, 'dataset']

        # Get path
        path = self.labels.loc[idx, 'path']

        try:
            # Load and process ECG waveform
            ecg = ECG(file_name=self.labels.loc[idx, 'file_name'], label=self.labels.loc[idx, 'label'],
                      waveform=self._load_waveform_file(path=path), filter_bands=[3, 45], fs=self.fs)

            # Set waveform duration
            waveform = self._set_duration(waveform=ecg.filtered)

            # Augment waveforms
            labels = self._augment(file_name=file_name, waveform=waveform, label=ecg.label_int, dataset=dataset)

            return labels

        except ValueError:
            return [None]

    def _augment(self, file_name, waveform, label, dataset):
        """Augment waveforms with variable padding."""
        # List for labels
        labels = list()

        if len(waveform) < self.length and dataset == 'train':
            for align in ['left', 'center', 'right']:

                # Pad waveform
                waveform_pad = self._zero_pad(waveform=waveform, align=align)

                # Set file name
                file_name_pad = '{}_{}'.format(file_name, align)

                # Save waveform
                self._save_waveform(waveform=waveform_pad, dataset=dataset, file_name=file_name_pad)

                # Add label
                labels.append({'dataset': dataset, 'file_name': file_name_pad, 'label': label})

        else:
            # Save waveform
            self._save_waveform(waveform=self._zero_pad(waveform=waveform, align='center'),
                                dataset=dataset, file_name=file_name)

            # Add label
            labels.append({'dataset': dataset, 'file_name': file_name, 'label': label})

        return labels

    def _save_waveform(self, waveform, dataset, file_name):
        """Save data array as .npy"""
        np.save(os.path.join(self.path_save, dataset, 'waveforms', file_name + '.npy'), waveform)

    def _set_duration(self, waveform):
        """Set duration of ecg waveform."""
        if len(waveform) > self.length:
            return waveform[0:self.length]
        else:
            return waveform

    def _zero_pad(self, waveform, align):
        """Zero pad waveform (align: left, center, right)."""
        # Get remainder
        remainder = self.length - len(waveform)

        if align == 'left':
            return np.pad(waveform, (0, remainder), 'constant', constant_values=0)
        elif align == 'center':
            return np.pad(waveform, (int(remainder / 2), remainder - int(remainder / 2)), 'constant', constant_values=0)
        elif align == 'right':
            return np.pad(waveform, (remainder, 0), 'constant', constant_values=0)

    def generate(self):
        """Generate training dataset."""
        outputs = Parallel(n_jobs=-1)(delayed(self._process_waveform)(idx)
                                      for idx in self.labels.index)

        # Save labels
        self._save_labels(outputs=outputs)

    def generate_test(self):
        """Generate training dataset (test)."""
        for idx in self.labels.index[0:200]:
            self._process_waveform(idx=idx)

    def _save_labels(self, outputs):
        """Save labels as JSON."""
        # Empty dicts for labels
        labels = {'train': {}, 'val': {}, 'test': {}}

        # Loop through output
        for output in outputs:
            for instance in output:
                # Add label
                if instance is not None:
                    labels[instance['dataset']][instance['file_name']] = instance['label']

        # Save labels
        self._save_json_labels(labels=labels['train'], dataset='train')
        self._save_json_labels(labels=labels['val'], dataset='val')
        self._save_json_labels(labels=labels['test'], dataset='test')

    def _save_json_labels(self, labels, dataset):
        """Save labels as JSON for defined dataset type."""
        with open(os.path.join(self.path_save, dataset, 'labels', 'labels.json'), 'w') as file:
            json.dump(labels, file, sort_keys=True)
