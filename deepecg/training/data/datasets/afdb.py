"""
afdb.py
-------
This module provides classes and methods for creating the MIT-BIH Atrial Fibrillation database.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import wfdb
import numpy as np
import pandas as pd
from scipy import interpolate

# Local imports
from deepecg.config.config import DATA_DIR


class AFDB(object):

    """
    The MIT-BIH Atrial Fibrillation Database
    https://physionet.org/physiobank/database/afdb/
    """

    def __init__(self):

        # Set attributes
        self.db_name = 'afdb'
        self.raw_path = os.path.join(DATA_DIR, 'datasets', self.db_name, 'raw')
        self.processed_path = os.path.join(DATA_DIR, 'datasets', self.db_name, 'processed')
        self.label_dict = {'AFIB': 'atrial fibrillation', 'AFL': 'atrial flutter', 'J': 'AV junctional rhythm'}
        self.fs = 300
        self.length = 60
        self.length_sp = self.length * self.fs
        self.record_ids = None
        self.sections = None
        self.samples = None
        self.labels = None

    def generate_db(self):
        """Generate raw and processed databases."""
        # Generate raw database
        self.generate_raw_db()

        # Generate processed database
        self.generate_processed_db()

    def generate_raw_db(self):
        """Generate the raw version of the MIT-BIH Atrial Fibrillation database in the 'raw' folder."""
        print('Generating Raw MIT-BIH Atrial Fibrillation Database ...')
        # Download database
        wfdb.dl_database(self.db_name, self.raw_path)

        # Get list of recordings
        self.record_ids = [file.split('.')[0] for file in os.listdir(self.raw_path) if '.dat' in file]
        print('Complete!\n')

    def generate_processed_db(self):
        """Generate the processed version of the MIT-BIH Atrial Fibrillation database in the 'processed' folder."""
        print('Generating Processed MIT-BIH Atrial Fibrillation Database ...')
        # Get sections
        self.sections = self._get_sections()

        # Get training samples
        self.samples = self._get_samples()

        # Create empty DataFrame
        self.labels = pd.DataFrame(data=[], columns=['file_name', 'label', 'db', 'path'])

        # Loop through samples
        for idx, sample in enumerate(self.samples):

            # Set file name
            file_name = '{}_{}_{}'.format(sample['db'], sample['record'], idx)

            # Save path
            save_path = os.path.join(self.processed_path, 'waveforms', file_name + '.npy')

            # Get labels
            self.labels = self.labels.append(
                pd.Series({'file_name': file_name, 'label': 'A' if sample['label'] == 'AFIB' else 'O',
                           'db': sample['db'], 'path': save_path}), ignore_index=True)

            # Save waveform as .npy
            np.save(save_path, sample['waveform'])

        # Save labels
        self.labels.to_csv(os.path.join(self.processed_path, 'labels', 'labels.csv'), index=False)
        print('Complete!\n')

    def _get_sections(self):
        """Collect continuous arrhythmia sections."""
        # Empty dictionary for arrhythmia sections
        sections = list()

        # Loop through records
        for record_id in self.record_ids:

            # Import recording
            record = wfdb.rdrecord(os.path.join(self.raw_path, record_id))

            # Import annotations
            annotation = wfdb.rdann(os.path.join(self.raw_path, record_id), 'atr')

            # Get sample frequency
            fs = record.__dict__['fs']

            # Get waveform
            waveform = record.__dict__['p_signal']

            # labels
            labels = [label[1:] for label in annotation.__dict__['aux_note']]

            # Samples
            sample = annotation.__dict__['sample']

            # Loop through labels and collect sections
            for idx, label in enumerate(labels):

                if any(label in val for val in list(self.label_dict.keys())):

                    if idx != len(labels) - 1:
                        sections.append({'label': label, 'section': idx, 'record': record_id, 'fs': fs, 'channel': 0,
                                         'db': self.db_name, 'waveform': waveform[sample[idx]:sample[idx + 1], 0]})
                        sections.append({'label': label, 'section': idx, 'record': record_id, 'fs': fs, 'channel': 1,
                                         'db': self.db_name, 'waveform': waveform[sample[idx]:sample[idx + 1], 1]})

                    elif idx == len(labels) - 1:
                        sections.append({'label': label, 'section': idx, 'record': record_id, 'fs': fs, 'channel': 0,
                                         'db': self.db_name, 'waveform': waveform[sample[idx]:, 0]})
                        sections.append({'label': label, 'section': idx, 'record': record_id, 'fs': fs, 'channel': 1,
                                         'db': self.db_name, 'waveform': waveform[sample[idx]:, 1]})

        return sections

    def _get_samples(self):
        """Collect arrhythmia training samples."""
        # Empty dictionary for arrhythmia training samples
        samples = list()

        # Loop through sections
        for section in self.sections[0:100]:

            # Set index
            idx = 0

            # Get number of samples in section
            num_samples = int(np.ceil(len(section['waveform']) / self.length_sp))

            # Loop through samples
            for sample_id in range(num_samples):

                # Get training sample
                if sample_id != num_samples - 1:
                    samples.append(
                        {'label': section['label'], 'section': section['section'], 'record': section['record'],
                         'sample': sample_id, 'fs': self.fs, 'db': section['db'], 'channel': section['channel'],
                         'waveform': self._resample_waveform(waveform=section['waveform'][idx:idx + self.length_sp + 1],
                                                             fs=self.fs)}
                    )
                    idx += self.length_sp

                elif sample_id == num_samples - 1:
                    samples.append(
                        {'label': section['label'], 'section': section['section'], 'record': section['record'],
                         'sample': sample_id, 'fs': self.fs, 'db': section['db'], 'channel': section['channel'],
                         'waveform': self._resample_waveform(waveform=section['waveform'][idx:], fs=self.fs)}
                    )

        return samples

    def _resample_waveform(self, waveform, fs):
        """Resample training sample to set sample frequency."""
        # Get time array
        time = np.arange(len(waveform)) * 1 / self.fs

        # Generate new resampling time array
        times_rs = np.arange(0, time[-1], 1 / fs)

        # Setup interpolation function
        interp_func = interpolate.interp1d(x=time, y=waveform, kind='linear')

        # Interpolate contiguous segment
        sample_rs = interp_func(times_rs)

        return sample_rs
