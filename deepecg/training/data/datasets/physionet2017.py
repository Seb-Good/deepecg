"""
physionet2017.py
----------------
This module provides classes and methods for creating the Physionet 2017 database.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import shutil
import urllib
import zipfile
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import interpolate

# Local imports
from deepecg.config.config import DATA_DIR


class Physionet2017DB(object):

    """
    AF Classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017
    https://physionet.org/challenge/2017/
    """

    def __init__(self):

        # Set attributes
        self.db_url = 'https://physionet.org/challenge/2017/training2017.zip'
        self.db_name = 'physionet2017'
        self.raw_path = os.path.join(DATA_DIR, 'datasets', self.db_name, 'raw')
        self.processed_path = os.path.join(DATA_DIR, 'datasets', self.db_name, 'processed')
        self.zip_file_path = os.path.join(self.raw_path, self.db_name + '.zip')
        self.extract_path = os.path.join(self.raw_path, 'training2017')
        self.fs = 200

    def generate_db(self):
        """Generate raw and processed databases."""
        # Generate raw database
        self.generate_raw_db()

        # Generate processed database
        self.generate_processed_db()

    def generate_raw_db(self):
        """Generate the raw version of the Physionet2017 database in the 'raw' folder."""
        print('Generating Raw Physionet2017 Database ...')
        # Download the database
        self._download_db()

        # Unzip the database
        self._unzip_db()

        # Restructure folder
        self._restructure()
        print('Complete!\n')

    def generate_processed_db(self):
        """Generate the processed version of the Physionet2017 database in the 'processed' folder."""
        print('Generating Processed Physionet2017 Database ...')
        # Load labels
        labels = self._load_labels()

        # Add database columns
        labels['db'] = self.db_name
        labels['path'] = np.nan

        # Loop through files
        for idx, row in labels.iterrows():

            # Load mat file
            waveform = self._load_mat_file(file_name=labels.loc[idx, 'file_name'] + '.mat')

            # Resample
            waveform = self._resample_waveform(waveform=waveform, fs=self.fs)

            # Save path
            save_path = os.path.join(self.processed_path, 'waveforms', labels.loc[idx, 'file_name'] + '.npy')

            # Save waveform as npy file
            np.save(save_path, waveform)

            # Add full processed path
            labels.loc[idx, 'path'] = save_path

        # Save labels
        labels.to_csv(os.path.join(self.processed_path, 'labels', 'labels.csv'), index=False)
        print('Complete!\n')

    def _load_mat_file(self, file_name):
        """Load Matlab waveform file."""
        return sio.loadmat(os.path.join(self.raw_path, file_name))['val'][0]

    def _load_labels(self):
        """Load CSV of rhythm labels."""
        return pd.read_csv(os.path.join(self.raw_path, 'REFERENCE.csv'), names=['file_name', 'label'])

    def _download_db(self):
        """Download Physionet2017 database as zip file."""
        print('Downloading {} ...'.format(self.db_url))
        urllib.request.urlretrieve(self.db_url, self.zip_file_path)

    def _unzip_db(self):
        """Unzip the raw db zip file."""
        print('Unzipping database ...')
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_path)

    def _restructure(self):
        """Restructure folders and files."""
        print('Restructuring folders and files ...')
        # Get list of file names
        file_names = os.listdir(self.extract_path)

        # Move all .mat files
        for file_name in file_names:
            if '.mat' in file_name:
                shutil.move(os.path.join(self.extract_path, file_name), os.path.join(self.raw_path, file_name))

        # Move v3 labels
        shutil.move(os.path.join(self.extract_path, 'REFERENCE.csv'), os.path.join(self.raw_path, 'REFERENCE.csv'))

        # Remove zip path
        shutil.rmtree(self.extract_path)

        # Remove zip file
        os.remove(self.zip_file_path)

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
