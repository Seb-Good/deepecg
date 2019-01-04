"""
physionet2017.py
---------------
This module provides classes and functions for creating a training dataset that is loaded directly from disc.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import shutil
import urllib
import zipfile

# Local imports
from deepecg.config.config import DATA_DIR


class Physionet2017DB(object):

    def __init__(self):

        # Set attributes
        self.db_url = 'https://physionet.org/challenge/2017/training2017.zip'
        self.db_name = 'physionet2017'
        self.raw_path = os.path.join(DATA_DIR, self.db_name, 'raw')
        self.processed_path = os.path.join(DATA_DIR, self.db_name, 'processed')
        self.zip_file_path = os.path.join(self.raw_path, self.db_name + '.zip')
        self.extract_path = os.path.join(self.raw_path, 'training2017')

        # Generate the raw
        self._generate_raw_db()

    def _generate_raw_db(self):
        """Generate the raw version of the Physionet2017 database in the 'raw' folder."""
        # Download the database
        self._download_db()

        # Unzip the database
        self._unzip_db()

        # Restructure folder
        self._restructure()

    def _download_db(self):
        """Download Physionet2017 database as zip file."""
        print('Downloading {}.'.format(self.db_url))
        urllib.request.urlretrieve(self.db_url, self.zip_file_path)

    def _unzip_db(self):
        """Unzip the raw db zip file."""
        print('Unzipping database.')
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_path)

    def _restructure(self):
        """Restructure folders and files."""
        print('Restructuring folders and files.')
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
