"""
datasets.py
-----------
This module provides classes and methods for pulling databases.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import pandas as pd

# Local imports
from deepecg.config.config import DATA_DIR
from deepecg.training.data.datasets.physionet2017 import Physionet2017DB
from deepecg.training.data.datasets.afdb import AFDB
from deepecg.training.data.datasets.nsrdb import NSRDB


class Datasets(object):

    """
    'physionet2017'
    AF Classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017
    https://physionet.org/challenge/2017/

    'afdb'
    The MIT-BIH Atrial Fibrillation Database
    https://physionet.org/physiobank/database/afdb/

    'nsrdb'
    The MIT-BIH Normal Sinus Rhythm Database
    https://physionet.org/physiobank/database/nsrdb/
    """

    def __init__(self, datasets):

        # Set parameters
        self.datasets = datasets

        # Set attributes
        self.labels = None

    def combine_dbs(self):
        """Combine listed databases."""
        # Create empty DataFrame
        self.labels = pd.DataFrame(data=[], columns=['file_name', 'label', 'db', 'path', 'func'])

        if 'physionet2017' in list(self.datasets.keys()):
            # Import labels
            labels = pd.read_csv(os.path.join(DATA_DIR, 'datasets', 'physionet2017',
                                              'processed', 'labels', 'labels.csv'))

            # Add dataset function
            labels['func'] = self.datasets['physionet2017']

            # Append labels
            self.labels = self.labels.append(labels)

        if 'afdb' in self.datasets:
            # Import labels
            labels = pd.read_csv(os.path.join(DATA_DIR, 'datasets', 'afdb', 'processed', 'labels', 'labels.csv'))

            # Add dataset function
            labels['func'] = self.datasets['afdb']

            # Append labels
            self.labels = self.labels.append(labels, ignore_index=True)

        if 'nsrdb' in self.datasets:
            # Import labels
            labels = pd.read_csv(os.path.join(DATA_DIR, 'datasets', 'nsrdb', 'processed', 'labels', 'labels.csv'))

            # Add dataset function
            labels['func'] = self.datasets['nsrdb']

            # Append labels
            self.labels = self.labels.append(labels, ignore_index=True)

        # Save labels
        self.labels.reset_index(drop=True).to_csv(os.path.join(DATA_DIR, 'datasets', 'labels.csv'), index=False)

    def generate_dbs(self):
        """Generate listed databases."""
        if 'physionet2017' in list(self.datasets.keys()):
            # Initialize
            db = Physionet2017DB()

            # Generate raw database
            db.generate_db()

        if 'afdb' in self.datasets:
            # Initialize
            db = AFDB()

            # Generate raw database
            db.generate_db()

        if 'nsrdb' in self.datasets:
            # Initialize
            db = NSRDB()

            # Generate raw database
            db.generate_db()
