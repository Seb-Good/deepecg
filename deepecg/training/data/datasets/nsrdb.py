"""
nsrdb.py
--------
This module provides classes and methods for creating the MIT-BIH Normal Sinus Rhythm database.
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


class NSRDB(object):

    """
    The MIT-BIH Normal Sinus Rhythm Database
    https://physionet.org/physiobank/database/nsrdb/
    """

    def __init__(self):
        pass