"""
device_check.py
---------------
This module provides a function for assessing available CPUs and GPUs.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
from tensorflow.python.client import device_lib


def get_device_names(device_type):
    """Return list of GPU names"""
    devices = device_lib.list_local_devices()
    return [x.name for x in devices if x.device_type == device_type]


def get_device_count(device_type):
    """Return number of available GPUs."""
    return len(get_device_names(device_type=device_type))


def print_device_counts():
    """Print number of available CPUs and GPUs"""
    print('Workstation has {0:.0f} CPUs.'.format(get_device_count(device_type='CPU')))
    print('Workstation has {0:.0f} GPUs.'.format(get_device_count(device_type='GPU')))
