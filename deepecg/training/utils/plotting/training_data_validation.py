"""
training_data_validation.py
---------------------------
This module provide functions, classes and methods for plotting training data for validation.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import json
import numpy as np
import matplotlib.pylab as plt
from ipywidgets import interact, fixed


def interval_plot(label_id, labels, path, dataset):
    """Plot measure vs time."""
    # Label lookup
    label_lookup = {0: 'N', 1: 'A', 2: 'O', 3: '~'}

    # File name
    file_name = list(labels.keys())[label_id]

    # Get label
    label = labels[file_name]

    # Load data
    data = np.load(os.path.join(path, dataset, 'waveforms', file_name + '.npy'))

    # Time array
    time = np.arange(data.shape[0]) * 1 / 300

    # Setup figure
    fig = plt.figure(figsize=(15, 5), facecolor='w')
    fig.subplots_adjust(wspace=0, hspace=0.05)
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    # ECG
    ax1.set_title('Dataset: {}\nFile Name: {}\nLabel: {}'.format(dataset, file_name, label_lookup[label]), fontsize=20)
    ax1.plot(time, data, '-k', lw=2)

    # Axes labels
    ax1.set_xlabel('Time, seconds', fontsize=20)
    ax1.set_ylabel('ECG', fontsize=20)
    ax1.set_xlim([time.min(), time.max()])
    plt.yticks(fontsize=12)

    plt.show()


def interval_plot_interact(path, dataset):
    """Launch interactive plotting widget."""
    # Get labels
    labels = json.load(open(os.path.join(path, dataset, 'labels', 'labels.json')))

    interact(interval_plot,
             label_id=(0, len(labels.keys()) - 1, 1),
             labels=fixed(labels),
             path=fixed(path),
             dataset=fixed(dataset))
