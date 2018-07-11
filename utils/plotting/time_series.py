"""
time_series.py
--------------
This module provides classes and functions for visualizing data and neural networks.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed


def plot_time_series(index, time_series, labels, fs, label_list):
    """Plots one univariate time series."""
    # Setup figure
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(wspace=0, hspace=0)
    ax = plt.subplot2grid((1, 1), (0, 0))

    # Set plot title
    ax.set_title(
        'Time Series Count: ' + str(labels.shape[0]) + '\n' +
        'Time Series ID: ' + str(index) + '\n' +
        'Label ID: ' + str(labels[index, 0]) + '\n' +
        'Label Name: ' + label_list[labels[index, 0]],
        fontsize=20, y=1.03
    )

    # Get time array
    time = np.arange(time_series.shape[1]) * 1 / fs

    # Plot image
    ax.plot(time, time_series[index, :], '-k', lw=2)

    # Axes labels
    ax.set_xlabel('Time, seconds', fontsize=20)
    ax.set_ylabel('Amplitude, mV', fontsize=20)
    ax.set_xlim([0, 60])
    ax.set_ylim([-1.5, 1.5])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlim([0, time.max()])

    plt.show()


def plot_time_series_widget(time_series, labels, fs, label_list):
    """Launch interactive plotting widget."""
    interact(
        plot_time_series,
        index=(0, labels.shape[0]-1, 1),
        time_series=fixed(time_series),
        labels=fixed(labels),
        fs=fixed(fs),
        label_list=fixed(label_list)
    )

