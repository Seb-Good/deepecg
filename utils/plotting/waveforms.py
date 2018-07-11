"""
waveforms.py
------------
This module provides classes and functions for plotting ECG waveforms.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
from ipywidgets import *
import matplotlib.pylab as plt


def plot_waveforms(index, waveforms):
    """Plots one univariate time series."""
    # Get file name
    file_name = list(waveforms.keys())[index]

    # Setup plot
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(hspace=0.25)
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.set_title(
        'File Name: ' + file_name + '\n'  
        'Label: ' + waveforms[file_name]['label_str'], fontsize=20
    )

    # Plot waveform
    ax1.plot(waveforms[file_name]['time'], waveforms[file_name]['filtered'], '-k', label='Filtered')
    ax1.vlines(
        waveforms[file_name]['rpeaks_ts'],
        waveforms[file_name]['filtered'].min() - 0.01,
        waveforms[file_name]['filtered'].max() + 0.01,
        color=[0.7, 0.7, 0.7],
        linewidth=4,
        label='R-Peaks'
    )

    ax1.set_xlabel('Time, seconds', fontsize=25)
    ax1.set_ylabel('Normalized Amplitude', fontsize=25)
    ax1.set_xlim([0, waveforms[file_name]['duration']])
    ax1.set_ylim([waveforms[file_name]['filtered'].min() - 0.01, waveforms[file_name]['filtered'].max() + 0.01])
    ax1.tick_params(labelsize=18)


def plot_waveforms_interact(waveforms):
    """Launch interactive plotting widget."""
    interact(
        plot_waveforms,
        index=(0, len(waveforms) - 1, 1),
        waveforms=fixed(waveforms)
    )
