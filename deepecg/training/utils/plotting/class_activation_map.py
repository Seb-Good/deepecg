"""
plotting.py
-----------
This module provides classes and functions for visualizing data and neural networks.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import tensorflow as tf
from biosppy.signals import ecg
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
from biosppy.signals.tools import filter_signal


def plot_class_activation_map(model, index, time_series, labels, fs):

    """
    Plots one univariate time series

    Parameters
    ----------
    model : object
        Active model with live session
    index : int
        time series id
    time_series : np.array([m, length])
        image array
    labels : np.array([m,])
        a 1D array of length m training examples containing class labels
    fs : int
        sample frequency
    """

    # Label lookup
    label_lookup = ['Normal Sinus Rhythm', 'Atrial Fibrillation', 'Other Rhythm']

    # Get logits
    logits = model.sess.run(
        fetches=[model.graph.logits],
        feed_dict={
            model.graph.x: time_series[[index]],
            model.graph.y: labels[[index]],
            model.graph.is_training: False,
        }
    )

    # Get output conv
    conv = model.sess.run(
        fetches=[model.graph.net],
        feed_dict={
            model.graph.x: time_series[[index]],
            model.graph.y: labels[[index]],
            model.graph.is_training: False,
        }
    )

    # Get class activation map
    cam = model.sess.run(get_class_map(conv[0], np.squeeze(np.argmax(logits))))
    # cam = ((cam - cam.min()) / (cam.max() - cam.min()))
    cam = cam[0, :, 0]
    cam_time = np.arange(conv[0].shape[1]) / (conv[0].shape[1] / 60)

    # Get non-zero-pad indices
    non_zero_index = np.where(time_series[index, :, 0] != 0)[0]

    # Get non-zero-pad waveform
    time_series_filt = time_series[index, non_zero_index, 0]
    time_series_filt_ts = np.arange(time_series_filt.shape[0]) * 1 / fs

    # Linear interpolation
    cam_time_intrp = np.arange(time_series[index].shape[0]) * 1 / fs
    cam_intrp = np.interp(cam_time_intrp, cam_time, cam)

    # Get non-zero-pad cam
    cam_filt = cam_intrp[non_zero_index]

    # Setup figure
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(wspace=0, hspace=0)
    ax1 = plt.subplot2grid((2, 5), (0, 0), colspan=5)
    ax2 = plt.subplot2grid((2, 5), (1, 0), colspan=5)
    # ax3 = plt.subplot2grid((3, 5), (2, 1), colspan=3)

    prob = model.sess.run(tf.nn.softmax(logits[0]))

    # Set plot title
    ax1.set_title(
        'True Label: ' + label_lookup[np.squeeze(np.argmax(labels[index]))] + '\n' +
        'Predicted Label: ' + label_lookup[np.squeeze(np.argmax(logits))] + '\n' +
        'Normal Sinus Rhythm: ' + str(np.round(prob[0][0], 2)) +
        '     Atrial Fibrillation: ' + str(np.round(prob[0][1], 2)) +
        '     Other Rhythm: ' + str(np.round(prob[0][2], 2)),
        fontsize=20, y=1.03
    )

    # Plot image
    ax1.plot(time_series_filt_ts, time_series_filt, '-k', lw=1.5)

    # Axes labels
    ax1.set_ylabel('Normalized Amplitude', fontsize=22)
    ax1.set_xlim([0, time_series_filt_ts.max()])
    ax1.tick_params(labelbottom='off')
    ax1.yaxis.set_tick_params(labelsize=16)

    # Plot CAM
    ax2.plot(time_series_filt_ts, cam_filt, '-k', lw=1.5)

    # Axes labels
    ax2.set_xlabel('Time, seconds', fontsize=22)
    ax2.set_ylabel('Class Activation Map', fontsize=22)
    ax2.set_xlim([0, time_series_filt_ts.max()])
    # ax2.set_ylim([cam_filt.min()-0.05, cam_filt.max()+0.05])
    ax2.set_ylim([-3, 35])
    ax2.xaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)

    # # Get ecg object
    # ecg_object = ecg.ecg(time_series_filt, sampling_rate=fs, show=False)
    #
    # # Get waveform templates
    # templates, _ = _get_templates(time_series_filt, ecg_object['rpeaks'], 0.4, 0.6, fs)
    #
    # cam_filt, _, _ = filter_signal(signal=cam_filt,
    #                                ftype='FIR',
    #                                band='bandpass',
    #                                order=int(0.3 * fs),
    #                                frequency=[3, 100],
    #                                sampling_rate=fs)
    #
    # # Get cam templates
    # cam_templates, _ = _get_templates(cam_filt, ecg_object['rpeaks'], 0.4, 0.6, fs)
    #
    # ax3.plot(templates, '-', color=[0.7, 0.7, 0.7])
    # ax3.plot(np.median(templates, axis=1), '-k')
    #
    # ax3.set_ylim([-0.5, 1.5])
    #
    # ax4 = ax3.twinx()
    # ax4.plot(cam_templates, '-r', lw=0.25, alpha=0.5)
    # ax4.plot(np.mean(cam_templates, axis=1), '-r')
    #
    # ax4.set_ylim([np.median(cam_templates, axis=1).min()-0.02, np.median(cam_templates, axis=1).max()+0.02])

    plt.show()


def _get_templates(waveform, rpeaks, before, after, fs):

    # convert delimiters to samples
    before = int(before * fs)
    after = int(after * fs)

    # Sort R-Peaks in ascending order
    rpeaks = np.sort(rpeaks)

    # Get number of sample points in waveform
    length = len(waveform)

    # Create empty list for templates
    templates = []

    # Create empty list for new rpeaks that match templates dimension
    rpeaks_new = np.empty(0, dtype=int)

    # Loop through R-Peaks
    for rpeak in rpeaks:

        # Before R-Peak
        a = rpeak - before
        if a < 0:
            continue

        # After R-Peak
        b = rpeak + after
        if b > length:
            break

        # Append template list
        templates.append(waveform[a:b])

        # Append new rpeaks list
        rpeaks_new = np.append(rpeaks_new, rpeak)

    # Convert list to numpy array
    templates = np.array(templates).T

    return templates, rpeaks_new


def get_class_map(conv, label):
    output_channels = int(conv.shape[-1])

    with tf.variable_scope('logits', reuse=True):
        label_w = tf.gather(tf.transpose(tf.get_variable('kernel')), label)
        label_w = tf.reshape(label_w, [-1, output_channels, 1])

    classmap = tf.matmul(conv, label_w)

    return classmap


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def plot_class_activation_map_widget(model, time_series, labels, fs):

    """
    Launch interactive plotting widget.

    Parameters
    ----------
    model : object
        Active model with live session
    time_series : np.array([m, length])
        image array
    labels : np.array([m,])
        a 1D array of length m training examples containing class labels
    fs : int
        sample frequency
    """

    # Launch interactive widget
    interact(
        plot_class_activation_map,
        model=fixed(model),
        index=(0, labels.shape[0]-1, 1),
        time_series=fixed(time_series),
        labels=fixed(labels),
        fs=fixed(fs)
    )
