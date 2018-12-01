"""
evaluation.py
-------------
This module provides classes and functions for evaluating a model.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import itertools
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap


def plot_confusion_matrix(y_true, y_pred, classes, figure_size=(8, 8)):
    """This function plots a confusion matrix."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Build Laussen Labs colormap
    cmap = LinearSegmentedColormap.from_list('laussen_labs_green', ['w', '#43BB9B'], N=256)

    # Setup plot
    plt.figure(figsize=figure_size)

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    # Modify axes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(np.round(cm[i, j], 2)) + ' %', horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.ylabel('True Label', fontsize=25)
    plt.xlabel('Predicted Label', fontsize=25)

    plt.show()
