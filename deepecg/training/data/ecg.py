"""
ecg.py
------
This module provides classes and functions for processing an ECG waveform.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal
from sklearn.preprocessing import StandardScaler


class ECG(object):

    # Label lookup
    label_lookup = {'N': 0, 'A': 1, 'O': 2, '~': 3}

    def __init__(self, file_name, label, waveform, filter_bands, fs):

        # Set parameters
        self.file_name = file_name
        self.label_str = label
        self.waveform = waveform
        self.filter_bands = filter_bands
        self.fs = fs

        # Set attributes
        self.label_int = self.label_lookup[label]
        self.time = np.arange(len(self.waveform)) * 1 / self.fs
        self.length = self._get_waveform_length(waveform=self.waveform)
        self.duration = self._get_waveform_duration(waveform=self.waveform)
        self.filtered = None
        self.templates = None
        self.rpeaks_ps = None
        self.rpeaks_ts = None
        self.rpeak_count = None

        # Scale waveform
        self._scale_amplitude()

        # Get rpeaks
        self.rpeaks_ps = self._get_rpeaks()

        # Filter waveform
        self.filtered = self._filter_waveform()
        
        # Get templates
        self.templates, self.rpeaks_ps = self._get_templates(waveform=self.filtered, rpeaks=self.rpeaks_ps,
                                                             before=0.25, after=0.4)

        # Get rpeaks time array
        self.rpeaks_ts = self._get_rpeaks_time_array()

        # Get rpeak count
        self.rpeak_count = len(self.rpeaks_ps)

        # Check polarity
        self._polarity_check()

        # Normalize waveforms
        self._normalize()

    def get_dictionary(self):
        """Return a dictionary of processed ECG waveforms and features."""
        return {'label_str': self.label_str, 'label_int': self.label_int, 'time': self.time, 'waveform': self.waveform,
                'filtered': self.filtered, 'templates': self.templates, 'rpeak_count': self.rpeak_count,
                'rpeaks_ps': self.rpeaks_ps, 'rpeaks_ts': self.rpeaks_ts, 'length': self.length,
                'duration': self.duration}

    def _scale_amplitude(self):
        """Scale amplitude to values with a mean of zero and standard deviation of 1."""
        # Get scaler object
        scaler = StandardScaler()

        # Fit scaler with finite data
        scaler = scaler.fit(self.waveform.reshape(-1, 1))

        # Scale signal
        self.waveform = scaler.transform(self.waveform.reshape(-1, 1)).reshape(-1,)

    def _get_rpeaks(self):
        """Hamilton-Tompkins r-peak detection."""
        # Get BioSPPy ecg object
        ecg_object = ecg.ecg(signal=self.waveform, sampling_rate=self.fs, show=False)

        return ecg_object['rpeaks']

    def _get_rpeaks_time_array(self):
        """Get an array of r-peak times."""
        return self.rpeaks_ps * 1 / self.fs

    def _filter_waveform(self):
        """Filter raw ECG waveform with bandpass finite-impulse-response filter."""
        # Calculate filter order
        order = int(0.3 * self.fs)

        # Filter waveform
        filtered, _, _ = filter_signal(signal=self.waveform, ftype='FIR', band='bandpass', order=order,
                                       frequency=self.filter_bands, sampling_rate=self.fs)

        return filtered
        
    def _get_templates(self, waveform, rpeaks, before, after):
        """Extract waveform PQRST-templates."""
        # convert delimiters to samples
        before = int(before * self.fs)
        after = int(after * self.fs)

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

    def _polarity_check(self):
        """Correct for inverted polarity."""
        # Get extremes of median templates
        templates_min = np.min(np.median(self.templates, axis=1))
        templates_max = np.max(np.median(self.templates, axis=1))

        if np.abs(templates_min) > np.abs(templates_max):

            # Flip polarity
            self.waveform *= -1
            self.filtered *= -1
            self.templates *= -1

    def _normalize(self):
        """Normalize waveform to median r-peak amplitude."""
        # Get median templates max
        templates_max = np.max(np.median(self.templates, axis=1))

        # Normalize ecg signals
        self.waveform /= templates_max
        self.filtered /= templates_max
        self.templates /= templates_max

    @staticmethod
    def _get_waveform_length(waveform):
        """Get waveform length in sample points."""
        return len(waveform)

    def _get_waveform_duration(self, waveform):
        """Get waveform duration in seconds."""
        return len(waveform) * 1 / self.fs



