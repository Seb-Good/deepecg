"""
model.py
--------
This module provides a class and methods for building and managing a model with tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import sys
import json
import pickle
import tensorflow as tf

# Local imports
from deepecg.training.model.memory.graph import Graph


class Model(object):

    """A class for managing a model through training."""

    def __init__(self, model_name, network_name, network_parameters, save_path, max_to_keep):

        # Set input parameters
        self.model_name = model_name
        self.network_name = network_name
        self.network_parameters = network_parameters
        self.save_path = os.path.join(save_path, self.model_name)
        self.max_to_keep = max_to_keep

        # Set attributes
        self.sess = None
        self.graph = None
        self.network = None

        # Create project file structure
        self._create_folder_structure()

        # Save parameters
        self._save_parameters()

        # Initialize graph
        self.initialize_graph()

    def initialize_graph(self):

        # Get neural network
        self.network = self._get_neural_network()

        # Build computational graph
        self.graph = Graph(network=self.network, save_path=self.save_path, max_to_keep=self.max_to_keep)

        # Start session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # Initialize global variables
        self.sess.run(self.graph.init_global)

        # Save network object
        self._pickle_network()

    @classmethod
    def build_training_graph(cls, save_path):
        """Build training graph."""
        # Import model parameters
        model_parameters = cls._import_model_parameters(save_path=save_path)

        # Import network parameters
        network_parameters = cls._import_network_parameters(save_path=save_path)

        # Initialize Model
        return cls(model_name=model_parameters['model_name'], network_name=model_parameters['network_name'],
                   network_parameters=network_parameters,  save_path=os.path.dirname(save_path),
                   max_to_keep=model_parameters['max_to_keep'])

    def restore(self, global_step):
        """Restore model from checkpoint."""
        # Initialize graph
        if self.sess._closed:
            self.initialize_graph()

        # Restore checkpoint
        self.graph.saver.restore(sess=self.sess, save_path=os.path.join(self.save_path, 'checkpoints', global_step))

    def close_session(self):
        """Close any active sessions."""
        try:
            self.sess.close()
        except AttributeError:
            print('No active Tensorflow session.')

    def _save_parameters(self):
        """Save model and network parameters to JSON."""
        # Save model parameters
        self._save_model_parameters()

        # Save network parameters
        self._save_network_parameters()

    def _save_model_parameters(self):
        """Save model parameters to JSON."""
        # Get model parameters
        model_parameters = dict(model_name=self.model_name, network_name=self.network_name,
                                max_to_keep=self.max_to_keep)

        # Save model parameters to JSON
        if not os.path.exists(os.path.join(self.save_path, 'parameters', 'model_parameters.json')):
            with open(os.path.join(self.save_path, 'parameters', 'model_parameters.json'), 'w') as file:
                json.dump(model_parameters, file)

    def _save_network_parameters(self):
        """Save network parameters to JSON."""
        if not os.path.exists(os.path.join(self.save_path, 'parameters', 'network_parameters.json')):
            with open(os.path.join(self.save_path, 'parameters', 'network_parameters.json'), 'w') as file:
                json.dump(self.network_parameters, file)

    def _get_neural_network(self):
        """Instantiate neural network."""
        # Convert string to class
        network = getattr(sys.modules[__name__], self.network_name)

        # Instantiate network class with network parameters
        network = network(**self.network_parameters)

        return network

    def _create_folder_structure(self):

        # Set list of folders
        folders = ['training', 'validation', 'checkpoints', 'network', 'graph', 'logs', 'parameters']

        # Main project directory
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Loop through and create project folders
        for folder in folders:
            self._create_folder(folder=folder)

    def _create_folder(self, folder):
        """Create folder."""
        if not os.path.exists(os.path.join(self.save_path, folder)):
            os.makedirs(os.path.join(self.save_path, folder))

    def _pickle_network(self):
        """Pickle graph."""
        with open(os.path.join(self.save_path, 'network', 'network.obj'), 'wb') as file:
            pickle.dump(obj=self.network, file=file)

    @staticmethod
    def _import_model_parameters(save_path):
        """Import model parameters."""
        with open(os.path.join(save_path, 'parameters', 'model_parameters.json')) as file:
            return json.load(file)

    @staticmethod
    def _import_network_parameters(save_path):
        """Import network parameters."""
        with open(os.path.join(save_path, 'parameters', 'network_parameters.json')) as file:
            return json.load(file)
