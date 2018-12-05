"""
graph.py
--------
This module provide a class and methods for building a computational graph with tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import tensorflow as tf

# Local imports
from utils.devices.device_check import get_device_count


class Graph(object):

    """Computational graph class."""

    def __init__(self, network, save_path, max_to_keep):

        # Set input parameters
        self.network = network          # network : neural network architecture
        self.save_path = save_path      # save_path : checkpoints, summaries, and graphs
        self.max_to_keep = max_to_keep  # Maximum number of checkpoints to keep

        # Set attributes
        self.x = None
        self.y = None
        self.is_training = None
        self.learning_rate = None
        self.global_step = None
        self.logits = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        self.train_summary_full_op = None
        self.train_summary_metrics_op = None
        self.init_global = None
        self.saver = None
        self.net = None
        self.var_list = None
        self.gradients = None
        self.train_op = None
        self.gpu_count = None
        self.metrics = None
        self.mini_batch_size = None
        self.update_metrics_op = None
        self.init_metrics_op = None

        # Build computational graph
        self.build_graph()

    def save_graph(self, sess):
        """Save computational graph."""
        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir=os.path.join(self.save_path, 'graph'),
                             name='graph.pbtxt', as_text=True)

    def build_graph(self):
        """Constructs a computational graph for training and evaluation."""
        """Setup"""
        # Reset graph
        tf.reset_default_graph()

        # Create placeholders
        self.x, self.y, self.is_training, self.learning_rate, self.mini_batch_size = self._create_placeholders()

        # Get or create global step
        self.global_step = tf.train.get_or_create_global_step()

        # Get number of GPUs
        self.gpu_count = get_device_count(device_type='GPU')

        # Initialize optimizer
        self.optimizer = self._create_optimizer(learning_rate=self.learning_rate)

        """Compute Forward Pass"""
        self._build_forward_graph()

        """Training"""
        # Run training operation
        self.train_op = self._run_optimization_step(optimizer=self.optimizer, gradients=self.gradients,
                                                    global_step=self.global_step)

        """Metrics"""
        # Compute loss and accuracy
        self.metrics = self._compute_metrics()

        # Update metrics
        self.update_metrics_op = self._update_metrics()

        # Initialize metrics
        self.init_metrics_op = self._initialize_metrics()

        """Summaries"""
        # Merge training summaries
        self.train_summary_full_op = tf.summary.merge_all('train_full')
        self.train_summary_metrics_op = tf.summary.merge_all('train_metrics')

        """Initialize Variables"""
        # Initialize global variables
        self.init_global = tf.global_variables_initializer()

        """Checkpoint Saver"""
        # Initialize saver
        self.saver = self._initialize_saver()

    def _create_placeholders(self):
        """Creates place holders: x, y, is_training, and learning_rate."""
        x, y = self.network.create_placeholders()

        with tf.variable_scope('is_training') as scope:
            is_training = tf.placeholder_with_default(True, shape=(), name=scope.name)

        with tf.variable_scope('learning_rate') as scope:
            learning_rate = tf.placeholder(dtype=tf.float32, name=scope.name)

        with tf.variable_scope('mini_batch_size') as scope:
            mini_batch_size = tf.placeholder(dtype=tf.int32, name=scope.name)

        return x, y, is_training, learning_rate, mini_batch_size

    def _build_forward_graph(self):
        """Build the portion of the graph that computes logits, loss, accuracy, and gradients."""
        if self.gpu_count <= 1:
            # Sequential graph
            self._build_sequential_forward_graph()

        elif self.gpu_count > 1:
            # Parallel graph
            self._build_parallel_forward_graph()

    def _build_sequential_forward_graph(self):
        """Build sequential forward graph for training on CPU or single GPU."""
        # Compute forward propagation
        self.logits, self.net = self.network.inference(input_layer=self.x, is_training=self.is_training)

        # Compute loss
        self.loss = self._compute_loss(logits=self.logits, labels=self.y)

        # Compute accuracy
        self.accuracy = self.network.compute_accuracy(logits=self.logits, labels=self.y)

        # Compute gradients
        self.gradients = self._compute_gradients(optimizer=self.optimizer, loss=self.loss, var_list=None)

    def _build_parallel_forward_graph(self):
        """Build parallel forward graph for training on more than one GPU."""
        # Initialize tower lists
        self.tower_losses = list()
        self.tower_accuracies = list()
        self.tower_gradients = list()
        self.tower_x = list()
        self.tower_y = list()
        self.tower_logits = list()

        # Loop through GPUs and build forward graph for each one
        for tower_id in range(self.gpu_count):
            with tf.device(self._assign_vars_to_cpu(device='/gpu:{}'.format(tower_id))):
                with tf.name_scope('tower_{}'.format(tower_id)) as name_scope:

                    print('/gpu:{}'.format(tower_id))

                    # Get mini-batch
                    x, y = self._get_mini_batch(tower_id=tower_id)

                    # Compute inference
                    logits, net = self.network.inference(input_layer=x, is_training=self.is_training)

                    # Compute loss
                    loss = self._compute_loss(logits=logits, labels=y)

                    # Compute accuracy
                    accuracy = self._compute_accuracy(logits=logits, labels=y)

                    # Compute gradients
                    gradients = self._compute_gradients(optimizer=self.optimizer, loss=loss, var_list=None)

                    # Append losses
                    self.tower_losses.append(loss)

                    # Append accuracies
                    self.tower_accuracies.append(accuracy)

                    # Append gradients
                    self.tower_gradients.append(gradients)

                    # Append images, labels, and logits
                    self.tower_x.append(x)
                    self.tower_y.append(y)
                    self.tower_logits.append(logits)

                    # Trigger batch_norm moving mean and variance update operation
                    if tower_id == 0:
                        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

                    # Get output from final convolutional layer
                    if tower_id == 0:
                        self.net = net

        # Merge towers
        self.loss = self._compute_mean_loss(tower_losses=self.tower_losses)
        self.accuracy = self._compute_mean_accuracy(tower_accuracies=self.tower_accuracies)
        self.gradients = self._compute_mean_gradients(tower_gradients=self.tower_gradients)
        self.x = tf.stack(self.tower_x, axis=0)
        self.y = tf.stack(self.tower_y, axis=0)
        self.logits = tf.stack(self.tower_logits, axis=0)

    def _get_mini_batch(self, tower_id):
        """Get mini-batch from larger feed_dict batch."""
        return (self.x[tower_id * self.mini_batch_size:(tower_id + 1) * self.mini_batch_size],
                self.y[tower_id * self.mini_batch_size:(tower_id + 1) * self.mini_batch_size])

    def _compute_metrics(self):
        """Collect loss and accuracy metrics."""
        with tf.variable_scope('train_metrics'):
            metrics = {'loss': tf.metrics.mean(values=self.loss), 'accuracy': tf.metrics.mean(values=self.accuracy)}
        return metrics

    def _update_metrics(self):
        """Collect metrics (loss and accuracy)."""
        return tf.group(*[op for _, op in self.metrics.values()])

    def _initialize_saver(self):
        """Initialize tf saver."""
        with tf.variable_scope('saver'):
            return tf.train.Saver(max_to_keep=self.max_to_keep)

    @staticmethod
    def _assign_vars_to_cpu(device):
        """Assign all variables to CPU."""
        # Variable groups to assign to CPU
        ps_ops = ['Variable', 'VariableV2', 'AutoReloadVariable']

        print('CPU Assign')

        # Variable assign function
        def _assign(op):
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            if node_def.op in ps_ops:
                return '//cpu:0'
            else:
                return device

        return _assign

    @staticmethod
    def _create_optimizer(learning_rate):
        """Create ADAM optimizer instance."""
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9,
                                               beta2=0.999, epsilon=1e-08,)

        # Get learning rate summary
        tf.summary.scalar(name='learning_rate/learning_rate', tensor=learning_rate,
                          collections=['train_metrics', 'train_full'])

        return optimizer

    @staticmethod
    def _compute_loss(logits, labels):
        """Computes the cross-entropy loss for a given set of logits (predicted) and labels (true)."""
        with tf.variable_scope('loss'):
            # Specify class weightings {'N': 0.5416995, 'A': 3.62752858, 'O': 1.13857833}
            class_weights = tf.constant([0.5416995, 3.62752858, 1.13857833])

            # Specify the weights for each sample in the batch
            weights = tf.reduce_sum(class_weights * labels, axis=1)

            # compute the loss
            losses = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, weights=weights)

            # Compute mean loss
            loss = tf.reduce_mean(losses)

        return loss

    def _compute_accuracy(self, logits, labels):
        """Computes the accuracy for a given set of logits (predicted) and labels (true)."""
        with tf.variable_scope('accuracy'):
            return self.network.compute_accuracy(logits=logits, labels=labels)

    @staticmethod
    def _compute_gradients(optimizer, loss, var_list):
        """Computes the model gradients for given loss."""
        return optimizer.compute_gradients(loss=loss, var_list=var_list)

    @staticmethod
    def _run_optimization_step(optimizer, gradients, global_step):
        """Computes one optimization step."""
        # Update operation for Batch Norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('train_op'):
            with tf.control_dependencies(update_ops):
                return optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

    @staticmethod
    def _compute_mean_loss(tower_losses):
        """Compute mean loss across towers."""
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tower_losses)

        # Get summary
        tf.summary.scalar(name='loss/loss', tensor=loss, collections=['train_metrics', 'train_full'])

        return loss

    @staticmethod
    def _compute_mean_accuracy(tower_accuracies):
        """Compute mean accuracy across towers."""
        with tf.variable_scope('accuracy'):
            accuracy = tf.reduce_mean(tower_accuracies)

        # Get summary
        tf.summary.scalar(name='accuracy/accuracy', tensor=accuracy, collections=['train_metrics', 'train_full'])

        return accuracy

    @staticmethod
    def _compute_mean_gradients(tower_gradients):
        """Compute mean gradients across towers."""
        average_grads = []
        for grad_and_vars in zip(*tower_gradients):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            # Average over the 'tower' dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    @staticmethod
    def _initialize_metrics():
        """Initialize stream metrics."""
        metric_vars = tf.get_collection(key=tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
        return tf.variables_initializer(var_list=metric_vars)
