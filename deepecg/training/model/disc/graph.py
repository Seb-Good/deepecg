"""
graph.py
--------
This module provide a class and methods for building a computational graph with tensorflow.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import tensorflow as tf

# Local imports
from deepecg.training.utils.devices.device_check import get_device_count


class Graph(object):

    """Computational graph class."""

    def __init__(self, network, save_path, data_path, max_to_keep):

        # Set input parameters
        self.network = network          # network: neural network architecture
        self.save_path = save_path      # save_path: checkpoints, summaries, and graphs
        self.data_path = data_path      # data_path: waveforms, labels
        self.max_to_keep = max_to_keep  # Maximum number of checkpoints to keep

        # Set attributes
        self.waveforms = None
        self.labels = None
        self.is_training = None
        self.learning_rate = None
        self.global_step = None
        self.logits = None
        self.loss = None
        self.accuracy = None
        self.f1 = None
        self.optimizer = None
        self.train_summary_all_op = None
        self.val_cam_plots_summary_op = None
        self.train_summary_metrics_op = None
        self.init_global = None
        self.saver = None
        self.gradients = None
        self.train_op = None
        self.gpu_count = None
        self.metrics = None
        self.mini_batch_size = None
        self.update_metrics_op = None
        self.init_metrics_op = None
        self.batch_size = None
        self.mode_handle = None
        self.generator_train = None
        self.generator_val = None
        self.generator_test = None
        self.iterator = None
        self.cams = None
        self.tower_losses = None
        self.tower_accuracies = None
        self.tower_f1 = None
        self.tower_gradients = None
        self.tower_waveforms = None
        self.tower_labels = None
        self.tower_logits = None
        self.tower_cam = None
        self.val_cam_plots = None

        # Build computational graph
        self.build_graph()

    def build_graph(self):
        """Constructs a computational graph for training and evaluation."""
        """Setup"""
        # Reset graph
        tf.reset_default_graph()

        # Create CPU device context
        with tf.device('/cpu:0'):

            # Create placeholders
            self.is_training, self.learning_rate, self.batch_size, self.mode_handle, self.val_cam_plots = \
                self._create_placeholders()

            # Get or create global step
            self.global_step = tf.train.get_or_create_global_step()

            # Get number of GPUs
            self.gpu_count = get_device_count(device_type='GPU')

            # Initialize optimizer
            self.optimizer = self._create_optimizer(learning_rate=self.learning_rate)

            # Data train, val, and test data generators
            self.generator_train, self.generator_val, self.generator_test = self._get_generators()

            # Initialize iterator
            self.iterator = self._initialize_iterator()

            """Compute Forward Pass"""
            self._build_forward_graph()

            """Training"""
            # Run training operation
            self.train_op = self._run_optimization_step(optimizer=self.optimizer, gradients=self.gradients,
                                                        global_step=self.global_step)

            """Metrics"""
            # Compute loss
            self.metrics = self._compute_metrics()

            # Update metrics
            self.update_metrics_op = self._update_metrics()

            # Initialize metrics
            self.init_metrics_op = self._initialize_metrics()

            """Summaries"""
            # Merge training summaries
            self.train_summary_metrics_op = tf.summary.merge_all('train_metrics')
            self.val_cam_plots_summary_op = tf.summary.image(name='val', tensor=self.val_cam_plots, max_outputs=256)

            """Initialize Variables"""
            # Initialize global variables
            self.init_global = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            """Checkpoint Saver"""
            # Initialize saver
            self.saver = self._initialize_saver()

    def _build_forward_graph(self):
        """Build the portion of the graph that computes logits, loss, and gradients."""
        if self.gpu_count <= 1:
            # Sequential graph
            self._build_sequential_forward_graph()

        elif self.gpu_count > 1:
            # Parallel graph
            self._build_parallel_forward_graph()

    def _build_sequential_forward_graph(self):
        """Build sequential forward graph for training on CPU or single GPU."""
        # Get mini-batch
        self.waveforms, self.labels = self._get_next_batch()

        # Compute forward propagation
        self.logits = self.network.inference(input_layer=self.waveforms, is_training=self.is_training)

        # Compute loss
        self.loss = self._compute_loss(logits=self.logits, labels=self.labels)

        # Compute accuracy
        self.accuracy = self._compute_accuracy(logits=self.logits, labels=self.labels)

        # Compute gradients
        self.gradients = self._compute_gradients(optimizer=self.optimizer, loss=self.loss)

    def _build_parallel_forward_graph(self):
        """Build parallel forward graph for training on more than one GPU."""
        # Initialize tower lists
        self.tower_losses = list()
        self.tower_accuracies = list()
        self.tower_f1 = list()
        self.tower_gradients = list()
        self.tower_waveforms = list()
        self.tower_labels = list()
        self.tower_logits = list()
        self.tower_cams = list()

        # Loop through GPUs and build forward graph for each one
        for tower_id in range(self.gpu_count):
            with tf.device(self._assign_vars_to_cpu(device='/gpu:{}'.format(tower_id))):
                with tf.name_scope('tower_{}'.format(tower_id)) as name_scope:

                    # Get mini-batch
                    waveforms, labels = self._get_next_batch()

                    # Compute inference
                    logits, cams = self.network.inference(input_layer=waveforms, reuse=tf.AUTO_REUSE,
                                                          is_training=self.is_training, name='ECGNet',
                                                          print_shape=True)

                    # Compute loss
                    loss = self._compute_loss(logits=logits, labels=labels)

                    # Compute accuracy
                    accuracy = self._compute_accuracy(logits=logits, labels=labels)

                    # Compute f1
                    f1 = self._compute_f1(logits=logits, labels=labels)

                    # Compute gradients
                    gradients = self._compute_gradients(optimizer=self.optimizer, loss=loss)

                    # Append losses
                    self.tower_losses.append(loss)

                    # Append accuracy
                    self.tower_accuracies.append(accuracy)

                    # Append f1
                    self.tower_f1.append(f1)

                    # Append gradients
                    self.tower_gradients.append(gradients)

                    # Append waveforms, labels, and logits
                    self.tower_waveforms.append(waveforms)
                    self.tower_labels.append(labels)
                    self.tower_logits.append(logits)
                    self.tower_cams.append(cams)

                    # Trigger batch_norm moving mean and variance update operation
                    if tower_id == 0:
                        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

        # Merge towers
        self.loss = self._compute_mean_loss(tower_losses=self.tower_losses)
        self.accuracy = self._compute_mean_accuracy(tower_accuracies=self.tower_accuracies)
        self.f1 = self._compute_mean_f1(tower_f1=self.tower_f1)
        self.gradients = self._compute_mean_gradients(tower_gradients=self.tower_gradients)
        self._group_data()

    def _group_data(self):
        """Group data (waveforms, labels, logits) from multiple GPUs."""
        with tf.variable_scope('group_data'):
            self.waveforms = tf.concat(self.tower_waveforms, axis=0)
            self.labels = tf.concat(self.tower_labels, axis=0)
            self.logits = tf.concat(self.tower_logits, axis=0)
            self.cams = tf.concat(self.tower_cams, axis=0)

    def _compute_metrics(self):
        """Collect loss metric."""
        with tf.variable_scope('metrics'):
            metrics = {'accuracy': tf.metrics.mean(values=self.accuracy),
                       'f1': tf.metrics.mean(values=self.f1),
                       'loss': tf.metrics.mean(values=self.loss)}
        return metrics

    def _update_metrics(self):
        """Collect metrics (loss)."""
        return tf.group(*[op for _, op in self.metrics.values()])

    def _initialize_saver(self):
        """Initialize tf saver."""
        with tf.variable_scope('saver'):
            return tf.train.Saver(max_to_keep=self.max_to_keep)

    def _initialize_iterator(self):
        """Initialize the iterator from a mode handle placeholder."""
        with tf.variable_scope('iterator'):
            iterator = tf.data.Iterator.from_string_handle(self.mode_handle, self.generator_train.dataset.output_types,
                                                           self.generator_train.dataset.output_shapes)
        return iterator

    def _get_next_batch(self):
        """Get next batch (waveforms, labels) from iterator."""
        with tf.name_scope('next_batch'):
            waveforms, labels = self.iterator.get_next()

        return waveforms, labels

    def _get_generators(self):
        """Create train, val, and test data generators."""
        with tf.variable_scope('train_generator'):
            generator_train = self.network.create_generator(path=self.data_path, mode='train',
                                                            batch_size=self.batch_size)
        with tf.variable_scope('val_generator'):
            generator_val = self.network.create_generator(path=self.data_path, mode='val',
                                                          batch_size=self.batch_size)
        with tf.variable_scope('test_generator'):
            generator_test = self.network.create_generator(path=self.data_path, mode='test',
                                                           batch_size=self.batch_size)
        return generator_train, generator_val, generator_test

    def _get_saver(self):
        """Create tensorflow checkpoint saver."""
        with tf.variable_scope('saver'):
            return tf.train.Saver(max_to_keep=self.max_to_keep)

    @staticmethod
    def _assign_vars_to_cpu(device):
        """Assign all variables to CPU."""
        # Variable groups to assign to CPU
        ps_ops = ['Variable', 'VariableV2', 'AutoReloadVariable']

        # Variable assign function
        def _assign(op):
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            if node_def.op in ps_ops:
                return '/cpu:0'
            else:
                return device

        return _assign

    @staticmethod
    def _create_optimizer(learning_rate):
        """Create ADAM optimizer instance."""
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Get learning rate summary
        tf.summary.scalar(name='learning_rate/learning_rate', tensor=learning_rate, collections=['train_metrics'])

        return optimizer

    @staticmethod
    def _compute_loss(logits, labels):
        """Computes the mean squared error for a given set of logits and labels."""
        with tf.variable_scope('loss'):
            # Specify class weightings {'N': 0.42001576, 'A': 2.81266491, 'O': 0.88281573, '~': 7.64157706}
            class_weights = tf.constant([0.42001576, 2.81266491, 0.88281573, 1.0])

            # Specify the weights for each sample in the batch
            # weights = tf.gather(params=class_weights, indices=tf.cast(labels, tf.int32))

            # compute the loss
            losses = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=tf.cast(labels, tf.int32))
            
            # Compute mean loss
            loss = tf.reduce_mean(losses)

        return loss

    def _compute_accuracy(self, logits, labels):
        """Computes the accuracy set of logits and labels."""
        with tf.variable_scope('accuracy'):
            accuracy = self.network.compute_accuracy(logits=logits, labels=labels)

        return accuracy

    def _compute_f1(self, logits, labels):
        """Computes the f1 score set of logits and labels."""
        with tf.variable_scope('f1'):
            f1 = self.network.compute_f1(logits=logits, labels=labels)

        return f1

    @staticmethod
    def _compute_gradients(optimizer, loss):
        """Computes the model gradients for given loss."""
        return optimizer.compute_gradients(loss=loss, var_list=None)

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
        tf.summary.scalar(name='loss/loss', tensor=loss, collections=['train_metrics'])

        return loss

    @staticmethod
    def _compute_mean_accuracy(tower_accuracies):
        """Compute mean accuracy across towers."""
        with tf.variable_scope('accuracy'):
            accuracy = tf.reduce_mean(tower_accuracies)

        # Get summary
        tf.summary.scalar(name='accuracy/accuracy', tensor=accuracy, collections=['train_metrics'])

        return accuracy

    @staticmethod
    def _compute_mean_f1(tower_f1):
        """Compute mean f1 score across towers."""
        with tf.variable_scope('f1'):
            f1 = tf.reduce_mean(tower_f1)

        # Get summary
        tf.summary.scalar(name='f1/f1', tensor=f1, collections=['train_metrics'])

        return f1

    @staticmethod
    def _compute_mean_gradients(tower_gradients):
        """Compute mean gradients across towers."""
        with tf.name_scope('mean_gradients'):
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

    @staticmethod
    def _create_placeholders():
        """Creates place holders: x, y, is_training, and learning_rate."""
        with tf.variable_scope('is_training') as scope:
            is_training = tf.placeholder_with_default(True, shape=(), name=scope.name)

        with tf.variable_scope('learning_rate') as scope:
            learning_rate = tf.placeholder(dtype=tf.float32, name=scope.name)

        with tf.variable_scope('batch_size') as scope:
            batch_size = tf.placeholder(dtype=tf.int64, name=scope.name)

        with tf.variable_scope('mode_handle') as scope:
            mode_handle = tf.placeholder(dtype=tf.string, shape=[], name=scope.name)

        with tf.variable_scope('val_cam_plots') as scope:
            val_cam_plots = tf.placeholder(dtype=tf.uint8, shape=[None, None, None, 4], name=scope.name)

        return is_training, learning_rate, batch_size, mode_handle, val_cam_plots
