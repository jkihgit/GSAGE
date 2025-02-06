from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import numpy as np
import tensorflow as tf
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(params=self.adj_info, ids=tf.dtypes.cast(ids, tf.int32)) 
        adj_lists = tf.reshape(adj_lists, [-1, FLAGS.max_degree])
        adj_lists = tf.transpose(a=adj_lists) #transpose1
        adj_lists = tf.random.shuffle(adj_lists)
        adj_lists = tf.transpose(a=adj_lists) #transpose2
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        adj_lists = tf.reshape(adj_lists, [-1])
        return adj_lists
