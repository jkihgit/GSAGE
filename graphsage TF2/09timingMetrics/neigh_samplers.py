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
        def fillTemplate(template):
            i = 0
            for rank in template:
                for f in range(len(rank)):
                    rank[f] = i
                i += 1
            return template
        # only works for 2 layered model
        # should be easy to expend to any number of layers
        ranktemplate = [0,0]
        ranktemplate[1] = np.zeros((FLAGS.batch_size, FLAGS.samples_2, 1), dtype=float)
        ranktemplate[0] = np.zeros((FLAGS.batch_size*FLAGS.samples_2, FLAGS.samples_1, 1), dtype=float)
        for template in ranktemplate:
            template = fillTemplate(template)
        self.ranktemplate = ranktemplate

    def _call(self, inputs):
        return self.randomfix(inputs)

    def randomfix(self, inputs):
        ids, num_samples, layer = inputs
        layer = layer.eval(session=tf.compat.v1.Session())
        adj_lists = tf.nn.embedding_lookup(params=self.adj_info, ids=tf.dtypes.cast(ids, tf.int32)) 
        adj_lists = tf.reshape(adj_lists, [-1, FLAGS.max_degree])
        randpicks = tf.random.uniform([tf.shape(ids)[0], num_samples, 1], 0, FLAGS.max_degree, dtype=tf.int32)
        template = tf.slice(self.ranktemplate[layer], [0,0,0], tf.shape(randpicks))
        randpicks = tf.concat([tf.cast(template, dtype=tf.int32), randpicks], axis=2)
        adj_lists = tf.gather_nd(adj_lists, randpicks)
        adj_lists = tf.reshape(adj_lists, [-1])
        return adj_lists

    def vanilla(self, inputs):
        ids, num_samples, _ = inputs
        adj_lists = tf.nn.embedding_lookup(params=self.adj_info, ids=tf.dtypes.cast(ids, tf.int32)) 
        adj_lists = tf.reshape(adj_lists, [-1, FLAGS.max_degree])
        adj_lists = tf.transpose(a=adj_lists) #transpose1
        adj_lists = tf.random.shuffle(adj_lists)
        adj_lists = tf.transpose(a=adj_lists) #transpose2
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        adj_lists = tf.reshape(adj_lists, [-1])
        return adj_lists
