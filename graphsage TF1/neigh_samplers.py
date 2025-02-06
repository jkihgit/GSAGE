from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
## import nvtx.plugins.tf as nvtx_tf
flags = tf.app.flags
FLAGS = flags.FLAGS


import time

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
        # init sampling sizes
        # 2 layers
        # 1 * maxDegree
        # samples2 * maxDegree
        if FLAGS.randomizer_fix:
            import numpy as np
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
            ranktemplate[1] = np.zeros((FLAGS.batch_size, FLAGS.samples_2, 1), dtype=np.float32)
            ranktemplate[0] = np.zeros((FLAGS.batch_size*FLAGS.samples_2, FLAGS.samples_1, 1), dtype=np.float32)
            for i in range(len(ranktemplate)):
                ranktemplate[i] = tf.constant(fillTemplate(ranktemplate[i]), dtype=tf.float32)
            self.ranktemplate = ranktemplate

    ## @nvtx_tf.ops.trace(message='Sampling Block', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvt, trainable=True)
    # trainable=True required for first op
    def _call(self, inputs):
        if not (FLAGS.lookup_fix or FLAGS.randomizer_fix):
            return self.vanilla(inputs)
        elif (FLAGS.lookup_fix and not FLAGS.randomizer_fix):
            return self.lookupfixOnly(inputs)
        elif ((not FLAGS.lookup_fix) and FLAGS.randomizer_fix):
            return self.randomfixOnly(inputs)
        elif (FLAGS.lookup_fix and FLAGS.randomizer_fix):
            return self.lookuprandomFix(inputs)
        else:
            assert False

    def lookuprandomFix(self, inputs):
        with tf.device('/gpu:0'):
            ids, num_samples, layer = inputs
            adj_lists = self.randomfixOnly((tf.cast(ids, tf.int32), num_samples, layer))
        return adj_lists

    def randomfixOnly(self, inputs):
        # need to shuffle each row
        # ideal:
        # 1) get random weight for all maxDegree neighbors
        # 2) sort by that weight
        # implementation:
        # 1) get random int [0, maxDegree] for all maxDegree neighbors
        # 2) gather
        ids, num_samples, layer = inputs
        #self.adj_info, nvtx_ctx = nvtx_tf.ops.start(self.adj_info, message='Sampling Lookup', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvtx)
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids, name='sampembed')
        #adj_lists = nvtx_tf.ops.end(adj_lists, nvtx_ctx)
        # random int [0, maxDegree]
        #ids, nvtx_ctx = nvtx_tf.ops.start(ids, message='Sampling Randomize', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvtx)
        randpicks = tf.random.uniform([tf.shape(ids)[0], num_samples, 1], 0, FLAGS.max_degree, dtype=tf.int32, name='samprand')
        #randpicks = nvtx_tf.ops.end(randpicks, nvtx_ctx)

        #randpicks, nvtx_ctx = nvtx_tf.ops.start(randpicks, message='Sampling Slice', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvtx)
        template = tf.slice(self.ranktemplate[layer], [0,0,0], tf.shape(randpicks))
        #template = nvtx_tf.ops.end(template, nvtx_ctx)

        #template, nvtx_ctx = nvtx_tf.ops.start(template, message='Sampling Concat', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvtx)
        randpicks = tf.concat([template, tf.cast(randpicks, dtype=tf.float32)], axis=2, name='sampconcat')
        randpicks = tf.cast(randpicks, dtype=tf.int32)
        #randpicks = nvtx_tf.ops.end(randpicks, nvtx_ctx)

        #adj_lists, nvtx_ctx = nvtx_tf.ops.start(adj_lists, message='Sampling Gather', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvtx)
        adj_lists = tf.gather_nd(adj_lists, randpicks, name='sampgather')
        #adj_lists = nvtx_tf.ops.end(adj_lists, nvtx_ctx)
        return adj_lists

    def lookupfixOnly(self, inputs):
        with tf.device('/gpu:0'):
            ids, num_samples, layer = inputs
        with tf.device('/gpu:0'):
            adj_lists = tf.nn.embedding_lookup(self.adj_info, tf.cast(ids, tf.int32), name='sampembed')
            adj_lists = tf.transpose(adj_lists)
        with tf.device('/cpu:0'):
            adj_lists = tf.random_shuffle(adj_lists)
        with tf.device('/gpu:0'):        
            # adj_lists = tf.Print(adj_lists, [tf.shape(self.adj_info), tf.shape(adj_lists)], 'asdfasdfasdfasdfasdf')
            adj_lists = tf.transpose(adj_lists)
            adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
            # adj_lists = tf.Print(adj_lists, [tf.shape(adj_lists)], 'jjjjjjjjjjjjjh')
        return adj_lists

    def vanilla(self, inputs):
        # ids: nodes being processed this batch
        #       batchSize or batchSize*samples_2 depending on layer
        # num_samples: target degree this layer (e.g. 25 & 10 for the default values)
        # layer: layer in reverse. e.g. in 2 layered model the immediate neighbors are layer 1, and 2nd order neighbors layer 0
        #           it's also orphaned and will be pruned unless passed somewhere
        ids, num_samples, layer = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids, name='sampembed')        
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists


# for profiling
# should return right sized garbage
# hardcoded for 2 layers
class BypassSampler(Layer):
    def __init__(self, adj_info, l0shape, l1shape, **kwargs):
        super(BypassSampler, self).__init__(**kwargs)
        self.l0output = tf.slice(adj_info, [0,0], l0shape)
        self.l1output = tf.slice(adj_info, [0,0], l1shape)

    def _call(self, inputs):
        ids, num_samples, layer = inputs
        output = None
        if layer == 0:
            output = self.l0output
        elif layer == 1:
            output = self.l1output
        else:
            raise ValueError('invalid layer '+str(layer))
        # output = tf.Print(output, [], 'haaaaaaaaaaaa')
        return output