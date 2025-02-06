from collections import namedtuple

import tensorflow as tf
import math

import graphsage.layers as layers
import graphsage.metrics as metrics

from .prediction import BipartiteEdgePredLayer
from .aggregators import MeanAggregator

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.legacy_var_list = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        


# SAGEInfo is a namedtuple that specifies the parameters 
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])

class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, legacy_var_list, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", identity_dim=0,
            **kwargs):
        '''
        Args:
            - legacy_var_list: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features. 
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        else:
            raise Exception("Unknown aggregator: ", aggregator_type)

        # get info from legacy_var_list...
        assert False
        self.inputs1 = legacy_var_list["batch1"]
        self.inputs2 = legacy_var_list["batch2"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           # self.embeds = tf.Variable([adj.get_shape().as_list()[0], identity_dim], name="node_embeddings", trainable=False)
           self.embeds = [adj.get_shape().as_list()[0], identity_dim]
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            # self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            self.features = features
            if not self.embeds is None:
                assert False
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat

        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = legacy_var_list["batch_size"]*legacy_var_list['gpu_count']
        self.legacy_var_list = legacy_var_list
        self.layer_infos = layer_infos

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    
    def aggregate(self, input_features, dims, batch_size,
            aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            input_features: the input features arranged in a 21x(FVLEN) matrix, starting with the input node's FV, immediate neighbor nodes' FV, and 2nd order neighbor nodes' FV
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        # reshape to 3x(variable)x(FVLEN), where the first dim is the hop count
        num_samples = [FLAGS.samples_1, FLAGS.samples_2]
        support_sizes = [1] + num_samples

        a, b, c = tf.split(input_features, [1, FLAGS.samples_2, FLAGS.samples_2], 1)
        b = tf.reshape(b, (batch_size*num_samples[1], -1))
        c = tf.reshape(c, (batch_size*num_samples[1], -1))
        hidden = [a, b, c]

        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                act = tf.nn.relu
                if layer == len(num_samples) - 1:
                    act = lambda x : x
                aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], name=name+str(layer), concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1

                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]

                if hop == 0:
                    reshape_neighs = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], x[1]), name=str(layer)+"_"+str(hop)+'unflatten_neigh_feats')
                    neigh_vecs = reshape_neighs((hidden[hop+1], neigh_dims))
                else:
                    neigh_vecs = hidden[hop+1]
                h = aggregator((hidden[hop], neigh_vecs), hop, layer)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]