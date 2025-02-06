import tensorflow as tf
## import nvtx.plugins.tf as nvtx_tf
import sys

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import BypassAggregator, MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
from graphsage.utils import boolparse, randInts, randFloats
from .inits import zeros

flags = tf.app.flags
FLAGS = flags.FLAGS

do_sampling = True
# if true will run normal aggregation regardless of other options
do_aggregation = True
do_aggregation_fetch = True
do_aggregation_nn = True
do_classification = True

# makes finding stuff in logs easier
magic = 'akduifasdkyfgaksdjf'

samplerBypassMinNodeID = 1
samplerBypassMaxNodeID = 50000-1
classifierBypassMinConfidence = -10
classifierBypassMaxConfidence = 10

def initOverrides(inputs):
    global do_sampling, do_aggregation, do_classification, do_aggregation_fetch, do_aggregation_nn
    do_sampling = inputs['do_sampling']
    do_aggregation = inputs['do_aggregation']
    do_classification = inputs['do_classification']
    do_aggregation_fetch = inputs['do_aggregation_fetch']
    do_aggregation_nn = inputs['do_aggregation_nn']

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0, 
            entryOverride=None, labelOverride=None,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == 'bypass':
            self.aggregator_cls = BypassAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        self.placeholders = placeholders
        # get info from placeholders... 
        if labelOverride != None:
            self.label = labelOverride
        else:
            self.label = self.placeholders['labels']
        if entryOverride != None:
            self.inputs1 = entryOverride
        else:
            self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            # default case
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        elif do_aggregation and do_aggregation_fetch:
            self.np_features = features
            self.placholder_feature = tf.placeholder(dtype=tf.float32, shape=self.np_features.shape)
            self.features = tf.Variable(None, trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        elif (not do_aggregation) or (do_aggregation and (not do_aggregation_fetch)):
            self.features = len(features[-1])
        else:
            raise NotImplementedError
        # self.features=tf.Print(self.features,[self.features.get_shape()],'HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
        # self.features=tf.Print(self.features, [self.adj_info.get_shape()], 'hhhhhhhhhhhhhh')
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = tf.shape(self.inputs1)[0]
        # self.batch_size = placeholders["batch_size"]
        self.layer_infos = layer_infos
        # self.features = tf.Print(self.features, [self.features.shape], magic+'features')

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

        # print (magic+'vertex output', self.node_pred.shape)
        # print (magic+'adj matrix', self.adj_info.shape)
        # print (magic+'degrees', self.degrees.shape)

    def build(self):
        # batch_size = tf.shape(self.inputs1)[0]
        dim_mult = 2 if self.concat else 1
        # samples1 structure
        # a pylist of 3 tensors, each 1D of size batchSize, immediateNeighborhood(e.g. batchSize*10), depth2Neighborhood(e.g. batchSize*10*25)
        # cant use zeros for this since this is used for aggregation mem access
        # using the same address x times != using x different addresses
        # support_sizes: dim so the next functions know how to split samples1 (e.g. [1,10,250])        
        #self.adj_info, nvtx_ctx_master = nvtx_tf.ops.start(self.adj_info, message='Manual NVTX Region', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvtx, trainable=True)
        if do_sampling:
            with tf.variable_scope(self.name + '/sampling_'+magic):
                samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
            # samples1[0] = tf.Print(samples1[0], [], 'truetruetruetruetruetruetrue')
        # hard coded for 2 layer op
        else:
            support_sizes1 = [1, FLAGS.samples_2, FLAGS.samples_2*FLAGS.samples_1]
            nodes = tf.Variable(list(range(FLAGS.batch_size)), trainable=False)
            immNeighs = tf.Variable(randInts(support_sizes1[1]*FLAGS.batch_size, samplerBypassMinNodeID, samplerBypassMaxNodeID), trainable=False)
            t2Neighs = tf.Variable(randInts(support_sizes1[2]*FLAGS.batch_size, samplerBypassMinNodeID, samplerBypassMaxNodeID), trainable=False)
            nodes = tf.slice(nodes, [0], [self.batch_size])
            immNeighs = tf.slice(immNeighs, [0], [support_sizes1[1]*self.batch_size])
            t2Neighs = tf.slice(t2Neighs, [0], [support_sizes1[2]*self.batch_size])
            samples1 = [nodes, immNeighs, t2Neighs]
            # samples1[0] = tf.Print(samples1[0], [], 'falsefalsefalsefalsefalse')
        if do_aggregation:
            with tf.variable_scope(self.name + '/aggregator_'+magic):
                num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
                self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                        support_sizes1, concat=self.concat, model_size=self.model_size, bypassFlags = [do_aggregation_fetch, do_aggregation_nn])
                if do_aggregation_nn:
                    #self.outputs1, nvtx_ctx = nvtx_tf.ops.start(self.outputs1, message='Aggregation Normalize', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvtx)
                    self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
                    #self.outputs1 = nvtx_tf.ops.end(self.outputs1, nvtx_ctx)
            # self.outputs1 = tf.Print(self.outputs1, [], 'truetruetruetruetruetruetrue')
        else:            
            import math
            # will disconnect the graph from samples1[0:1]
            # probably not a big deal since they're << samples1[2]
            self.outputs1 = tf.cast(samples1[-1], tf.float32)
            self.outputs1 = tf.tile(self.outputs1, [self.batch_size * math.ceil(dim_mult*self.dims[-1]/support_sizes1[2])])
            self.outputs1 = tf.reshape(self.outputs1, [self.batch_size, -1])
            # zeroes disconnects the whole sampling function
            # self.outputs1 = zeros([FLAGS.batch_size, dim_mult*self.dims[-1]], name='aggBypassDos', trainable=False)
            self.outputs1 = tf.slice(self.outputs1, [0,0], [self.batch_size, dim_mult*self.dims[-1]])
            # self.outputs1 = tf.Print(self.outputs1, [], 'falsefalsefalsefalsefalse')
            # self.outputs1 = tf.Print(self.outputs1, [tf.shape(self.outputs1)], 'aaaaaaaaaaaaaa')

        # node_pred: degree of confidence (non-normalized), shape (batchSize, num_classes)
        if do_classification:
            with tf.variable_scope(self.name + '/classification_'+magic):
                # dropout bypass implemented in layers.Dense
                self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, dropout=self.placeholders['dropout'], act=lambda x : x)
                #dim_mult, nvtx_ctx = nvtx_tf.ops.start(dim_mult, message='Classifier NN', domain_name='Forward', grad_domain_name='Gradient', enabled=FLAGS.enable_nvtx)
                with tf.variable_scope('classificationnn'):
                # TF graph management
                    self.node_preds = self.node_pred(self.outputs1)
                #self.node_pred = nvtx_tf.ops.end(self.node_pred, nvtx_ctx)
            # self.node_preds = tf.Print(self.node_preds, [], 'truetruetruetruetruetruetrue')
        elif do_aggregation:
            # ignoring previous steps disconnects the whole sampling & aggregation functions
            # self.node_preds = tf.Variable([randFloats(self.num_classes, classifierBypassMinConfidence, classifierBypassMaxConfidence) for x in range(FLAGS.batch_size)], trainable=False)
            self.node_preds = tf.slice(self.outputs1, [0,0], [self.batch_size, self.num_classes])
            # self.node_preds = tf.Print(self.node_preds, [], 'falsefalsefalsefalsefalse')
        else:
            # needs something to optimize
            bias = zeros((self.num_classes))
            self.node_preds = tf.slice(self.outputs1, [0,0], [self.batch_size, self.num_classes]) + bias
        #self.node_preds = nvtx_tf.ops.end(self.node_preds, nvtx_ctx_master)

        with tf.variable_scope(self.name + '/backprop_'+magic):
            with tf.variable_scope(self.name + '/backprop_'+magic + '/bp_loss'):
                self._loss()
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            with tf.variable_scope(self.name + '/backprop_'+magic + '/bp_clip'):
                if do_aggregation or do_classification or do_aggregation_nn:
                    optimizer_input = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) for grad, var in grads_and_vars]
                else:
                    optimizer_input = grads_and_vars
                    # optimizer_input = [(tf.Variable(grad, dtype=tf.float32) if ((grad is not None) and (not ifinstance(grad, tf.Variable))) else None, var) for grad, var in grads_and_vars]
            self.grad, _ = optimizer_input[0]
            self.opt_op = self.optimizer.apply_gradients(optimizer_input)
        with tf.variable_scope(self.name + '/classification_'+magic+'/final_pred'):

            if do_classification:
                self.preds = self.predict()
            else:
                self.preds = self.node_preds
        # self.preds = tf.Print(self.preds, [tf.shape(self.preds)], 'dfsiojddfjkkjhdsoi')

    def _loss(self):
        if do_aggregation and do_aggregation_nn:
            # Weight decay loss
            # not used by default (weight == 0)

            with tf.variable_scope(self.name + '/backprop_'+magic + '/bp_loss/agg'):
                for aggregator in self.aggregators:
                    for var in aggregator.vars.values():
                        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            with tf.variable_scope(self.name + '/backprop_'+magic + '/bp_loss/cls'):
                if do_classification:
                    for var in self.node_pred.vars.values():
                        self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        if do_classification:
            with tf.variable_scope(self.name + '/backprop_'+magic + '/bp_loss/crossent'):
                # classification loss
                if self.sigmoid_loss:
                    self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.node_preds,
                            labels=self.label))
                else:
                    self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            logits=self.node_preds,
                            labels=self.label))

        if not (do_aggregation or do_classification):
            self.loss += tf.nn.l2_loss(self.node_preds)

        if FLAGS.tensorboard:
            tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
