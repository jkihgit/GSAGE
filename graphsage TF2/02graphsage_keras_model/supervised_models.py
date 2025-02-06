import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            legacy_var_list, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - legacy_var_list: Stanford TensorFlow placeholder object.
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
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from legacy_var_list...
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = [adj.get_shape().as_list()[0], identity_dim]
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = features
            if not self.embeds is None:
                assert False
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = legacy_var_list["batch_size"]
        self.legacy_var_list = legacy_var_list
        self.layer_infos = layer_infos

        self.build()


    def build(self):
        batch = tf.keras.Input(shape=(), batch_size=self.batch_size, dtype=tf.int32)
        samples1, support_sizes1 = self.sample(batch, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size, name="")

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        # self.node_pred = tf.keras.layers.Dense(self.num_classes, kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg', distribution='uniform'), kernel_regularizer=tf.keras.regularizers.l2(0.5 * (FLAGS.weight_decay)), name='cls')
        self.node_pred = tf.keras.layers.Dense(self.num_classes, kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg', distribution='uniform'), name='cls')
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)
        self.model = tf.keras.Model(inputs=batch, outputs=self.node_preds, name='autogenmodel')     
        # self.model.compile(optimizer='adam', loss=custom_loss((self.aggregators, self.node_pred, self.sigmoid_loss)), metrics=['accuracy'])

        # TODO: FLAGS.learning_rate
        self.model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        self.model.summary()


    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)

    # @tf.function
    def train(self, batch, labels):
        import numpy as np
        batch = np.array(batch).flatten()
        history = self.model.fit(batch, labels, batch_size=512, epochs=1, validation_split=0.2, steps_per_epoch=1)
        s = self.model.evaluate(batch, labels, verbose=2)
        print (s)
        return (self.loss(), self.predict())

# unnecessary under default config
# sigmoid is always on
# weight decay is always off
def custom_loss(inputs):
    aggregators, node_pred, sigmoid_loss = inputs
    output = 0

    for aggregator in aggregators:
        for var in aggregator.weights():
            output += FLAGS.weight_decay * tf.nn.l2_loss(var)
    for var in node_pred.weights:
        output += FLAGS.weight_decay * tf.nn.l2_loss(var)

    clsloss = tf.nn.softmax_cross_entropy_with_logits
    if sigmoid_loss:
        clsloss = tf.nn.sigmoid_cross_entropy_with_logits    

    # function signature dictated by specs
    def loss(labels, preds):
        return output + tf.reduce_mean(clsloss(logits=preds, labels=labels))
    return loss