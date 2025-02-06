import tensorflow as tf
from datetime import datetime


import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")


class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            legacy_var_list, features, train_adj, val_adj, degrees,
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
        self.train_adj = train_adj
        self.val_adj = val_adj
        if identity_dim > 0:
            assert False
            self.embeds = tf.Variable([train_adj.get_shape().as_list()[0], identity_dim], dtype=tf.int32, trainable=False)
        else:
            self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.dtypes.cast(features, dtype=tf.float32)
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
        self.node_pred = tf.keras.layers.Dense(self.num_classes, kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg', distribution='uniform'), name='cls')
        assert FLAGS.sigmoid
        # TODO: make this not retardo
        finalfunc = lambda x: x
        self.node_preds = finalfunc(self.node_pred(self.outputs1))
        self.model = tf.keras.Model(inputs=batch, outputs=self.node_preds, name='autogenmodel')     
        # best practices
        # opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, epsilon=1e-8, clipvalue=5.0, decay=0.15)
        # TF1 copy
        opt = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, clipvalue=5.0)
        # note original code used explicit batch averaging after the loss function
        # not sure if this is necessary with TF2
        # docs dont say one way or the other
        loss = tf.nn.sigmoid_cross_entropy_with_logits
        self.model.compile(optimizer=opt, loss=loss, metrics=['binary_accuracy', 'categorical_accuracy'])
        self.model.summary()

    def train(self, trnds, tstds, vds):
        callbacks = []
        callbacks += [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, verbose=1)]
        if FLAGS.tensorboard:
            logdir = "tblogs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks += [tf.keras.callbacks.TensorBoard(log_dir=logdir)]
        # history = self.model.fit_generator(trnds, epochs=1, validation_data=tstds, validation_freq=FLAGS.validate_iter, callbacks=callbacks)
        history = self.model.fit_generator(trnds, epochs=100, validation_data=tstds, validation_freq=FLAGS.validate_iter, callbacks=callbacks)
        # TODO: patience mode
        # https://www.tensorflow.org/guide/keras/train_and_evaluate#using_callbacks
        print ('fit done')
        self.validate(vds, verbose=2)
        # TODO: adj info swapping
        # samplers = [x.neigh_sampler for x in self.layer_infos]
        # for sampler in samplers:
        #     sampler.adj_info = self.val_adj
        # self.validate(vds, verbose=2)
        # TODO: return validation F1 instead of None
        return (s[0], None)

    def validate(self, vds, verbose=0):
        s = self.model.evaluate(vds, verbose=verbose)
        print ('validation loss:', s[0])
        print ('validation acc :', s[1])
        return s
