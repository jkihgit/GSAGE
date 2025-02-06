import tensorflow as tf
from datetime import datetime, timedelta


import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")


class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            legacy_var_list, features, train_sampler, val_sampler, degrees,
            layer_infos, concat=True, aggregator_type="mean", strategy=None,
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

        self.model_size = model_size
        if identity_dim > 0:
            assert False
            # self.embeds = tf.Variable([train_adj.get_shape().as_list()[0], identity_dim], dtype=tf.int32, trainable=False)
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
        self.legacy_var_list = legacy_var_list
        self.layer_infos = layer_infos

        self.train_model = self.build(train_sampler, legacy_var_list["batch_size"], legacy_var_list["gpu_count"])
        self.val_model = self.build(val_sampler, legacy_var_list["val_batch_size"], legacy_var_list["gpu_count"])

    def build(self, sampler, batch_size, gpu_count):
        batch = tf.keras.Input(shape=(), batch_size=batch_size*gpu_count, dtype=tf.int32)
        samples1, support_sizes1 = self.sample(batch, self.layer_infos, sampler, batch_size)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        outputs1 = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, batch_size, concat=self.concat, model_size=self.model_size, name="")

        outputs1 = tf.nn.l2_normalize(outputs1, 1)
        node_pred = tf.keras.layers.Dense(self.num_classes, kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg', distribution='uniform'), name='cls')
        assert FLAGS.sigmoid
        # TODO: make this not retardo
        finalfunc = lambda x: x
        node_preds = finalfunc(node_pred(outputs1))
        model = tf.keras.Model(inputs=batch, outputs=node_preds, name='autogenmodel')     
        # best practices
        # opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, epsilon=1e-8, clipvalue=5.0, decay=0.15)
        # TF1 copy
        opt = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, clipvalue=5.0)
        # note original code used explicit batch averaging after the loss function
        # not sure if this is necessary with TF2
        # docs dont say one way or the other
        loss = tf.nn.sigmoid_cross_entropy_with_logits
        model.compile(optimizer=opt, loss=loss, metrics=['binary_accuracy', 'categorical_accuracy'])
        return model

    def minimum_viable(self, x, y, batch_size):
        return self.train_model.fit(x=x, y=y, epochs=1, batch_size=batch_size)

    def train(self, trnds, tstds, vds, verbose=2):
        callbacks = []
        callbacks += [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, verbose=1)]
        callbacks += [TrainingTimeCallback()]
        if FLAGS.tensorboard:
            logdir = "tblogs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks += [tf.keras.callbacks.TensorBoard(log_dir=logdir)]
        # history = self.model.fit_generator(trnds, epochs=1, validation_data=tstds, validation_freq=FLAGS.validate_iter, callbacks=callbacks)
        history = self.train_model.fit(trnds, epochs=100, validation_data=tstds, validation_freq=FLAGS.validate_iter, callbacks=callbacks, verbose=verbose)
        for key in history.history.keys():
            print (key, ':', history.history[key][-1])
        self.validate(vds, 2)
        # TODO: patience mode
        # https://www.tensorflow.org/guide/keras/train_and_evaluate#using_callbacks                
        # self.validate(vds, verbose=2)
        # TODO: return validation F1 instead of None
        return history

    def validate(self, vds, verbose=0):
        self.val_model.set_weights(self.train_model.get_weights())
        print ('validation:')
        s = self.val_model.evaluate(vds, verbose=verbose)
        return s

class TrainingTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TrainingTimeCallback, self).__init__()
        self.batches = 0
        self.epochStarted = 0
        self.timeSoFar = timedelta(0)
    def on_epoch_begin(self, epoch, logs=None):
        self.epochStarted = datetime.now()
    def on_epoch_end(self, epoch, logs=None):
        self.timeSoFar += (datetime.now() - self.epochStarted)
        self.epochStarted = -1
    def on_train_batch_begin(self, batch, logs=None):
        self.batches += 1
    def on_predict_batch_begin(self, batch, logs=None):
        self.batches += 1
    def on_train_end(self, batch, logs=None):
        assert self.epochStarted < 0
        print ('Train batch time:', self.timeSoFar/self.batches)
        print ('Across', self.batches, 'batches')
        print ('Total time:', self.timeSoFar)
    def on_predict_end(self, batch, logs=None):
        assert self.epochStarted < 0
        print ('Inference batch time:', self.timeSoFar/self.batches)
        print ('Across', self.batches, 'batches')
        print ('Total time:', self.timeSoFar)

# class ValidationCallback(tf.keras.callbacks.Callback):
#     def __init__(self):
