import tensorflow as tf
from tensorflow.python.client import timeline
from datetime import datetime, timedelta
import sklearn
from sklearn import metrics


import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# total time, num instances
batchTime = [0,0]
epochTime = [0,0]
wallTime = 0
bestF1 = [0,0]

tf_config = tf.compat.v1.ConfigProto()
tf_config.allow_soft_placement = True
# tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)
graph = tf.compat.v1.get_default_graph()

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def timeDeltaDicts(name):
    d = dict()
    d['name'] = name
    d['batchTotal'] = str(batchTime[0])
    d['epochTime'] = str(epochTime[0])
    d['wallTime'] = str(wallTime)
    d['batches'] = batchTime[1]
    d['epochs'] = epochTime[1]
    return d

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            legacy_var_list, features, train_sampler, val_sampler, degrees,
            layer_infos, minibatch, concat=True, aggregator_type="mean", strategy=None,
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
            # self.features = tf.random.uniform(tf.shape(features), -1, 1, dtype=tf.float32)
            self.features = tf.Variable(features, trainable=False, dtype=tf.float32)
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
        self.minibatch = minibatch

        # with tf.device('device:XLA_CPU:0'):
        self.train_model = self.build(train_sampler, legacy_var_list["batch_size"], legacy_var_list["gpu_count"])
        if (not FLAGS.minimini) and (not FLAGS.noval):
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
        self.node_preds = node_preds

        # compute graph
        if FLAGS.tensorboard:
            modelGraph = node_preds.graph
            logdir = "tblogs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            writer = tf.compat.v1.summary.FileWriter(logdir=logdir, graph=modelGraph)
            writer.flush()
            # import sys
            # sys.exit()

        model = tf.keras.Model(inputs=batch, outputs=node_preds, name='autogenmodel')     
        # best practices
        # opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, epsilon=1e-8, clipvalue=5.0, decay=0.15)
        # TF1 copy
        opt = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-8, clipvalue=5.0)
        # note original code used explicit batch averaging after the loss function
        # not sure if this is necessary with TF2
        # docs dont say one way or the other
        loss = tf.nn.sigmoid_cross_entropy_with_logits
        if FLAGS.timeline:
            self.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.run_metadata = tf.compat.v1.RunMetadata()
            model.compile(optimizer=opt, loss=loss, metrics=['binary_accuracy', 'categorical_accuracy'], options=self.run_options, run_metadata=self.run_metadata)
        else:
            model.compile(optimizer=opt, loss=loss, metrics=['binary_accuracy', 'categorical_accuracy'])
        # # start patch
        # global sess
        # global graph
        # with graph.as_default():
        #     init = tf.compat.v1.global_variables_initializer()
        #     sess.run(init)
        #     tf.compat.v1.keras.backend.set_session(sess)
        #     x = self.minibatch.codeTestX
        #     y = self.minibatch.codeTestY
        #     batch_size = self.legacy_var_list["batch_size"] * self.legacy_var_list["gpu_count"]
        #     callbacks = []
        #     callbacks += [TimeMeasurementsCallback()]
        #     history = model.fit(x=x, y=y, epochs=10, batch_size=batch_size, callbacks=callbacks)
        #     tl = timeline.Timeline(step_stats=self.run_metadata.step_stats)
        #     logdir = "tblogs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        #     with open(logdir+'.json', 'w') as f:
        #         f.write(tl.generate_chrome_trace_format())
        # # end patch
        model.summary()
        return model

    def minimini(self, epochs=1):
        global sess
        global graph
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            tf.compat.v1.keras.backend.set_session(sess)
            x = self.minibatch.codeTestX
            y = self.minibatch.codeTestY
            batch_size = self.legacy_var_list["batch_size"] * self.legacy_var_list["gpu_count"]
            callbacks = []
            callbacks += [TimeMeasurementsCallback()]
            history = self.train_model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
            trainTimes = timeDeltaDicts('training times')
            return (history, trainTimes)

    def train(self, verbose=2):
        global sess
        global graph
        with graph.as_default():
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            tf.compat.v1.keras.backend.set_session(sess)
            callbacks = []
            # callbacks += [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, verbose=1)]
            callbacks += [TimeMeasurementsCallback()]
            if FLAGS.patience > 0:
                x, ground_truth = self.minibatch.raw_validation_set()
                callbacks += [EarlyTerminationCallback(FLAGS.patience, self.val_model, x, ground_truth, self.legacy_var_list['val_batch_size']*self.legacy_var_list['gpu_count'])]
            # histograms
            if FLAGS.tensorboard:
                # for compute graph use the code in build()
                # https://github.com/tensorflow/tensorboard/issues/1961
                logdir = "tblogs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                callbacks += [tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True)]
            history = self.train_model.fit(self.minibatch.entire_training_set(), epochs=FLAGS.epochs, validation_data=self.minibatch.entire_testing_set(), validation_freq=FLAGS.validate_iter, callbacks=callbacks, verbose=verbose)
            for key in history.history.keys():
                print (key, ':', history.history[key][-1])
            trainTimes = timeDeltaDicts('training times')
            if FLAGS.noval:
                s1 = dict()
                f1s = [-1,-1]
                valTimes = trainTimes
            else:
                s1, f1s = self.validate(2)
                valTimes = timeDeltaDicts('validation times')
            # TODO: patience mode
            # https://www.tensorflow.org/guide/keras/train_and_evaluate#using_callbacks                
            return (history, s1, f1s, trainTimes, valTimes)

    def test(self, verbose=0):
        tst_x, ground_truth = self.minibatch.raw_testing_set()
        preds = self.train_model.predict(tst_x, self.legacy_var_list['batch_size']*self.legacy_var_list['gpu_count'], verbose)
        f1s = calc_f1(ground_truth, preds)
        print ('F1 micro', f1s[0], 'F1 macro', f1s[1])
        return f1s

    def validate(self, verbose=0):
        callbacks = []
        callbacks += [TimeMeasurementsCallback()]
        self.val_model.set_weights(self.train_model.get_weights())
        print ('validation:')
        s1 = self.val_model.evaluate(self.minibatch.entire_validation_set(), verbose=verbose)
        x, ground_truth = self.minibatch.raw_validation_set()
        preds = self.val_model.predict(x, batch_size=self.legacy_var_list['val_batch_size']*self.legacy_var_list['gpu_count'], callbacks=callbacks, verbose=verbose)
        f1s = calc_f1(ground_truth, preds)
        print ('F1 micro', f1s[0], 'F1 macro', f1s[1])
        return s1, f1s

class TimeMeasurementsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimeMeasurementsCallback, self).__init__()
        self.epochs = 0
        self.batches = 0
        self.epochStarted = 0
        self.batchStarted = 0
        self.epochTimeTotal = timedelta(0)
        self.batchTimeTotal = timedelta(0)
        self.wallStarted = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epochStarted = datetime.now()
    def on_epoch_end(self, epoch, logs=None):
        self.epochTimeTotal += (datetime.now() - self.epochStarted)
        self.epochs += 1
    def on_train_batch_begin(self, batch, logs=None):
        self.batchStarted = datetime.now()
    def on_predict_batch_begin(self, batch, logs=None):
        self.batchStarted = datetime.now()
    def on_train_batch_end(self, batch, logs=None):
        self.batchTimeTotal += datetime.now() - self.batchStarted
        self.batches += 1
    def on_predict_batch_end(self, batch, logs=None):
        self.batchTimeTotal += datetime.now() - self.batchStarted        
        self.batches += 1
    def on_predict_begin(self, logs=None):
        self.wallStarted = datetime.now()
    def on_train_begin(self, logs=None):
        self.wallStarted = datetime.now()
    def on_train_end(self, logs=None):
        global batchTime, epochTime, wallTime
        print ('Training Time Stats')
        print ('Batch time:', self.batchTimeTotal, '/', self.batches, '=', self.batchTimeTotal/self.batches)
        print ('Epoch time:', self.epochTimeTotal, '/', self.epochs, '=', self.epochTimeTotal/self.epochs)
        print ('Wall time:', datetime.now() - self.wallStarted)
        wallTime = (datetime.now() - self.wallStarted).total_seconds()
        batchTime = [self.batchTimeTotal.total_seconds(), self.batches]
        epochTime = [self.epochTimeTotal.total_seconds(), self.epochs]
    def on_predict_end(self, logs=None):
        global batchTime, epochTime, wallTime
        print ('Inference Time Stats')
        print ('Batch time:', self.batchTimeTotal, '/', self.batches, '=', self.batchTimeTotal/self.batches)
        if self.epochs > 0:
            print ('Epoch time:', self.epochTimeTotal, '/', self.epochs, '=', self.epochTimeTotal/self.epochs)
        print ('Wall time:', datetime.now() - self.wallStarted)
        wallTime = (datetime.now() - self.wallStarted).total_seconds()
        batchTime = [self.batchTimeTotal.total_seconds(), self.batches]
        epochTime = [self.epochTimeTotal.total_seconds(), self.epochs]

# does not store best weights
# feed this BS * GPU_Count
class EarlyTerminationCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience, val_model, val_x, val_y, batch_size):
        super(EarlyTerminationCallback, self).__init__()
        self.targetPatience = patience
        self.epochsSinceLastMicroImprovement = 0
        self.epochsSinceLastMacroImprovement = 0
        self.bestF1Micro = 0
        self.bestF1Macro = 0
        self.val_model = val_model
        self.val_x = val_x
        self.val_y = val_y
        self.batch_size = batch_size
        self.epochs = 0
    def on_epoch_end(self, epoch, logs=None):
        self.val_model.set_weights(self.model.get_weights())
        preds = self.val_model.predict(self.val_x, batch_size=self.batch_size, verbose=0)
        f1micro, f1macro = calc_f1(self.val_y, preds)
        self.epochs += 1
        self.epochsSinceLastMicroImprovement += 1
        self.epochsSinceLastMacroImprovement += 1
        if f1micro > self.bestF1Micro:
            self.bestF1Micro = f1micro
            self.epochsSinceLastMicroImprovement = 0
            print ('New F1 Micro', f1micro)
        if f1macro > self.bestF1Macro:
            self.bestF1Macro = f1macro
            self.epochsSinceLastMacroImprovement = 0
            print ('New F1 Macro', f1macro)
        microTerminate = False
        macroTerminate = False
        if (self.epochsSinceLastMicroImprovement >= self.targetPatience):
            print ('F1 micro votes stop')
            microTerminate = True
        if self.epochsSinceLastMacroImprovement >= self.targetPatience:
            print ('F1 macro votes stop')
            macroTerminate = True
        if microTerminate and macroTerminate:
            print ('Stop vote wins')
            self.model.stop_training = True
    def on_train_end(self, batch, logs=None):
        global bestF1
        print ('total epochs:', self.epochs)
        print ('best F1 micro:', self.bestF1Micro)
        print ('best F1 macro:', self.bestF1Macro)
        bestF1[0] = self.bestF1Micro
        bestF1[1] = self.bestF1Macro
