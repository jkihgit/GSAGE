from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
# #from nvtx.plugins.tf.estimator import NVTXHook as nvtx_hook

from tensorflow.python.client import timeline
import numpy as np
import sklearn
from sklearn import metrics

from graphsage.supervised_models import SupervisedGraphsage, initOverrides
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator, TFBatching
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.neigh_samplers import BypassSampler
from graphsage.utils import load_json, load_adj, load_edgelist, boolparse
from graphsage.inits import minimini_glorot_intercept


load_data = load_edgelist
# load_data = load_json
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

GPU_MEM_FRACTION = 0.95

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
trainingTimeTotal = 0
timingfunction = time.time

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', -1.0, 'dropout rate (1 - keep probability) -ve to disable (not same as 0 drop)')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', -1, "which gpu to use.")
flags.DEFINE_integer('print_every', 128, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

flags.DEFINE_string('minimini', 'False', 'Only run one iteration and stop. Useful for nvprof')
flags.DEFINE_string('print_param_count', 'False', 'if true will exit after initializing the tensor graph and reporting the trainable parameter count')
flags.DEFINE_string('trainable_params_only', 'True', 'if false will include non-trainable vars as well')
flags.DEFINE_string('print_param_log_file', '', 'if not empty will print param count log here')
flags.DEFINE_string('do_training', 'True', 'if false will supersede all other settings. Not tested without minimini')
flags.DEFINE_string('do_testing', 'True', 'if false will supersede all other settings. Not tested without minimini')

flags.DEFINE_string('do_sampling', 'True', 'if false will return garbage')
flags.DEFINE_string('do_aggregation', 'True', 'if false will return garbage and overrides other aggregation bypass options. i.e. must be on for other agg bypass params to work')
flags.DEFINE_string('do_aggregation_fetch', 'True', 'if false will bypass embedding layer')
flags.DEFINE_string('do_aggregation_nn', 'True', 'if false will only do embedding')
flags.DEFINE_string('do_classification', 'True', 'if false will return garbage')
flags.DEFINE_float('dataset_size_ratio', 0.0, 'remove this much of the data while loading')
flags.DEFINE_float('feature_size_ratio', 0.0, 'remove this much of the feature vector')
flags.DEFINE_integer('patience', -1, 'quit early if no validation improvement for this much epochs. -1 for off')
flags.DEFINE_integer('tensorboard', 0, 'on or off')
flags.DEFINE_integer('timeline', 0, 'on or off. NOT COMPATIBLE WITH NVTX')
flags.DEFINE_integer('timeline_every', 50, 'batches')
flags.DEFINE_integer('timeline_mini', 0, 'if true will exit after writing one timeline file')
flags.DEFINE_integer('allow_soft_placement', 0, 'if false will crash when unable to assign op to designated device')
flags.DEFINE_integer('lookup_fix', 1, 'fix adjacency table by loading it to GPU instead of host memory')
flags.DEFINE_integer('randomizer_fix', 1, 'use randomizer with CUDA kernels')
flags.DEFINE_integer('feeddict_fix', 0, 'use data pipelining instead of feed dict')
# #flags.DEFINE_integer('enable_nvtx', 0, 'NVTX is used to speed up and better categorize nvprof results. NOT COMPATIBLE WITH TF TIMELINE')

if FLAGS.feeddict_fix:
    # feeddict_fix breaks placeholder['batch_size'], which will store max batch size instead of real time batch size
    # last batch is almost always < FLAGS.batch_size
    assert (FLAGS.do_sampling and FLAGS.do_aggregation and FLAGS.do_aggregation_fetch and FLAGS.do_aggregation_nn and FLAGS.do_classification)

#assert not (FLAGS.enable_nvtx and FLAGS.timeline)
#assert not FLAGS.enable_nvtx # disabled due to incompatibility issues between containers
print ('timeline:', boolparse(FLAGS.timeline))
print ('timeline mini:', boolparse(FLAGS.timeline_mini))
# #print ('NVTX:', boolparse(FLAGS.enable_nvtx))

print ('lookup fix:', boolparse(FLAGS.lookup_fix))
print ('randomizer fix:', boolparse(FLAGS.randomizer_fix))
print ('preloader enabled:', boolparse(FLAGS.feeddict_fix))
print ('dropout pruning:', FLAGS.dropout < 0)


minimini = boolparse(FLAGS.minimini)
print_param_count = boolparse(FLAGS.print_param_count)
trainable_params_only = boolparse(FLAGS.trainable_params_only)
do_training = boolparse(FLAGS.do_training)
do_testing = boolparse(FLAGS.do_testing)
# assumes 0 or 1 are false
# definitely incorrect for both aggregation and classification being off
# not sure about other combinations
do_sampling = boolparse(FLAGS.do_sampling)
do_aggregation = boolparse(FLAGS.do_aggregation)
do_classification = boolparse(FLAGS.do_classification)
do_aggregation_fetch = boolparse(FLAGS.do_aggregation_fetch)
do_aggregation_nn = boolparse(FLAGS.do_aggregation_nn)

if (not do_aggregation_nn) or (not do_aggregation_fetch):
    raise NotImplementedError


overrideKVP = dict()
overrideKVP['do_sampling'] = do_sampling
overrideKVP['do_aggregation'] = do_aggregation
overrideKVP['do_classification'] = do_classification
overrideKVP['do_aggregation_fetch'] = do_aggregation_fetch
overrideKVP['do_aggregation_nn'] = do_aggregation_nn
initOverrides(overrideKVP)

if FLAGS.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)
print ('minimini mode:', minimini)
print ('sampling mode:', do_sampling)
print ('aggregation mode:', do_aggregation)
print ('aggregation fetch mode:', do_aggregation_fetch)
print ('aggregation nn mode:', do_aggregation_nn)
print ('classification mode:', do_classification)
minimini_glorot_intercept = minimini


def print_parameter_count():
    def trainables_only():
        return tf.trainable_variables()
    def nontrainables_included():
        # results in a plethora of unnecessary nodes
        # much better to manually do stuff
        graph = tf.get_default_graph()    
        tensors_per_node = [node.values() for node in graph.get_operations()]
        return [tensor for tensors in tensors_per_node for tensor in tensors]
    total_parameters = 0
    log_output = 'Name\tShape\tSize\n'
    tensors = []
    if trainable_params_only:
        tensors = trainables_only()
    else:
        tensors = nontrainables_included()
    for variable in tensors:
        log_line = variable.name + '\t'
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        log_line += str(shape) + '\t'
        variable_parameters = 1
        exceptionFlag = False
        batchAssumeFlag = False
        try:
            for dim in shape:
                # if dim.value == None:
                    # assume batch size
                    # dynamic shape is used for 2 things
                    # 1) dynamic batch size
                    # 2) dynamic input vector size
                    # variable_parameters *= FLAGS.batch_size
                    # batchAssumeFlag = True
                # else:
                    # variable_parameters *= dim.value
                variable_parameters *= dim.value
        except:
            exceptionFlag = True
        if not exceptionFlag:
            log_line += str(variable_parameters) 
        if batchAssumeFlag:
            log_line += '\tAssumed unknown dim was batch size'
        log_line += '\n'
        log_output += log_line
        total_parameters += variable_parameters
    print('Total param count:', total_parameters)
    log_output += 'Total: ' + str(total_parameters) + '\n'
    if len(print_param_log_file):
        with open(print_param_log_file, 'w') as f:
            f.write(log_output)
        print ('Param count saved to', print_param_log_file)

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = timingfunction()
    if FLAGS.feeddict_fix:
        sess.run(minibatch.dsInitVal)        
        node_outs_val = sess.run([model.preds, model.loss])
        labels = minibatch_iter.get_labels_val(iter_num)
    else:
        feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
        node_outs_val = sess.run([model.preds, model.loss], feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (timingfunction() - t_test)

def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def incremental_evaluate(sess, model, minibatch_iter, size, test=False, bsph=None):
    t_test = timingfunction()
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    if FLAGS.feeddict_fix:
        if test:
            initop = minibatch_iter.dsInitTest
            lblfunc= minibatch_iter.get_labels_test
        else:
            initop = minibatch_iter.dsInitVal
            lblfunc= minibatch_iter.get_labels_val
        sess.run(initop)
        try:
            while True:
                node_outs_val = sess.run([model.preds, model.loss])
                val_preds.append(node_outs_val[0])
                labels.append(lblfunc(iter_num))
                val_losses.append(node_outs_val[1])
                iter_num += 1
                if minimini and (not do_training):
                    break
        except tf.errors.OutOfRangeError:
            # this is actually the recommended method
            # wut
            pass
    else:
        finished = False
        while not finished:
            feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
            node_outs_val = sess.run([model.preds, model.loss], 
                             feed_dict=feed_dict_val)
            val_preds.append(node_outs_val[0])
            labels.append(batch_labels)
            val_losses.append(node_outs_val[1])
            iter_num += 1
            if minimini and (not do_training):
                break
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (timingfunction() - t_test)

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, test_data=None):
    global trainingTimeTotal

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    if FLAGS.feeddict_fix:
        minibatch = TFBatching(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs,
            validation_batch_size=FLAGS.validate_batch_size)
    else:
        minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs) 
    if FLAGS.lookup_fix:
        adj_info = tf.constant(minibatch.adj, dtype=tf.float32)
        adj_info = tf.Variable(adj_info, trainable=False, name="adj_info")
        test_adj_info = tf.constant(minibatch.adj, dtype=tf.float32)
        test_adj_info = tf.Variable(minibatch.test_adj, trainable=False, name="test_adj_info")
    else:
        adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    if FLAGS.feeddict_fix:
        iterTemplatebatch, iterTemplatelabels = minibatch.iter.get_next()
    # model used
    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        if not do_sampling:
            # hardcoded for 2 layers
            # also, this layer 0 = layer 1 thing is super confusing
            l1shape = [FLAGS.batch_size*FLAGS.samples_2, FLAGS.samples_1]
            l2shape = [FLAGS.batch_size, FLAGS.samples_2]
            print (l1shape, l2shape)
            sampler = BypassSampler(adj_info, l1shape , l2shape)
        agg = "mean"
        if not do_aggregation:
            agg = "bypass"
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            # model used
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
        eo = None
        lo = None
        if FLAGS.feeddict_fix:
            eo = iterTemplatebatch
            lo = iterTemplatelabels
        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos, 
                                     aggregator_type=agg,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True,
                                     entryOverride=eo,
                                     labelOverride=lo)
    elif FLAGS.model == 'gcn':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = FLAGS.allow_soft_placement
    
    # Initialize session
    # #sess = tf.train.MonitoredSession(hooks=[nvtx_hook(skip_n_steps=0, name='MSHOOK')])
    sess = tf.Session(config=config)
    if FLAGS.tensorboard:
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    feed_dict_init = dict()
    if FLAGS.dropout >= 0:
        feed_dict_init.update({placeholders['dropout']: FLAGS.dropout})
    if FLAGS.lookup_fix == False:
        feed_dict_init.update({adj_info_ph: minibatch.adj})
    if not len(feed_dict_init.keys()):
        feed_dict_init = None
    sess.run(tf.global_variables_initializer(), feed_dict=feed_dict_init)
    
    # Train model
    
    total_steps = 0
    avg_time = 0.0
    best_F1_mic = 0
    best_F1_mac = 0
    no_improvement_last_this_epochs_mic = 0
    no_improvement_last_this_epochs_mac = 0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    if print_param_count:
        print_parameter_count()
        import sys
        sys.exit()

    if not do_training:
        FLAGS.epochs = 0
    elif minimini:
        FLAGS.epochs = 1

    val_cost = -1
    val_f1_mic = -1
    val_f1_mac = -1

    for epoch in range(FLAGS.epochs): 
        if minimini == False:    
            minibatch.shuffle() 
        if FLAGS.feeddict_fix:
            sess.run(minibatch.dsInit)

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        batch = 0
        while not minibatch.end(batch):
            batch += 1
            # print ('batch', batch)
            # Construct feed dictionary
            # must be None when using feeddict_fix?
            feed_dict = None
            if FLAGS.feeddict_fix == False:
                feed_dict, labels = minibatch.next_minibatch_feed_dict()
            else:
                labels = minibatch.get_labels(batch-1)

            # Training step
            if FLAGS.tensorboard:
                compute_graphs = [merged, model.opt_op, model.loss, model.preds]
            else:
                compute_graphs = [model.opt_op, model.loss, model.preds]
            options = None
            run_metadata = None
            if FLAGS.timeline and (batch % FLAGS.timeline_every == 0):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            t = timingfunction()
            outs = sess.run(compute_graphs, options=options, feed_dict=feed_dict, run_metadata=run_metadata)
            endTime = timingfunction()
            if minimini:
                # known issue: nvprof may hang unless told explicitly to stop
                import sys
                sys.exit()
            if FLAGS.tensorboard:
                train_cost = outs[2]
            else:
                train_cost = outs[1]
            if FLAGS.timeline and (batch % FLAGS.timeline_every == 0):
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                fp = log_dir() + str(batch)+'.json'
                with open(fp, 'w') as f:  
                    f.write(chrome_trace)
                    print ("saved trace to", fp)
                if boolparse(FLAGS.timeline_mini):
                    return


            # feeddict_fix breaks interleaved validation
            # dataset init resets the whole epoch
            if (do_testing) and (not FLAGS.feeddict_fix) and (iter % FLAGS.validate_iter == 0) :
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
                else:
                    val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost
            elif (not do_testing) and (minimini):
                break
    
            # Print results
            if FLAGS.tensorboard:
                if total_steps % FLAGS.print_every == 0:
                    summary_writer.add_summary(outs[0], total_steps)
            trainingTimeTotal += endTime - t
            avg_time = (avg_time * total_steps + endTime - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic), 
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac), 
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic), 
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac), 
                      "time=", "{:.5f}".format(avg_time))
 
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break
            if minimini:
                break

        if total_steps > FLAGS.max_total_steps:
            print ('iteration limit reached')
            break
        if minimini:
            break
        if (FLAGS.patience > 0):
            sess.run(val_adj_info.op)
            # turning on test will run test nodes instead of validation nodes
            val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.validate_batch_size, test=False, bsph=placeholders['batch_size'])
            no_improvement_last_this_epochs_mic += 1
            no_improvement_last_this_epochs_mac += 1
            if val_f1_mic > best_F1_mic:
                print ('new best F1 mic', best_F1_mic, '->',val_f1_mic,'after',no_improvement_last_this_epochs_mic)
                best_F1_mic = val_f1_mic
                no_improvement_last_this_epochs_mic = 0
            if val_f1_mac > best_F1_mac:
                print ('new best F1 mac', best_F1_mac, '->',val_f1_mac,'after',no_improvement_last_this_epochs_mac)
                best_F1_mac = val_f1_mac
                no_improvement_last_this_epochs_mac = 0
            if ((FLAGS.patience <= no_improvement_last_this_epochs_mic) and (FLAGS.patience <= no_improvement_last_this_epochs_mac)):
                print ('patience limit reached', no_improvement_last_this_epochs_mic, no_improvement_last_this_epochs_mac, epoch)
                print ('best figures:',best_F1_mic,best_F1_mac)
                break
            sess.run(train_adj_info.op)
    
    print("Optimization Finished!")
    print("Total time spent training:", trainingTimeTotal)
    sess.run(val_adj_info.op)
    if do_testing:
        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, bsph=placeholders['batch_size'])
        print("Full validation stats:",
                    "loss=", "{:.5f}".format(val_cost),
                    "f1_micro=", "{:.5f}".format(val_f1_mic),
                    "f1_macro=", "{:.5f}".format(val_f1_mac),
                    "time=", "{:.5f}".format(duration))
    if not minimini:
        with open(log_dir() + "val_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                    format(val_cost, val_f1_mic, val_f1_mac, duration))

        print("Writing test set stats to file (don't peak!)")
        val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test=True, bsph=placeholders['batch_size'])
        with open(log_dir() + "test_stats.txt", "w") as fp:
            fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                    format(val_cost, val_f1_mic, val_f1_mac))
    return trainingTimeTotal, avg_time, total_steps, val_cost, val_f1_mic, val_f1_mac, duration

def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix, node_remove_proportion=FLAGS.dataset_size_ratio, feat_remove_proportion=FLAGS.feature_size_ratio)
    print("Done loading training data..")
    output = train(train_data)
    print('trainingTimeTotal, avg_time, total_steps, val_cost, val_f1_mic, val_f1_mac, val duration')
    print(str(output))
    print ('Finished, quitting...')
    # known issue: nvprof may hang unless told explicitly to stop
    # nvprofStop()
    # nvprofReset()

if __name__ == '__main__':
    tf.app.run()

