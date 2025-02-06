import os
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.client import timeline
from datetime import datetime
# tf.get_logger().setLevel(3)
# 0-3, 3 the most silent
import numpy as np
import sklearn
from sklearn import metrics

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import TFBatching
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data, load_from_file
from graphsage.cuda_plugin import start as nvprofStart
from graphsage.cuda_plugin import stop as nvprofStop
from graphsage.cuda_plugin import reset as nvprofReset

import cProfile

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)


# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
flags.DEFINE_boolean('soft_device_placement', True, "Allow device placement to be ignored if necessary")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'number of epochs to train.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('dim_1', 512, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 512, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('validate_batch_size', 64, "how many nodes per validation sample.")
flags.DEFINE_boolean('sigmoid', True, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")

flags.DEFINE_integer('patience', -1, 'quit early if no validation improvement for this much epochs. -1 for off')
flags.DEFINE_integer('timeline', 0, "Chrome trace json")
flags.DEFINE_integer('tensorboard', 0, "Graph is bugged. Scalar reporting is not. Reloading the page may solve the graph bug. No srsly.")
flags.DEFINE_integer('minimini', 0, "Run single batch only")
flags.DEFINE_integer('noval', 0, "no validation")


gpus = tf.config.experimental.list_physical_devices('GPU')
batch_size = FLAGS.batch_size * len(gpus)
validate_batch_size = FLAGS.validate_batch_size * len(gpus)
print ('# of Found GPU:', len(gpus))
# why not use a flag for this?
# because the only reason you would enable this is to get VRAM use measurements
# also, remember to do this again in supervised_models.py
# for gpu in range(len(gpus)):
#     tf.config.experimental.set_memory_growth(gpus[gpu], True)
tf.config.set_soft_device_placement(FLAGS.soft_device_placement)
tf.debugging.set_log_device_placement(FLAGS.log_device_placement)
tf.keras.backend.set_floatx('float32')
strategy = None

print ('patience', FLAGS.patience)
print ('minimini', FLAGS.minimini)
print ('noval', FLAGS.noval) 

def train(train_data):
    prefetch_depth = 1
    if type(train_data) is dict:
        features = train_data['features']
        num_classes = len(train_data['val_labels'][0])
        minibatch = TFBatching('new', (train_data, prefetch_depth, batch_size, validate_batch_size, num_classes))
    else:
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

        minibatch = TFBatching('old', (G, 
                        id_map,
                        class_map,
                        num_classes,
                        batch_size,
                        validate_batch_size,
                        FLAGS.max_degree, 
                        prefetch_depth,
                        True,
                        features,
                        'pickled_data/'+FLAGS.train_prefix.split('/')[-1]+'/'))

    # train_adj_info = tf.random.uniform(tf.shape(minibatch.adj), 0, FLAGS.max_degree, dtype=tf.int32)
    # val_adj_info = train_adj_info
    train_adj_info = tf.Variable(minibatch.adj, trainable=False, name="adj_info_train", dtype=tf.float32)
    val_adj_info = tf.Variable(minibatch.test_adj, trainable=False, name="adj_info_val", dtype=tf.float32)
    # train_adj_info = minibatch.adj
    # val_adj_info = minibatch.test_adj

    if FLAGS.model == 'graphsage_mean':
        # Create model
        train_sampler = UniformNeighborSampler(train_adj_info)
        val_sampler = UniformNeighborSampler(val_adj_info)
        if FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", FLAGS.samples_1, FLAGS.dim_1)]

        legacy_var_list = dict()
        # each GPU gets FLAGS.BS, not BS
        legacy_var_list['batch_size'] = FLAGS.batch_size
        legacy_var_list['val_batch_size'] = FLAGS.validate_batch_size
        legacy_var_list['gpu_count'] = len(gpus)
        model = SupervisedGraphsage(num_classes, legacy_var_list, 
                                     features,
                                     train_sampler, val_sampler,
                                     -1, #degrees isnt used by GrapSAGE. Might have been used by GCN or something
                                     layer_infos, 
                                     minibatch,
                                     strategy = strategy,
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    else:
        raise Exception('Error: model name unrecognized.')

# Train model
    # actually using nvprofStart() option causes 10-20x slowdown
    # that shows NCCL is 99% of running time
    # so not only does it cause slowdown, it causes wrong measurements
    # even with 20x slowdown all caused by NCCL, NCCL would be ~97% of running time
    # use minimini and/or noval options instead
    # nvprofStart()
    if FLAGS.minimini:
        outs = model.minimini(1)
    else:
        outs = model.train()
    # nvprofStop()
    # nvprofReset()
    if FLAGS.timeline:
        tl = timeline.Timeline(step_stats=model.run_metadata.step_stats)
        logdir = "tblogs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(logdir+'.json', 'w') as f:
            f.write(tl.generate_chrome_trace_format())
    return outs

def prettyPrint(output):
    def times(d):
        newDict = dict()
        for key in d.keys():
            if key == 'name':
                continue
            newDict[d['name']+"_"+key] = d[key]
        return newDict
    if FLAGS.minimini:
        history, trainTimes = output
        trainTimes = times(trainTimes)
        mergedDicts = history.history
        for key in mergedDicts.keys():
            mergedDicts[key] = mergedDicts[key][-1]
        mergedDicts.update(trainTimes)
    else:
        history, s1, f1s, trainTimes, valTimes = output
        trainTimes = times(trainTimes)
        valTimes = times(valTimes)
        mergedDicts = history.history
        for key in mergedDicts.keys():
            mergedDicts[key] = mergedDicts[key][-1]
        mergedDicts.update(trainTimes)
        if not FLAGS.noval:
            mergedDicts.update(valTimes)
            mergedDicts['valF1Micro'] = f1s[0]
            mergedDicts['valF1Macro'] = f1s[1]
            mergedDicts['valLoss'] = s1[0]
            mergedDicts['valBinAcc'] = s1[1]
            mergedDicts['valCatAcc'] = s1[2]

    keys = [x for x in mergedDicts.keys()]
    keys.sort()

    maxlen = 0
    for key in keys:
        if len(key) > maxlen:
            maxlen = len(key)
    for key in keys:
        lendiff = maxlen - len(key)
        keydisp = key + ' '*lendiff
        print (keydisp, ':', mergedDicts[key])


def multiGPU(argv=None):
    global strategy
    print ('multi GPU mode')
    with tf.device('/cpu:0'):
        # train_data = load_data(FLAGS.train_prefix)
        train_data = load_from_file('pickled_data/'+FLAGS.train_prefix.split('/')[-1]+'/')
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    with strategy.scope():
        print("Loading training data..")
        print("Done loading training data..")
        return train(train_data)

def singleGPU():
    print ('single GPU mode')
    global gpus, batch_size
    if len(gpus) > 1:
        gpus=[gpus[0]]
    print (gpus)
    batch_size = FLAGS.batch_size
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    # train_data = load_from_file('pickled_data/'+FLAGS.train_prefix.split('/')[-1]+'/')
    print("Done loading training data..")
    return train(train_data)

if __name__ == '__main__':
    if len(gpus) > 1:
        output = multiGPU()
    else:
        output = multiGPU()
        # output = singleGPU()
    prettyPrint(output)

# def main():
#     if len(gpus) > 1:
#         output = multiGPU()
#     else:
#         output = multiGPU()
#         # output = singleGPU()
#     prettyPrint(output)
# if __name__=='__main__':
#     cProfile.run('main()')