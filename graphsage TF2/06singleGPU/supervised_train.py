from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import sklearn
from sklearn import metrics

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import TFBatching
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data, boolparse

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
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('dim_1', 512, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 512, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', True, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

flags.DEFINE_string('minimini', "False", "Run only 1 batch")
flags.DEFINE_integer('tensorboard', 0, "Graph is bugged. Scalar reporting is not.")

minimini = boolparse(FLAGS.minimini)
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) == 1
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_soft_device_placement(FLAGS.soft_device_placement)
tf.keras.backend.set_floatx('float32')

def train(train_data, test_data=None):

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
    minibatch = TFBatching(G, 
            id_map,
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs)

    # train_adj_info = tf.Variable(minibatch.adj, trainable=False, name="adj_info_train", dtype=tf.int32)
    # val_adj_info = tf.Variable(minibatch.test_adj, trainable=False, name="adj_info_val", dtype=tf.int32)
    train_adj_info = minibatch.adj
    val_adj_info = minibatch.test_adj

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(train_adj_info)
        if FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        legacy_var_list = dict()
        legacy_var_list['batch_size'] = FLAGS.batch_size
        model = SupervisedGraphsage(num_classes, legacy_var_list, 
                                     features,
                                     train_adj_info, val_adj_info,
                                     minibatch.deg,
                                     layer_infos, 
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    else:
        raise Exception('Error: model name unrecognized.')
    
    # Train model
    
    trnds = minibatch.entire_training_set()
    tstds = minibatch.entire_testing_set()
    vds = minibatch.entire_validation_set()
    # TODO: fix train time
    #       this includes val time
    outs = model.train(trnds, tstds, vds)
    
def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    train(train_data)

if __name__ == '__main__':
    main()
