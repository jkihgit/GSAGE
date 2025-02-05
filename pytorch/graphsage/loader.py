import numpy as np
import random
import pickle
from collections import defaultdict

reddit_dir = "reddit/"
ppi_dir = "ppi/"
pubmed_dir = "pubmed/"
synth_dir = "synth/"

def load_file(path, dtype=np.float32):
    with open(path, 'rb') as f:
        return np.array(pickle.load(f), dtype=dtype)

def reduce_classes(labels, num_classes):
    retval = []
    for i in range(len(labels)):
        cls = np.argmax(labels[i])
        cls %= num_classes
        line = np.zeros(num_classes)
        line[cls] = 1
        retval.append(line)
    return np.array(retval)

def load_from_file(prefix, num_classes):
    retval = dict()
    retval['train_nodes'] = load_file(prefix+'train_nodes.pkl', np.int32)
    retval['train_labels'] = load_file(prefix+'train_labels.pkl', np.int32)
    retval['test_nodes'] = load_file(prefix+'test_nodes.pkl', np.int32)
    retval['test_labels'] = load_file(prefix+'test_labels.pkl', np.int32)
    retval['val_nodes'] = load_file(prefix+'val_nodes.pkl', np.int32)
    retval['val_labels'] = load_file(prefix+'val_labels.pkl', np.int32)
    retval['features'] = load_file(prefix+'features.pkl')
    retval['adj'] = load_file(prefix+'adj.pkl', np.int32)
    retval['test_adj'] = load_file(prefix+'test_adj.pkl', np.int32)

    if num_classes > 0:
        old_classes = len(retval['val_labels'][0])
        if old_classes < num_classes:
            raise NotImplemented
        if old_classes == num_classes:
            return retval
        print ('Overriding existing num_classes from', old_classes, 'to', num_classes)
        retval['train_labels'] = reduce_classes(retval['train_labels'], num_classes)
        retval['test_labels'] = reduce_classes(retval['test_labels'], num_classes)
        retval['val_labels'] = reduce_classes(retval['val_labels'], num_classes)
        assert len(retval['val_labels'][0]) == num_classes

    return retval

def load_reddit(prefix, num_classes):
    return load_from_file(prefix+reddit_dir, num_classes)

def load_ppi(prefix, num_classes):
    return load_from_file(prefix+ppi_dir, num_classes)

def load_pubmed(prefix, num_classes):
    return load_from_file(prefix+pubmed_dir, num_classes)

# num_nodes, feat_size, max_deg max_deg not implemented
# use gen_synth instead
def load_synth(prefix, synth_args):
    num_nodes, feat_size, num_classes, avg_deg, max_deg = synth_args
    prefix += synth_dir

    retval = dict()
    retval['train_labels'] = np.zeros((num_nodes, num_classes), dtype=np.int32)
    # load parent set
    retval['train_nodes'] = load_file(prefix+'train_nodes.pkl', np.int32)
    train_labels = load_file(prefix+'train_labels.pkl', np.int32)
    retval['features'] = load_file(prefix+'features.pkl')
    retval['adj'] = load_file(prefix+'adj'+str(avg_deg)+'.pkl', np.int32)

    # print ('get adj')
    # s = 0
    # for adj in retval['adj']:
    #     i = len(set(adj))
    #     print (i)
    #     s += i
    # print ('___________')
    # print (s / num_nodes)

    # "cut" num classes by lumping together classes
    # assumes sample size balance is not affected by this
    train_labels %= num_classes
    
    # expand labels to be one-hot vectors
    # only supports single labels for now (as opposed to multi label)
    for i in range(num_nodes):
        retval['train_labels'][i][train_labels[i]] = 1

    retval['val_labels'] = retval['train_labels']
    retval['test_adj'] = retval['adj']
    return retval