from __future__ import division
from __future__ import print_function
import tensorflow as tf

import numpy as np
import math

np.random.seed(123)

class SeqGen(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        l = math.floor(len(self.x) / self.batch_size)
        self.x = self.x[:l*batch_size]
        self.y = self.y[:l*batch_size]
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array(batch_x), np.array(batch_y)    

class TFBatching(object):
    # moved over only training data
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx, 
            label_map, num_classes, 
            batch_size, max_degree, prefetch_depth=1, 
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.label_map = label_map
        self.num_classes = num_classes
        self.validation_batch_size = self.batch_size

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)

        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

        print (len(self.train_nodes), 'train nodes')
        print (len(self.test_nodes), 'test nodes')
        print (len(self.val_nodes), 'val nodes')
        print (len(self.train_nodes) + len(self.test_nodes) + len(self.val_nodes), 'nodes total')

        self.train_nodes, self.train_labels = self.processNodes(self.train_nodes)
        self.test_nodes, self.test_labels = self.processNodes(self.test_nodes)
        self.val_nodes, self.val_labels = self.processNodes(self.val_nodes)

        padding_shape = (tf.TensorShape([]), tf.TensorShape([self.num_classes]))

        self.ds = SeqGen(self.train_nodes, self.train_labels, self.batch_size)

        self.dsTest = SeqGen(self.test_nodes, self.test_labels, self.batch_size)

        self.dsVal = tf.data.Dataset.from_tensor_slices((self.val_nodes, self.val_labels))
        self.dsVal = self.dsVal.padded_batch(self.validation_batch_size, padding_shape, None, True)
        self.dsVal = self.dsVal.prefetch(buffer_size=prefetch_depth)

    def entire_training_set(self, trim_to_batch_size=True):
        return self.ds

    def entire_testing_set(self, trim_to_batch_size=True):
        return self.dsTest

    def entire_validation_set(self, trim_to_batch_size=True):
        return self.dsVal

    def processNodes(self, nodes):
        labels = [self._make_label_vec(n) for n in nodes]
        nodes = [self.id2idx[n] for n in nodes]

        while len(nodes) < self.batch_size:
            diff = self.batch_size - len(nodes)
            if diff > len(nodes):
                diff = len(nodes)
            nodes = nodes + nodes[:diff]
            labels = labels + labels[:diff]

        nodes = np.array(nodes, dtype=np.int32)
        labels = np.array(labels, dtype=np.float32)
        return nodes, labels 

    # batching
    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    # init
    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))


        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    # init
    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    # epoch reset
    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.ds = self.ds.shuffle(len(self.train_nodes))