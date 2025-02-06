from __future__ import division
from __future__ import print_function
import tensorflow as tf

import numpy as np
import pickle

np.random.seed(123)

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
    features used to save everything at once. Not actually used anywhere.
    """
    def __init__(self, init_type, inputs, linfos):
        self.layer_infos = linfos
        if init_type == 'old':
            self.init_old(inputs)
        elif init_type == 'new':
            self.init_new(inputs)
        else:
            raise NotImplementedError

    def init_old(self, inputs):
        G, id2idx, label_map, num_classes, batch_size, validation_batch_size, max_degree, prefetch_depth, save_result, features, save_path = inputs
        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.label_map = label_map
        self.num_classes = num_classes
        self.validation_batch_size = validation_batch_size

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)

        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

        # print ("WARN: REMOVING TRAINING NODES")
        # print ("WARN: REMOVING TRAINING NODES")
        # print ("WARN: REMOVING TRAINING NODES")
        # print ("WARN: REMOVING TRAINING NODES")
        # print ("WARN: REMOVING TRAINING NODES")
        # print ("WARN: REMOVING TRAINING NODES")
        # print ("WARN: REMOVING TRAINING NODES")
        # print ("WARN: REMOVING TRAINING NODES")
        # print ("WARN: REMOVING TRAINING NODES")
        # self.train_nodes = self.train_nodes[:512*8*2]

        print (len(self.train_nodes), 'train nodes')
        print (len(self.test_nodes), 'test nodes')
        print (len(self.val_nodes), 'val nodes')
        print (len(self.train_nodes) + len(self.test_nodes) + len(self.val_nodes), 'nodes total')

        self.train_nodes, self.train_labels = self.processNodes(self.train_nodes)
        self.test_nodes, self.test_labels = self.processNodes(self.test_nodes)
        self.val_nodes, self.val_labels = self.processNodes(self.val_nodes)

        self.init(prefetch_depth)

        if save_result:
            assert len(features) > 0
            self.save(self.train_nodes, save_path+'train_nodes.pkl')
            self.save(self.train_labels, save_path+'train_labels.pkl')
            self.save(self.test_nodes, save_path+'test_nodes.pkl')
            self.save(self.test_labels, save_path+'test_labels.pkl')
            self.save(self.val_nodes, save_path+'val_nodes.pkl')
            self.save(self.val_labels, save_path+'val_labels.pkl')
            self.save(features, save_path+'features.pkl')
            self.save(self.adj, save_path+'adj.pkl')
            self.save(self.test_adj, save_path+'test_adj.pkl')
            print ('save successful, exiting.')
            import sys
            sys.exit()

    def save(self, data, path):
        print ('saving', path)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def preproc(self, inputs):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(self.layer_infos)):
            t = len(self.layer_infos) - k - 1
            support_size *= self.layer_infos[t].num_samples
            sampler = self.layer_infos[t].neigh_sampler
            node = sampler((samples[k], self.layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * self.batch_size,]))
            support_sizes.append(support_size)
        hidden = [tf.nn.embedding_lookup(params=self.features, ids=tf.dtypes.cast(x, tf.int32)) for x in samples]
        hidden = tf.reshape(hidden, [-1])
        return hidden

    def init_new(self, inputs):
        data_dict, prefetch_depth, batch_size, validation_batch_size, num_classes = inputs
        self.features = data_dict['features']
        self.train_nodes = data_dict['train_nodes']
        self.train_labels = data_dict['train_labels']
        self.test_nodes = data_dict['test_nodes']
        self.test_labels = data_dict['test_labels']
        self.val_nodes = data_dict['val_nodes']
        self.val_labels = data_dict['val_labels']
        self.adj = data_dict['adj']
        self.test_adj = data_dict['test_adj']
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.validation_batch_size = validation_batch_size
        self.init(prefetch_depth)

    def init(self, prefetch_depth):
        self.codeTestX = np.array(self.train_nodes[:self.batch_size])
        self.codeTestY = np.array(self.train_labels[:self.batch_size])
        print ('batch input shape:', self.codeTestX.shape)
        print ('batch label shape:', self.codeTestY.shape)

        padding_shape = (tf.TensorShape([]), tf.TensorShape([self.num_classes]))

        self.ds = tf.data.Dataset.from_tensor_slices((self.train_nodes, self.train_labels))
        self.ds = self.ds.map(self.preproc)
        self.ds = self.ds.batch(self.batch_size, True)
        # self.ds = self.ds.padded_batch(self.batch_size, padding_shape, None, True)
        self.ds = self.ds.prefetch(buffer_size=prefetch_depth)

        # for some reason keras.model.fit() shuffle AFTER each epoch
        # and complains that I didnt shuffle the dataset before feeding it
        self.shuffle()

        self.dsTest = tf.data.Dataset.from_tensor_slices((self.test_nodes, self.test_labels))
        self.dsTest = self.dsTest.padded_batch(self.batch_size, padding_shape, None, True)
        # self.dsTest = self.dsTest.padded_batch(self.batch_size, padding_shape, None, True)
        self.dsTest = self.dsTest.prefetch(buffer_size=prefetch_depth)

        self.dsVal = tf.data.Dataset.from_tensor_slices((self.val_nodes, self.val_labels))
        self.dsVal = self.dsVal.padded_batch(self.validation_batch_size, padding_shape, None, True)
        # self.dsVal = self.dsVal.padded_batch(self.validation_batch_size, padding_shape, None, True)
        self.dsVal = self.dsVal.prefetch(buffer_size=prefetch_depth)


    def entire_training_set(self):
        return self.ds

    def entire_testing_set(self):
        return self.dsTest

    def entire_validation_set(self):
        return self.dsVal

    def raw_testing_set(self):
        return self.test_nodes, self.test_labels

    def raw_validation_set(self):
        return self.val_nodes, self.val_labels

    def processNodes(self, nodes, pad=True):
        labels = [self._make_label_vec(n) for n in nodes]
        nodes = [self.id2idx[n] for n in nodes]        

        if pad:
            while (len(nodes) < self.batch_size) or (len(nodes) % self.batch_size > 0):
                diff1 = self.batch_size - len(nodes)
                diff2 = self.batch_size - (len(nodes) % self.batch_size)
                diff = max((diff1, diff2))
                print (len(nodes), self.batch_size, len(nodes)%self.batch_size, diff1, diff2, max((diff1, diff2)))
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