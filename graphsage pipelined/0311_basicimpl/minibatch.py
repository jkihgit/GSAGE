from __future__ import division
from __future__ import print_function
import tensorflow as tf
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

import numpy as np
import pickle

np.random.seed(123)
num_threads = 16


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
    def __init__(self, init_type, inputs):
        def fillTemplate(template):
            i = 0
            for rank in template:
                for f in range(len(rank)):
                    rank[f] = i
                i += 1
            return template
        # only works for 2 layered model
        # should be easy to expend to any number of layers
        ranktemplate = [0,0]
        ranktemplate[1] = np.zeros((FLAGS.batch_size, FLAGS.samples_2, 1), dtype=float)
        ranktemplate[0] = np.zeros((FLAGS.batch_size*FLAGS.samples_2, FLAGS.samples_1, 1), dtype=float)
        for template in ranktemplate:
            template = fillTemplate(template)
        self.ranktemplate = ranktemplate

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

    def init_new(self, inputs):
        data_dict, prefetch_depth, batch_size, validation_batch_size, num_classes = inputs
        self.train_nodes = data_dict['train_nodes']
        self.train_labels = data_dict['train_labels']
        self.test_nodes = data_dict['test_nodes']
        self.test_labels = data_dict['test_labels']
        self.val_nodes = data_dict['val_nodes']
        self.val_labels = data_dict['val_labels']
        self.adj = tf.cast(data_dict['adj'], tf.int32)
        self.test_adj = tf.cast(data_dict['test_adj'], tf.int32)
        self.features = data_dict['features']
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.validation_batch_size = validation_batch_size
        self.init(prefetch_depth)

    def init(self, prefetch_depth):
        self.fvsize = len(self.features[0])
        def fetchAdj(nodeIDs, hop, adj, name):
            # this is likely loading from GPU to CPU
            adj_lists = tf.nn.embedding_lookup(adj, nodeIDs, name='sampling_'+name)
            adj_lists = tf.transpose(a=adj_lists) #transpose1
            adj_lists = tf.random.shuffle(adj_lists)
            adj_lists = tf.transpose(a=adj_lists) #transpose2
            shape = [-1, FLAGS.samples_2]
            if hop == 1:
                shape = [-1, FLAGS.samples_2, FLAGS.samples_1]
            adj_lists = tf.slice(adj_lists, [0]*(hop+2), shape)
            return adj_lists
        def fetchBatchFeatures(nodeIDs, labels, adj):
            # input: a batch of IDs and labels (both int32)
            # 1) fetch adj vectors
            # 2) fetch random portion 
            # 3) fetch features

            fvs0 = tf.nn.embedding_lookup(self.features, nodeIDs, name='aggemb_0')
            # (BS)x(FVLEN)
            adjlists1 = fetchAdj(nodeIDs, 0, adj, '1')
            fvs1 = tf.nn.embedding_lookup(self.features, adjlists1, name='aggemb_1')
            # (BS)x(SAMPLES2)x(FVLEN)
            adjlists2 = fetchAdj(adjlists1, 1, adj, '2')
            fvs2 = tf.nn.embedding_lookup(self.features, adjlists2, name='aggemb_2')
            # (BS)x(SAMPLES2)x(SAMPLES1)x(FVLEN)
            fvs2 = tf.reduce_sum(fvs2, -2)
            # (BS)x(SAMPLES2)x(FVLEN)

            # return (BS)x261x(FVLEN) FV matrix
            fvs0 = tf.reshape(fvs0, (self.batch_size, 1, -1))
            fvs = tf.concat((fvs0, fvs1, fvs2), 1)
            
            return fvs, labels

        def dsmap(nodeIDs, labels):
            # with tf.device('device:XLA_CPU:0'):
            return fetchBatchFeatures(nodeIDs, labels, self.adj)
        def testdsmap(nodeIDs, labels):
            # with tf.device('device:XLA_CPU:0'):
            return fetchBatchFeatures(nodeIDs, labels, self.test_adj)

        padding_shape = (tf.TensorShape([]), tf.TensorShape([self.num_classes]))

        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard = False

        self.ds = tf.data.Dataset.from_tensor_slices((self.train_nodes, self.train_labels))
        self.ds = self.ds.batch(self.batch_size, drop_remainder=True)
        adj = self.adj
        self.ds = self.ds.map(dsmap, num_parallel_calls=num_threads)
        # self.ds = self.ds.padded_batch(self.batch_size, padding_shape, None, True)
        self.ds = self.ds.prefetch(buffer_size=prefetch_depth)
        # self.ds = self.ds.with_options(options)

        # for some reason keras.model.fit() shuffle AFTER each epoch
        # and complains that I didnt shuffle the dataset before feeding it
        self.shuffle()

        self.dsTest = tf.data.Dataset.from_tensor_slices((self.test_nodes, self.test_labels))
        self.dsTest = self.dsTest.padded_batch(self.batch_size, padding_shape, None, True)
        # self.dsTest = self.dsTest.padded_batch(self.batch_size, padding_shape, None, True)
        self.dsTest = self.dsTest.map(dsmap, num_parallel_calls=num_threads)
        self.dsTest = self.dsTest.prefetch(buffer_size=prefetch_depth)
        # self.dsTest = self.dsTest.with_options(options)

        self.dsVal = tf.data.Dataset.from_tensor_slices((self.val_nodes, self.val_labels))
        self.dsVal = self.dsVal.padded_batch(self.validation_batch_size, padding_shape, None, True)
        # self.dsVal = self.dsVal.padded_batch(self.validation_batch_size, padding_shape, None, True)
        self.dsVal = self.dsVal.map(testdsmap, num_parallel_calls=num_threads)
        self.dsVal = self.dsVal.prefetch(buffer_size=prefetch_depth)
        # self.dsVal = self.dsVal.with_options(options)

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
        labels = np.array(labels, dtype=np.int32)
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