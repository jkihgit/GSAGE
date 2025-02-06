from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx, 
            label_map, num_classes, 
            batch_size, max_degree,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

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

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label, dtype=np.float32)
        else:
            label_vec = np.zeros((self.num_classes), dtype=np.float32)
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

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

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def trim_to_size(self, data, segment_size=None):
        print ('===============================')
        print ('orig:', len(data))
        if segment_size == None:
            segment_size = self.batch_size
        if len(data) >= segment_size:
            section_A = (len(data))//segment_size
            data = data[:int(section_A*segment_size)]
        else:
            while len(data) < segment_size:
                data = np.append(data, data, axis=0)
            data = data[:segment_size]
        print ('out:', len(data))
        print ('===============================')
        return data

    def entire_training_set(self, trim_to_batch_size=True):
        data, labels = self.batch_feed_dict(self.train_nodes)
        if trim_to_batch_size:
            data = self.trim_to_size(data)
            labels = self.trim_to_size(labels)
        print ('===============================')
        print ('TRN data:', data.shape)
        print ('TRN labels:', labels.shape)
        print ('===============================')

        assert len(labels) >= self.batch_size
        return data, labels

    def entire_testing_set(self, trim_to_batch_size=True):
        data, labels = self.batch_feed_dict(self.test_nodes)
        if trim_to_batch_size:
            data = self.trim_to_size(data)
            labels = self.trim_to_size(labels)
        print ('===============================')
        print ('TST data:', data.shape)
        print ('TST labels:', labels.shape)
        print ('===============================')
        # assert len(labels) >= self.batch_size
        return data, labels

    def entire_validation_set(self, trim_to_batch_size=True):
        data, labels = self.batch_feed_dict(self.val_nodes)
        if trim_to_batch_size:
            data = self.trim_to_size(data)
            labels = self.trim_to_size(labels)
        print ('===============================')
        print ('V data:', data.shape)
        print ('V labels:', labels.shape)
        print ('===============================')
        # assert len(labels) >= self.batch_size
        return data, labels

    def batch_feed_dict(self, batch_nodes):
        batch1id = batch_nodes
        batch1 = np.array([self.id2idx[n] for n in batch1id], dtype=np.int32)
              
        labels = np.vstack([self._make_label_vec(node) for node in batch1id])

        return batch1, labels

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

