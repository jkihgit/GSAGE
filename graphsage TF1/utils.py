from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
import random

import networkx as nx
from networkx.readwrite import json_graph, read_adjlist, read_edgelist
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50
REDUCE_TRAINING_DATA = False
REDUCE_TESTING_DATA = False
REDUCE_VALIDATION_DATA = False

# returns a 1D list size long containing [min, max) ints
def randInts(size, min, max):
    return [random.randint(min, max-1) for x in range(size)]

# returns a 1D list size long containing [min, max] floats
def randFloats(size, min, max):
    return [random.uniform(min, max) for x in range(size)]

def boolparse(boolstr):
    if str.isnumeric(str(boolstr)):
        return float(boolstr) != 0
    if len(boolstr):
        if boolstr.upper() == 'FALSE':
            return False
        elif boolstr.upper() == 'TRUE':
            return True
    raise ValueError('bool argument must be true or false not case sensitive.\nReceived '+boolstr)
    return False

# src1 dest1 {optional meta data as python dict}
# src2 dest3 {"e.g":"example", "edge_weight":10}

# Requires:
# -edgelist
# -feats
# -id_map
# -class_map
def load_edgelist(prefix, normalize=True, load_walks=False, node_remove_proportion=0.0, feat_remove_proportion=0.0, testset_ratio=0.1, validationset_ratio=0.1):
    G = read_edgelist(prefix + "-edgelist.txt")

    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy", encoding='latin1')
        feat_size = feats.shape[-1]
        print ('Reducing feature vector size from', feat_size)
        remove_this_much_feats = int(feat_size * feat_remove_proportion)
        feats = feats[:,remove_this_much_feats:]
        print ('to',feats.shape[-1], 'by removing the first', remove_this_much_feats, 'items in the feature vectors')
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    oldCount = len(G.nodes())
    trainingSetSize = 0
    testingSetSize = 0
    validationSetSize = 0
    broken_count = 0

    # set train test val nodes
    # which are not present in edgelist files
    i = 0
    test_cutoff = len(G.nodes())*(1-testset_ratio-validationset_ratio)
    val_cutoff = len(G.nodes())*(1-validationset_ratio)
    for n in G.nodes():
        if not (n in id_map.keys()):
            G.remove_node(n)
            broken_count += 1
        elif i < test_cutoff:
            G.node[n]['train'] = True
            G.node[n]['test']  = False
            G.node[n]['val']   = False
            trainingSetSize += 1
        elif i < val_cutoff:
            G.node[n]['train'] = False
            G.node[n]['test']  = True
            G.node[n]['val']   = False
            testingSetSize += 1
        else:
            G.node[n]['train'] = False
            G.node[n]['test']  = False
            G.node[n]['val']   = True
            validationSetSize += 1
        i += 1
    print (broken_count, 'nodes removed due to parsing issues (missing ID)')

    oldRemoveTrainingCount = remove_training_count = int(trainingSetSize * node_remove_proportion)
    oldRemoveTestingCount = remove_testing_count = int(testingSetSize * node_remove_proportion)
    oldRemoveValidationCount = remove_validation_count = int(validationSetSize * node_remove_proportion)
    if REDUCE_TRAINING_DATA:
        assert node_remove_proportion <= 0
        print ('Original training set size:', trainingSetSize)
        print ('Removing training nodes:', remove_training_count)
    else:
        print ('Training data reduction off. Set manually in utils.py')
    if REDUCE_TESTING_DATA:
        assert node_remove_proportion <= 0
        print ('Original testing set size:', testingSetSize)
        print ('Removing testing nodes:', remove_testing_count)
    else:
        print ('Testing data reduction off. Set manually in utils.py')
    if REDUCE_VALIDATION_DATA:
        assert node_remove_proportion <= 0
        print ('Original validation set size:', validationSetSize)
        print ('Removing validation nodes:', remove_validation_count)
    else:
        print ('Validation data reduction off. Set manually in utils.py')

    for node in G.nodes():
        if remove_training_count > 0 and isTrainingNode(node):
            G.remove_node(node)
            remove_training_count -= 1
        elif remove_testing_count > 0 and isTestingNode(node):
            G.remove_node(node)
            remove_testing_count -= 1
        elif remove_validation_count > 0 and isValidationNode(node):
            G.remove_node(node)
            remove_validation_count -= 1
        # else leave it in
    print ('Nodes left:', len(G.nodes()))
    assert remove_training_count < 1
    assert remove_testing_count < 1
    assert remove_validation_count < 1
    assert len(G.nodes()) == oldCount - broken_count - oldRemoveTrainingCount - oldRemoveTestingCount - oldRemoveValidationCount


    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    # normalize training set
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


# src1 dest1 dest2 ...
# src2 dest3 dest4 ...

# Requires:
# -adjlist
# -feats
# -id_map
# -class_map
def load_adj(prefix, normalize=True, load_walks=False, node_remove_proportion=0.0, feat_remove_proportion=0.0):
    def isTrainingNode(n):
        return (not G.node[n]['val']) and (not G.node[n]['test'])
    def isTestingNode(n):
        return G.node[n]['test']
    def isBrokenNode(n):
        return not 'val' in G.node[n] or not 'test' in G.node[n]
    def isValidationNode(n):
        return (not isTrainingNode(n)) and (not isTestingNode(n))

    G = read_adjlist(prefix + "-adjlist.txt")

    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
        feat_size = feats.shape[-1]
        print ('Reducing feature vector size from', feat_size)
        remove_this_much_feats = int(feat_size * feat_remove_proportion)
        feats = feats[:,remove_this_much_feats:]
        print ('to',feats.shape[-1], 'by removing the first', remove_this_much_feats, 'items in the feature vectors')
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for n in G.nodes():
        if isBrokenNode(n):
            G.remove_node(n)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    oldCount = len(G.nodes())
    trainingSetSize = 0
    testingSetSize = 0
    validationSetSize = 0
    for node in G.nodes():
        if isTrainingNode(node):
            trainingSetSize += 1
        elif isTestingNode(node):
            testingSetSize += 1
        elif isValidationNode(node):
            validationSetSize += 1
        else:
            # must be one of the above
            raise NotImplementedError
    oldRemoveTrainingCount = remove_training_count = int(trainingSetSize * node_remove_proportion)
    oldRemoveTestingCount = remove_testing_count = int(testingSetSize * node_remove_proportion)
    oldRemoveValidationCount = remove_validation_count = int(validationSetSize * node_remove_proportion)
    if REDUCE_TRAINING_DATA:
        assert node_remove_proportion <= 0
        print ('Original training set size:', trainingSetSize)
        print ('Removing training nodes:', remove_training_count)
    else:
        print ('Training data reduction off. Set manually in utils.py')
    if REDUCE_TESTING_DATA:
        assert node_remove_proportion <= 0
        print ('Original testing set size:', testingSetSize)
        print ('Removing testing nodes:', remove_testing_count)
    else:
        print ('Testing data reduction off. Set manually in utils.py')
    if REDUCE_VALIDATION_DATA:
        assert node_remove_proportion <= 0
        print ('Original validation set size:', validationSetSize)
        print ('Removing validation nodes:', remove_validation_count)
    else:
        print ('Validation data reduction off. Set manually in utils.py')

    for node in G.nodes():
        if remove_training_count > 0 and isTrainingNode(node):
            G.remove_node(node)
            remove_training_count -= 1
        elif remove_testing_count > 0 and isTestingNode(node):
            G.remove_node(node)
            remove_testing_count -= 1
        elif remove_validation_count > 0 and isValidationNode(node):
            G.remove_node(node)
            remove_validation_count -= 1
        # else leave it in
    print ('Nodes left:', len(G.nodes()))
    assert remove_training_count < 1
    assert remove_testing_count < 1
    assert remove_validation_count < 1
    assert len(G.nodes()) == oldCount - oldRemoveTrainingCount - broken_count - oldRemoveTestingCount - oldRemoveValidationCount


    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    # normalize training set
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


# Requires
# -G
# -feats
# -id_map
# -class_map
def load_json(prefix, normalize=True, load_walks=False, node_remove_proportion=0.0, feat_remove_proportion=0.0):
    def isTrainingNode(n):
        return (not G.node[n]['val']) and (not G.node[n]['test'])
    def isTestingNode(n):
        return G.node[n]['test']
    def isBrokenNode(n):
        return not 'val' in G.node[n] or not 'test' in G.node[n]
    def isValidationNode(n):
        return (not isTrainingNode(n)) and (not isTestingNode(n))


    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
        feat_size = feats.shape[-1]
        print ('Reducing feature vector size from', feat_size)
        remove_this_much_feats = int(feat_size * feat_remove_proportion)
        feats = feats[:,remove_this_much_feats:]
        print ('to',feats.shape[-1], 'by removing the first', remove_this_much_feats, 'items in the feature vectors')
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for n in G.nodes():
        if isBrokenNode(n):
            G.remove_node(n)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    oldCount = len(G.nodes())
    trainingSetSize = 0
    testingSetSize = 0
    validationSetSize = 0
    for node in G.nodes():
        if isTrainingNode(node):
            trainingSetSize += 1
        elif isTestingNode(node):
            testingSetSize += 1
        elif isValidationNode(node):
            validationSetSize += 1
        else:
            # must be one of the above
            raise NotImplementedError
    oldRemoveTrainingCount = remove_training_count = int(trainingSetSize * node_remove_proportion)
    oldRemoveTestingCount = remove_testing_count = int(testingSetSize * node_remove_proportion)
    oldRemoveValidationCount = remove_validation_count = int(validationSetSize * node_remove_proportion)
    if REDUCE_TRAINING_DATA:
        assert node_remove_proportion <= 0
        print ('Original training set size:', trainingSetSize)
        print ('Removing training nodes:', remove_training_count)
    else:
        print ('Training data reduction off. Set manually in utils.py')
    if REDUCE_TESTING_DATA:
        assert node_remove_proportion <= 0
        print ('Original testing set size:', testingSetSize)
        print ('Removing testing nodes:', remove_testing_count)
    else:
        print ('Testing data reduction off. Set manually in utils.py')
    if REDUCE_VALIDATION_DATA:
        assert node_remove_proportion <= 0
        print ('Original validation set size:', validationSetSize)
        print ('Removing validation nodes:', remove_validation_count)
    else:
        print ('Validation data reduction off. Set manually in utils.py')

    for node in G.nodes():
        if remove_training_count > 0 and isTrainingNode(node):
            G.remove_node(node)
            remove_training_count -= 1
        elif remove_testing_count > 0 and isTestingNode(node):
            G.remove_node(node)
            remove_testing_count -= 1
        elif remove_validation_count > 0 and isValidationNode(node):
            G.remove_node(node)
            remove_validation_count -= 1
        # else leave it in
    print ('Nodes left:', len(G.nodes()))
    assert remove_training_count < 1
    assert remove_testing_count < 1
    assert remove_validation_count < 1
    assert len(G.nodes()) == oldCount - oldRemoveTrainingCount - broken_count - oldRemoveTestingCount - oldRemoveValidationCount


    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    # normalize training set
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
