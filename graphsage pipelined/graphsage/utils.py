from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
import pickle

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

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

def load_file(path, dtype=np.float32):
    with open(path, 'rb') as f:
        return np.array(pickle.load(f), dtype=dtype)

def load_from_file(prefix):
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
    return retval
    
def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")

        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")
        # print ("WARN: truncated FV")

        # feats = feats[:,:50]
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
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

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


if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
