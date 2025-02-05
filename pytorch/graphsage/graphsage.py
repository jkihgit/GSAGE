import torch
import torch.nn as nn
from torchvision import transforms

from torch.cuda.nvtx import range_push as nvtxPush
from torch.cuda.nvtx import range_pop as nvtxPop
from torch.cuda.nvtx import mark as nvtxTimestamp

class Sampler(nn.Module):
    def __init__(self, adj, maxdeg, samples1, samples2):
        super().__init__()
        self.adj = adj
        self.maxdeg = maxdeg
        self.samples1 = samples1
        self.samples2 = samples2

    def lookup(self, x, samples):
        adj_lists = self.adj(x)
        adj_lists = adj_lists[:, torch.randperm(self.maxdeg)]
        return torch.narrow(adj_lists, 1, 0, samples)

    def forward(self, x):
        nvtxPush('Samp/0')
        hop0 = self.lookup(x, self.samples2)
        nvtxPop()
        # does not write over hop0 value
        nvtxPush('Samp/1')
        hop1 = self.lookup(hop0.flatten(), self.samples1)
        nvtxPop()
        return hop0, hop1

class AggregatorL1(nn.Module):
    def __init__(self, features, fvlen, hdim):
        super().__init__()
        self.features = features
        self.self = nn.Linear(fvlen, hdim)
        self.neigh = nn.Linear(fvlen, hdim)
        nn.init.xavier_uniform_(self.self.weight)
        nn.init.xavier_uniform_(self.neigh.weight)
        self.relu = nn.ReLU()

    def forward(self, x0, x1):
        nvtxPush('FeatFet')
        nvtxPush('FeatFet/Self')
        self_feats = self.features(x0)
        nvtxPop()
        nvtxPush('FeatFet/Neigh')
        neigh_feats = self.features(x1)
        nvtxPop()
        nvtxPush('FeatFet/Reduce')
        neigh_feats = neigh_feats.mean(dim=1)
        nvtxPop()
        nvtxPop()
        nvtxPush('FC')
        nvtxPush('FC/Self')
        self_out = self.self(self_feats)
        nvtxPop()
        nvtxPush('FC/Neigh')
        neigh_out = self.neigh(neigh_feats)
        nvtxPop()
        nvtxPop()
        nvtxPush('concat')
        out = torch.cat((self_out, neigh_out), dim=1)
        nvtxPop()
        nvtxPush('relu')
        retval = self.relu(out)
        nvtxPop()
        return retval

# same thing but without the feature lookup
class AggregatorL2(nn.Module):
    def __init__(self, hdim, samples2):
        super().__init__()
        self.self = nn.Linear(hdim*2, hdim)
        self.neigh = nn.Linear(hdim*2, hdim)
        nn.init.xavier_uniform_(self.self.weight)
        nn.init.xavier_uniform_(self.neigh.weight)
        self.hdim = hdim
        self.samples2 = samples2
        self.norm = nn.LayerNorm(hdim*2, eps=1e-12)

    def forward(self, x0, x1):
        nvtxPush('Reshape')
        nvtxPop()
        x1 = x1.reshape(-1, self.samples2, self.hdim*2)
        nvtxPush('Reduce')
        x1 = x1.mean(dim=1)
        nvtxPop()
        nvtxPush('FC')
        nvtxPush('FC/Self')
        self_out = self.self(x0)
        nvtxPop()
        nvtxPush('FC/Neigh')
        neigh_out = self.neigh(x1)
        nvtxPop()
        nvtxPop()
        nvtxPush('concat')
        out = torch.cat((self_out, neigh_out), dim=1)
        nvtxPop()
        nvtxPush('l2')
        retval = self.norm(out)
        nvtxPop()
        return retval

# hardcoded for 2 layer model
class GraphSAGE(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        num_nodes = dataset['adj'].shape[0]
        num_test_nodes = dataset['test_adj'].shape[0]
        fvlen = dataset['features'].shape[-1]
        maxdeg = dataset['test_adj'].shape[-1]
        hdim = args.num_hidden
        num_classes = dataset['val_labels'].shape[-1]

        self.features = nn.Embedding(num_nodes, fvlen)
        self.features.weight = nn.Parameter(torch.FloatTensor(dataset['features']), requires_grad=False)
        self.adj = nn.Embedding(num_nodes, maxdeg)
        self.adj.weight = nn.Parameter(torch.LongTensor(dataset['adj']), requires_grad=False)
        self.test_adj = nn.Embedding(num_test_nodes, maxdeg)
        self.test_adj.weight = nn.Parameter(torch.LongTensor(dataset['test_adj']), requires_grad=False)
        self.sampler = Sampler(self.adj, maxdeg, args.samples1, args.samples2)
        self.test_sampler = Sampler(self.test_adj, maxdeg, args.samples1, args.samples2)
        self.agg1A = AggregatorL1(self.features, fvlen, hdim)
        self.agg1B = AggregatorL1(self.features, fvlen, hdim)
        self.agg2 = AggregatorL2(hdim, args.samples2)
        self.cls = nn.Linear(hdim*2, num_classes)
        # no fan_avg init in pytorch
        nn.init.kaiming_uniform_(self.cls.weight)

    def forward(self, x):
        nvtxPush('Samp')
        hop0, hop1 = self.sampler(x)
        nvtxPop()
        # does not write over hop0
        nvtxPush('Agg')
        nvtxPush('Agg/0_0')
        featmap1 = self.agg1B.forward(hop0.reshape(-1), hop1)
        nvtxPop()
        nvtxPush('Agg/0_1')
        featmap0 = self.agg1A.forward(x, hop0)
        nvtxPop()
        nvtxPush('Agg/1')
        featmap = self.agg2.forward(featmap0, featmap1)
        nvtxPop()
        nvtxPop()
        nvtxPush('Cls')
        featmap = self.cls.forward(featmap)
        nvtxPop()
        return featmap

    def test_forward(self, x):
        hop0, hop1 = self.test_sampler(x)
        # does not write over hop0
        featmap1 = self.agg1B.forward(hop0.reshape(-1), hop1)
        featmap0 = self.agg1A.forward(x, hop0)
        featmap = self.agg2.forward(featmap0, featmap1)
        featmap = self.cls.forward(featmap)
        return featmap